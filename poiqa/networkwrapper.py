# the former learn.py + trainer.py wrapped into a class
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import copy

from .config import override_model_args
from .data import KnowledgeBase
from .MemNN import *
from .mdn import mdn_loss_function, psuedo_seek_mode,get_max_mode
from .networks import BOW_MDN, AttnMDN, Regression
from .metrics import compute_dis,compute_dis_haversine


logger = logging.getLogger(__name__)


class NetworkWrapper(object):
    # --------------------------------------------------------------------------
    # Initialization
    # --------------------------------------------------------------------------


    def __init__(self, args, word_dict, feature_dict,
                 state_dict=None, ):
        # Book-keeping.
        self.args = args
        self.word_dict = word_dict
        self.args.vocab_size = len(word_dict)
        self.feature_dict = feature_dict
        self.args.num_features = len(feature_dict)
        self.updates = 0
        self.use_cuda = False
        self.parallel = False
        self.loss_function = mdn_loss_function
        # building the network (without kb info)
        if args.network == 'mem' or args.network == 'mem-bow' or args.network == 'mem-attn':
            self.network = MemNN(self.args)
        elif args.network == 'bow-mdn':
            self.network = BOW_MDN(self.args)
        elif args.network == 'attn-mdn':
            self.network = AttnMDN(self.args)
        elif args.network == 'regression':
            self.network = Regression(self.args)
            self.loss_function = nn.MSELoss()

        # Load saved state
        if state_dict:
            # Load buffer separately
            if 'fixed_embedding' in state_dict:
                fixed_embedding = state_dict.pop('fixed_embedding')
                self.network.load_state_dict(state_dict)
                self.network.register_buffer('fixed_embedding', fixed_embedding)
            else:
                # load only the parameters that are defined upon initialization
                pretrained_dict = {k: v for k, v in state_dict.items() if k in self.network.state_dict()}
                # self.network.update(pretrained_dict)
                self.network.load_state_dict(pretrained_dict)
                if args.network in ['mem', 'mem-bow', 'mem-attn']:
                    # mem
                    self.network.mem = nn.Parameter(state_dict['mem'])
                    self.network.value_mean = nn.Parameter(state_dict['value_mean'], requires_grad=False)
                    self.network.value_dev = nn.Parameter(state_dict['value_dev'], requires_grad=True)

    def init_kb(self, kb):
        kb_data = KnowledgeBase(kb, self)

        kb_vec = kb_data.batchify()
        if self.use_cuda:
            kb_vec = [autograd.Variable(e).cuda() for e in kb_vec]
        else:
            kb_vec = [autograd.Variable(e) for e in kb_vec]
        self.network.init_memory(kb_vec)

        # self.network.init_regions(regions)
        return

    def expand_dictionary(self, words):
        """Add words to the DocReader dictionary if they do not exist. The
        underlying embedding matrix is also expanded (with random embeddings).

        Args:
            words: iterable of tokens to add to the dictionary.
        Output:
            added: set of tokens that were added.
        """
        to_add = {self.word_dict.normalize(w) for w in words
                  if w not in self.word_dict}

        # Add words to dictionary and expand embedding layer
        if len(to_add) > 0:
            logger.info('Adding %d new words to dictionary...' % len(to_add))
            for w in to_add:
                self.word_dict.add(w)
            self.args.vocab_size = len(self.word_dict)
            logger.info('New vocab size: %d' % len(self.word_dict))

            old_embedding = self.network.embedding.weight.data
            self.network.embedding = torch.nn.Embedding(self.args.vocab_size,
                                                        self.args.embedding_dim,
                                                        padding_idx=0)
            new_embedding = self.network.embedding.weight.data
            new_embedding[:old_embedding.size(0)] = old_embedding

        # Return added words
        return to_add

    def load_embeddings(self, words, embedding_file):
        """Load pretrained embeddings for a given list of words, if they exist.

        Args:
            words: iterable of tokens. Only those that are indexed in the
              dictionary are kept.
            embedding_file: path to text file of embeddings, space separated.
        """
        if not hasattr(self.network,'embedding'):
            logger.info('No embedding used')
            return
        words = {w for w in words if w in self.word_dict}
        logger.info('Loading pre-trained embeddings for %d words from %s' %
                    (len(words), embedding_file))
        embedding = self.network.embedding.weight.data

        # When normalized, some words are duplicated. (Average the embeddings).
        vec_counts = {}
        with open(embedding_file) as f:
            for line in f:
                parsed = line.rstrip().split(' ')
                assert (len(parsed) == embedding.size(1) + 1)
                w = self.word_dict.normalize(parsed[0])
                if w in words:
                    vec = torch.Tensor([float(i) for i in parsed[1:]])
                    if w not in vec_counts:
                        vec_counts[w] = 1
                        embedding[self.word_dict[w]].copy_(vec)
                    else:
                        logging.warning(
                            'WARN: Duplicate embedding found for %s' % w
                        )
                        vec_counts[w] = vec_counts[w] + 1
                        embedding[self.word_dict[w]].add_(vec)

        for w, c in vec_counts.items():
            embedding[self.word_dict[w]].div_(c)

        logger.info('Loaded %d embeddings (%.2f%%)' %
                    (len(vec_counts), 100 * len(vec_counts) / len(words)))

    def tune_embeddings(self, words):
        """Unfix the embeddings of a list of words. This is only relevant if
        only some of the embeddings are being tuned (tune_partial = N).

        Shuffles the N specified words to the front of the dictionary, and saves
        the original vectors of the other N + 1:vocab words in a fixed buffer.

        Args:
            words: iterable of tokens contained in dictionary.
        """
        words = {w for w in words if w in self.word_dict}

        if len(words) == 0:
            logger.warning('Tried to tune embeddings, but no words given!')
            return

        if len(words) == len(self.word_dict):
            logger.warning('Tuning ALL embeddings in dictionary')
            return

        # Shuffle words and vectors
        embedding = self.network.embedding.weight.data
        for idx, swap_word in enumerate(words, self.word_dict.START):
            # Get current word + embedding for this index
            curr_word = self.word_dict[idx]
            curr_emb = embedding[idx].clone()
            old_idx = self.word_dict[swap_word]

            # Swap embeddings + dictionary indices
            embedding[idx].copy_(embedding[old_idx])
            embedding[old_idx].copy_(curr_emb)
            self.word_dict[swap_word] = idx
            self.word_dict[idx] = swap_word
            self.word_dict[curr_word] = old_idx
            self.word_dict[old_idx] = curr_word

        # Save the original, fixed embeddings
        self.network.register_buffer(
            'fixed_embedding', embedding[idx + 1:].clone()
        )

    def init_optimizer(self, state_dict=None):
        """Initialize an optimizer for the free parameters of the network.

        Args:
            state_dict: network parameters
        """
        if self.args.fix_embeddings:
            for p in self.network.embedding.parameters():
                p.requires_grad = False
        parameters = [p for p in self.network.parameters() if p.requires_grad]
        if self.args.optimizer == 'sgd':
            self.optimizer = optim.SGD(parameters, self.args.learning_rate,
                                       momentum=self.args.momentum,
                                       weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'adamax':
            self.optimizer = optim.Adamax(parameters,
                                          weight_decay=self.args.weight_decay)
        else:
            raise RuntimeError('Unsupported optimizer: %s' %
                               self.args.optimizer)



    def update(self, ex, mode):
        '''train/evaluate one batch of examples'''

        if mode == 'train':
            self.network.train()
        else:
            self.network.eval()
        self.network.zero_grad()
        if self.use_cuda:
            inputs = [e if e is None else autograd.Variable(e.cuda(async=True))
                      for e in ex[:3]]
            target = autograd.Variable(ex[3].cuda())
            idxs = ex[-1]
            self.network = self.network.cuda()
        else:
            inputs = [autograd.Variable(e) for e in ex[:3]]
            target = autograd.Variable(ex[3])
            idxs = ex[-1]

        # forward
        if self.args.network == 'regression':
            predicted = self.network.forward(inputs)
            loss = self.loss_function(predicted.transpose(0, 1), target)
            if mode == 'test':
                dis, acc = compute_dis_haversine(predicted.transpose(0,1), target)
                entropy = torch.zeros(target.data.shape[-1])
                return loss.data, dis.data, acc, entropy

        else:
            [out_pi, out_sigma, out_mu] = self.network.forward(inputs)  # batch* targetset_size

            # compute loss, gradients and update

            loss = self.loss_function(out_pi, out_sigma, out_mu, torch.squeeze(target))
            if mode == 'test':
                m, m_val = psuedo_seek_mode(out_pi, out_sigma, out_mu)
                # m_val batch*mode
                # normalize m_val
                val_normed = m_val / (torch.sum(m_val, dim=1).unsqueeze(dim=1).expand_as(m_val))
                entropy = -1 * torch.sum(val_normed * torch.log(val_normed), dim=1)
                predicted = get_max_mode(m,m_val)
                dis, acc = compute_dis_haversine(predicted,target)
                return loss.data, dis.data, acc,entropy.data
        if mode == 'train':
            loss.backward()
            torch.nn.utils.clip_grad_norm(self.network.parameters(), self.args.grad_clipping)
            self.optimizer.step()
            return loss.data

        return loss.data

    # --------------------------------------------------------------------------
    # Saving and loading
    # --------------------------------------------------------------------------

    def save(self, filename):
        state_dict = copy.copy(self.network.state_dict())
        if 'fixed_embedding' in state_dict:
            state_dict.pop('fixed_embedding')
        params = {
            'state_dict': state_dict,
            'word_dict': self.word_dict,
            'feature_dict': self.feature_dict,
            'args': self.args,
        }
        try:
            torch.save(params, filename)
        except BaseException:
            logger.warning('WARN: Saving failed... continuing anyway.')

    def checkpoint(self, filename, epoch):
        params = {
            'state_dict': self.network.state_dict(),
            'word_dict': self.word_dict,
            'feature_dict': self.feature_dict,
            'args': self.args,
            'epoch': epoch,
            'optimizer': self.optimizer.state_dict(),
        }
        try:
            torch.save(params, filename)
        except BaseException:
            logger.warning('WARN: Saving failed... continuing anyway.')

    @staticmethod
    def load(filename, new_args=None):
        logger.info('Loading model %s' % filename)
        saved_params = torch.load(
            filename, map_location=lambda storage, loc: storage
        )
        word_dict = saved_params['word_dict']
        feature_dict = saved_params['feature_dict']
        state_dict = saved_params['state_dict']
        args = saved_params['args']
        if new_args:
            args = override_model_args(args, new_args)
        return NetworkWrapper(args, word_dict, feature_dict, state_dict)

    @staticmethod
    def load_checkpoint(filename, args):
        logger.info('Loading model %s' % filename)
        saved_params = torch.load(
            filename, map_location=lambda storage, loc: storage
        )
        word_dict = saved_params['word_dict']
        feature_dict = saved_params['feature_dict']
        state_dict = saved_params['state_dict']
        epoch = saved_params['epoch']
        optimizer = saved_params['optimizer']
        args = saved_params['args']
        model = NetworkWrapper(args, word_dict, feature_dict, state_dict)
        model.init_optimizer(optimizer)
        return model, epoch

    # --------------------------------------------------------------------------
    # Runtime
    # --------------------------------------------------------------------------

    def cuda(self):
        self.use_cuda = True
        self.network = self.network.cuda()

    def cpu(self):
        self.use_cuda = False
        self.network = self.network.cpu()

    def parallelize(self):
        """Use data parallel to copy the model across several gpus.
        This will take all gpus visible with CUDA_VISIBLE_DEVICES.
        """
        self.parallel = True
        self.network = torch.nn.DataParallel(self.network)
