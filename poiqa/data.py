#python 3.6
import os 
from torch.utils.data import Dataset, sampler, DataLoader
from torch.utils.data.dataloader import default_collate
import torch

import numpy as np
import json
import unicodedata
import logging
import sys
from itertools import chain
logger = logging.getLogger(__name__)
# ------------------------------------------------------------------------------
# Dictionary class for tokens.
# ------------------------------------------------------------------------------


class Dictionary(object):
    NULL = '<NULL>'
    UNK = '<UNK>'
    START = 2

    @staticmethod
    def normalize(token):
        return unicodedata.normalize('NFD', token)

    def __init__(self):
        self.tok2ind = {self.NULL: 0, self.UNK: 1}
        self.ind2tok = {0: self.NULL, 1: self.UNK}

    def __len__(self):
        return len(self.tok2ind)

    def __iter__(self):
        return iter(self.tok2ind)

    def __contains__(self, key):
        if type(key) == int:
            return key in self.ind2tok
        elif type(key) == str:
            return self.normalize(key) in self.tok2ind

    def __getitem__(self, key):
        if type(key) == int:
            return self.ind2tok.get(key, self.UNK)
        if type(key) == str:
            return self.tok2ind.get(self.normalize(key),
                                    self.tok2ind.get(self.UNK))

    def __setitem__(self, key, item):
        if type(key) == int and type(item) == str:
            self.ind2tok[key] = item
        elif type(key) == str and type(item) == int:
            self.tok2ind[key] = item
        else:
            raise RuntimeError('Invalid (key, item) types.')

    def add(self, token):
        token = self.normalize(token)
        if token not in self.tok2ind:
            index = len(self.tok2ind)
            self.tok2ind[token] = index
            self.ind2tok[index] = token

    def tokens(self):
        """Get dictionary tokens.

        Return all the words indexed by this dictionary, except for special
        tokens.
        """
        tokens = [k for k in self.tok2ind.keys()
                  if k not in {'<NULL>', '<UNK>'}]
        return tokens

# --------------------------
# Dataset class 
# --------------------------


def toVector(ex,model):
    args = model.args 
    word_dict = model.word_dict
    feature_dict = model.feature_dict

    text = torch.LongTensor([word_dict[w] for w in ex['text']])
    # Create extra features vector, one hot encoding
    if len(feature_dict) > 0:
        features = torch.zeros(len(ex['text']), len(feature_dict))
    else:
        features = None

    # f_{token} (POS)
    if args.use_pos:
        for i, w in enumerate(ex['pos']):
            f = 'pos=%s' % w
            if f in feature_dict:
                features[i][feature_dict[f]] = 1.0
    loc = torch.FloatTensor([ex['lat_raw'],ex['lng_raw']])
    vec_ex = {'id':ex['id'],
    'text':text,
    'features':features,
    'loc':loc,

    }
    return vec_ex
def kbtoVector(ex,model,length):
    word_dict = model.word_dict

    word_list = ex['text']
    word_vec = torch.LongTensor([word_dict[w] for w in word_list])
    padding = torch.zeros((length-len(word_list))).long()
    text = torch.cat([word_vec,padding])
    loc = torch.FloatTensor([ex['lat_raw'],ex['lng_raw']])
    vec_ex = {
        'text':text,
        'loc':loc,
    }
    return vec_ex


def qa_collate(batch):
    #batch is a list of samples 
    ids = [ex['id'] for ex in batch]
    texts = [ex['text'] for ex in batch]
    features = [ex['features'] for ex in batch]
    locs = [ex['loc'] for ex in batch]

    # Batch documents and features
    max_length = max([t.size(0) for t in texts])

    x1 = torch.LongTensor(len(texts), max_length).zero_()
    x1_mask = torch.ByteTensor(len(texts), max_length).fill_(1)
    if features[0] is None:
        x1_f = None
    else:
        x1_f = torch.zeros(len(texts), max_length, features[0].size(1))
    for i, t in enumerate(texts):
        x1[i, :t.size(0)].copy_(t)
        x1_mask[i, :t.size(0)].fill_(0)
        if x1_f is not None:
            x1_f[i, :t.size(0)].copy_(features[i])


    cor = torch.stack(locs,dim=1)
    return x1, x1_f, x1_mask,cor,ids

def kb_collate(batch):
    # batch is a list of samples
    texts = [ex['text'] for ex in batch]
    values = [ex['loc'] for ex in batch]
    x1 = torch.stack(texts)
    y = torch.stack(values)  # concatenate label LongTensors
    return x1, y


class GeoTweetDataset(Dataset):
    def __init__(self,exs,model):
        '''
        filename: accepts json files
        transform: transforms can be composed
        '''
        self.messages = exs
        self.model = model
        self.dataset_size = len(self.messages)

    def __len__(self):
        return self.dataset_size

    def __getitem__(self,idx): #item in questions
        tweet = self.messages[idx]
        sample = toVector(tweet,self.model)
        return sample
    def TrainValidationSplit(self,ratio,shuffle=True,seed=None):
        '''

        :param ratio: (training ratio, validation ratio)
        :return:
        '''
        total = self.dataset_size
        indices = list(range(total))
        split_train = int(np.floor(ratio * total))

        if shuffle == True:
            if seed:
                np.random.seed(seed)
            np.random.shuffle(indices)

        train_idx, valid_idx = indices[:split_train],indices[split_train:]

        train_sampler = sampler.SubsetRandomSampler(train_idx)
        valid_sampler = sampler.SubsetRandomSampler(valid_idx)
        return [train_sampler,valid_sampler]


class KnowledgeBase(Dataset):
    def __init__(self,kb,model):
        self.kb = kb
        self.model = model
        self.length = max([len(ex['text']) for ex in self.kb])
        # {
        #     'id': kb_data['id'][idx],
        #     'text': text,
        #     'pos': pos,
        #     'label': label,
        #     'lat': lat, #index
        #     'lng': lng, #index
        #     'lat_raw': kb_data[idx]['lat'],
        #     'lng_raw': kb_data[idx]['lng'],
        #     'category': kb_data[idx]['category'],
        # }
    def __len__(self):
        return len(self.kb)
    def __getitem__(self, idx):
        entry = self.kb[idx]
        sample = kbtoVector(entry, self.model,self.length)
        return sample
    def batchify(self):
        k = self.model.args.kb_n

        kb_list = self.kb[:k]
        logger.info('used %d kb entries'% k)
        kb_vec_list = [kbtoVector(ex,self.model,self.length) for ex in kb_list]
        batch_kb = kb_collate(kb_vec_list)
        logger.info('Converted kb entries to tensor')
        return batch_kb