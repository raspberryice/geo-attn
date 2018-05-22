import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from .attn import weighted_avg, BilinearAttn,weighted_avg_seq, SeqAttn,LocAttn
import math


import logging
logger = logging.getLogger()

class MemNN(nn.Module):
    def __init__(self, args):
        super(MemNN, self).__init__()
        self.args = args

        self.embedding = nn.Embedding(self.args.vocab_size, self.args.embedding_dim, padding_idx=0)



        input_size = args.embedding_dim + args.num_features
        if args.network =='mem' :
            self.q_rnn = nn.GRU(input_size=input_size,
                                hidden_size=self.args.hidden_size,
                                num_layers=self.args.layers,
                                dropout=self.args.dropout_rnn,
                                bidirectional=True,
                                batch_first=True)
            # hidden size for one direction
            # ignoring padding
            q_size = 2 * self.args.hidden_size
        elif args.network=='mem-attn':
            # using output of self-attention as query
            self.q_rnn = nn.GRU(input_size=input_size,
                                hidden_size=self.args.hidden_size,
                                num_layers=self.args.layers,
                                dropout=self.args.dropout_rnn,
                                bidirectional=True,
                                batch_first=True)
            self.q_attn = LocAttn(2* self.args.hidden_size,self.args.max_len)
            q_size = self.args.hidden_size*2
        else:
            #network == 'mem-bow'
            q_size = input_size

        # memory
        self.key_size = args.embedding_dim
        self.value_size = 2
        attn_size = self.value_size +self.key_size

        self.sem_attn = BilinearAttn(q_size, self.key_size)



    def forward(self, ex):
        words = ex[0]
        features = ex[1]
        mask = ex[2]

        q_embed = self.embedding(words)
        # dropout after embedding
        if self.args.dropout_emb > 0:
            q_embed = F.dropout(q_embed, p=self.args.dropout_emb)
        q_input = [q_embed]
        if self.args.num_features > 0:
            q_input.append(features)
        if self.args.network=='mem' or self.args.network=='mem-attn':
            rnn_output,q = self.q_rnn(torch.cat(q_input, dim=2))
            if self.args.dropout_rnn_output:
                q = F.dropout(q, p=self.args.dropout_rnn)
            q_hidden_size = 2 * self.args.hidden_size
            q = q.view(-1, q_hidden_size)
            if self.args.network == 'mem-attn':
                attn_mask = self.q_attn(rnn_output)  # batch*seq_len

                applied_attn = weighted_avg_seq(rnn_output, attn_mask)
                q = applied_attn
        else:
            q = torch.mean(torch.cat(q_input, dim=2),dim=1)
        mem = self.mem



        sem_attn = self.sem_attn(q, mem)# batch* K
        out_sigma = torch.exp(self.value_dev)
        out_pi = sem_attn.transpose(0,1)
        out_mu = self.value_mean #K*2

        return out_pi,out_sigma,out_mu

    def init_memory(self, kb):
        # initialize memory with loaded embeddings
        words = kb[0]
        loc = kb[-1]
        # bag of word representation for POI names
        mem_embed = self.embedding(words)
        mem = torch.mean(mem_embed, dim=1)

        # entry_n * embed_dim
        self.mem = nn.Parameter(mem.data,requires_grad=True)

        self.value_mean = nn.Parameter(loc.data,requires_grad=False)
        self.value_dev = nn.Parameter(torch.rand(loc.data.shape)*math.log(0.01),requires_grad=True)
        return
