import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from .attn import SeqAttn,Attn, weighted_avg_seq, LocAttn
import math

CENTER = [40.7127, -74.0059]
class BOW_MDN(nn.Module):
    '''
    This is the model from Rahimi et al. EMNLP 2017
    no embedding layer is used
    args.component: number of Gaussian components
    '''
    def __init__(self,args):
        super(BOW_MDN,self).__init__()
        self.args = args
        # self.embedding = nn.Embedding(args.vocab_size, args.embedding_dim, padding_idx=0)
        self.hidden = nn.Linear(args.vocab_size,args.embedding_dim)
        self.hidden2pi = nn.Linear(args.embedding_dim,args.componentn)
        self.mu = nn.Parameter(
            (torch.rand(args.componentn, 2) - 0.5) * 4 + torch.FloatTensor(CENTER).expand(args.componentn,
                                                                                                       2),
            requires_grad=True)
        self.sigma = nn.Parameter(torch.rand(args.componentn, 2) * math.log(0.01), requires_grad=True)

    def forward(self,inputs):
        words = inputs[0]
        features = inputs[1]
        mask = inputs[2]
        #q_embed = self.embedding(words)
        q_bow = []
        for i in range(words.data.shape[0]):
            s = torch.zeros(self.args.vocab_size).cuda()
            s[words[i,:].data] = 1
            q_bow.append(s)
        q = autograd.Variable(torch.stack(q_bow,dim=0))
        #q = torch.mean(q_embed,dim=1) #bow
        hidden = F.tanh(self.hidden(q))
        pi = F.softmax(self.hidden2pi(hidden)).transpose(0,1)
        sigma = torch.exp(self.sigma)
        mu = self.mu
        return pi,sigma,mu

class Regression(nn.Module):
    '''
    the lat and lng are directly predicted from the message representation.
    '''
    def __init__(self,args):
        super(Regression,self).__init__()
        self.args = args
        self.embedding = nn.Embedding(args.vocab_size, args.embedding_dim, padding_idx=0)
        input_size = args.embedding_dim + args.num_features

        self.rnn = nn.GRU(input_size, args.hidden_size,
                          dropout=self.args.dropout_rnn,
                          batch_first=True,
                          num_layers=self.args.layers,
                          bidirectional=True)
        q_hidden_size = args.hidden_size * 2
        self.attn = LocAttn(q_hidden_size, args.max_len)
        self.attn2hidden = nn.Linear(q_hidden_size, q_hidden_size)
        self.hidden2cor  = nn.Linear(q_hidden_size,2)

    def forward(self,inputs):
        words = inputs[0]
        features = inputs[1]
        mask = inputs[2]
        q_embed = self.embedding(words)
        # dropout after embedding
        if self.args.dropout_emb > 0:
            q_embed = F.dropout(q_embed, p=self.args.dropout_emb)
        q_input = [q_embed]
        if self.args.num_features > 0:
            q_input.append(features)

        rnn_output, q_hidden = self.rnn(torch.cat(q_input, dim=2))
        q_hidden = q_hidden.view(-1, self.args.hidden_size * 2)

        if self.args.dropout_rnn_output:
            q_hidden = F.dropout(q_hidden, p=self.args.dropout_rnn)
        attn_mask = self.attn(rnn_output)
        applied_attn = weighted_avg_seq(rnn_output, attn_mask)
        hidden = self.attn2hidden(applied_attn)
        cor = self.hidden2cor(F.relu(hidden))
        return cor

class AttnMDN(nn.Module):
    '''
    mixture density model with location-based attention
    args.componentn: number of gaussian components
    '''
    def __init__(self,args):
        super(AttnMDN, self).__init__()
        self.args = args
        self.embedding = nn.Embedding(args.vocab_size, args.embedding_dim, padding_idx=0)
        input_size = args.embedding_dim + args.num_features

        self.rnn = nn.GRU(input_size, args.hidden_size,
                          dropout=self.args.dropout_rnn,
                          batch_first=True,
                          num_layers=self.args.layers,
                          bidirectional=True)
        q_hidden_size = args.hidden_size * 2
        self.attn = LocAttn(q_hidden_size, args.max_len)
        self.attn2hidden = nn.Linear(q_hidden_size,q_hidden_size)
        self.hidden2pi = nn.Linear(q_hidden_size,args.componentn)
        self.mu = nn.Parameter((torch.rand(args.componentn,2)-0.5)*4 + torch.FloatTensor(CENTER).expand(args.componentn,2),requires_grad=True)
        self.sigma = nn.Parameter(torch.rand(args.componentn,2)*math.log(0.01),requires_grad=True)
    def forward(self,inputs):
        words = inputs[0]
        features = inputs[1]
        mask = inputs[2]
        q_embed = self.embedding(words)
        # dropout after embedding
        if self.args.dropout_emb > 0:
            q_embed = F.dropout(q_embed, p=self.args.dropout_emb)
        q_input = [q_embed]
        if self.args.num_features > 0:
            q_input.append(features)

        rnn_output, q_hidden = self.rnn(torch.cat(q_input, dim=2))
        q_hidden = q_hidden.view(-1, self.args.hidden_size * 2)

        if self.args.dropout_rnn_output:
            q_hidden = F.dropout(q_hidden, p=self.args.dropout_rnn)
        attn_mask = self.attn(rnn_output)
        applied_attn = weighted_avg_seq(rnn_output, attn_mask)
        hidden = self.attn2hidden(applied_attn)
        pi = F.softmax(self.hidden2pi(F.relu(hidden))).transpose(0,1)
        sigma = torch.exp(self.sigma)
        mu = self.mu
        return pi,sigma,mu


