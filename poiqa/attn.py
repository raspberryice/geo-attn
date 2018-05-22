import torch.nn as nn
import torch
import torch.nn.functional as F


class Attn(nn.Module):
    def __init__(self, method, hidden_size, max_length):
        super(Attn, self).__init__()
        self.seq_len=max_length
        self.method = method
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, self.hidden_size)

        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, self.hidden_size)
            self.other = nn.Parameter(torch.rand(1, 1,self.hidden_size)) #expand on need

    def forward(self, hidden, encoder_outputs):
        attn_energies = self.score(hidden,encoder_outputs)
        return F.softmax(attn_energies).view(-1,1,self.seq_len)

    def score(self, hidden, encoder_outputs):
        '''
        :param hidden: h_c (num_layers * num_directions, batch, hidden_size)
        :param encoder_output: output(batch,seq_len, hidden_size * num_directions)
        :return:
        '''
        encoder_outputs = encoder_outputs.view(self.seq_len,-1,self.hidden_size)
        hidden = hidden.view(1,-1,self.hidden_size).expand_as(encoder_outputs)

        if self.method == 'dot':
            assert (hidden.data.shape==encoder_outputs.data.shape)
            energy = hidden.mul(encoder_outputs).sum(dim=2)
            return energy

        elif self.method == 'general':
            energy = self.attn(encoder_outputs)
            assert (hidden.data.shape==energy.data.shape)
            energy = hidden.mul(energy).sum(dim=2)
            return energy

        elif self.method == 'concat':
            energy = self.attn(torch.cat((hidden, encoder_outputs), 2))
            energy = self.other.expand_as(energy).mul(energy).sum(dim=2)
            return energy


class BilinearMaxAttn(nn.Module):
    def __init__(self,q_size,key_size):
        super(BilinearMaxAttn, self).__init__()
        self.linear = nn.Linear(q_size,key_size)

    def forward(self,query,key):
        '''

                :param query: batch* query_dim
                :param key: n_keys *key_dim
                :return: attn of n_keys batch*n_keys
                '''
        Wquery = self.linear(query)
        # batch * key_dim
        score = torch.matmul(Wquery, torch.t(key))
        # batch*n_keys
        expscore = torch.exp(score)
        alpha = expscore/(torch.max(expscore,dim=1)[0].unsqueeze(dim=1).expand_as(expscore))
        # normalize alpha
        alpha = alpha/(torch.sum(alpha,dim=1).unsqueeze(dim=1).expand_as(alpha))
        return alpha


class BilinearAttn(nn.Module):
    def __init__(self, q_size, key_size):
        super(BilinearAttn, self).__init__()


        self.linear = nn.Linear(q_size, key_size)

    def forward(self,query,key):
        '''

        :param query: batch* query_dim
        :param key: n_keys *key_dim
        :return: attn of n_keys batch*n_keys
        '''
        Wquery = self.linear(query)
        # batch * key_dim
        score = torch.matmul(Wquery,torch.t(key))
        #batch*n_keys
        alpha = F.softmax(score)
        return alpha

class SeqAttn(nn.Module):
    def __init__(self,q_size,key_size):
        super(SeqAttn,self).__init__()
        self.linear = nn.Linear(q_size,key_size)

    def forward(self,query,key):
        '''

        :param key: batch *seq_len*key_dim
        :param query:batch* query_dim
        :return: attn over batch* seq_len
        '''
        Wquery = self.linear(query)
        xWy = key.bmm(Wquery.unsqueeze(2)).squeeze(2)
        # batch*seq_len*key_dim
        return F.softmax(xWy)


class LocAttn(nn.Module):
    def __init__(self,key_size,max_length):
        # key batch *seq_len*key_dim
        super(LocAttn, self).__init__()
        self.linear = nn.Linear(key_size,key_size)
        self.q = nn.Parameter(torch.rand(max_length,key_size))
    def forward(self,key):
        Wkey = F.tanh(self.linear(key)) #same size as key
        seq_len = key.data.shape[1]
        score = torch.sum(torch.mul(self.q[:seq_len,:],Wkey),2)
        alpha = F.softmax(score)
        return alpha



def weighted_avg_seq(value, attn):
    """Return a weighted average of x (a sequence of vectors).

    Args:
        value: batch * len * hdim
        attn: batch * len, sum(dim = 1) = 1
    Output:
        x_avg: batch * hdim
    """
    return attn.unsqueeze(1).bmm(value).squeeze(1)

def weighted_avg(values,attn):
    '''values: n_keys* value_dim
       attn: batch*n_keys
       return: applied: batch*value_dim
    '''
    return torch.matmul(attn,values)
