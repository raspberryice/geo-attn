
import math
import torch
from torch import tensor
from torch.autograd import Variable
from torch import optim
from torch import nn

oneDivSqrtTwoPI = 1.0 / (2.0*math.pi)# normalisation factor for gaussian.
def gaussian_distribution(y, mu, sigma):
    '''

    :param y: 2*batch
    :param mu: K*2
    :param sigma:K*2
    :return: prob K*batch
    '''
    try:
        batch = y.data.shape[1]
    except IndexError:
        batch=1
        y = y.unsqueeze(dim=1)

    K = mu.data.shape[0]
    y = y.unsqueeze(dim=0).expand(K, 2, batch)  # poin *2*batch
    mu = mu.unsqueeze(dim=2).expand_as(y)
    sigma = sigma.unsqueeze(dim=2).expand_as(y)

    diffsigma = (y - mu) * torch.reciprocal(sigma)
    expterm = -0.5* torch.sum(diffsigma*diffsigma,dim=1)
    sigmareci = torch.reciprocal(sigma[:,0,:]*sigma[:,1,:]) #should be able to slice along a dimension
    prob = (torch.exp(expterm) * sigmareci * oneDivSqrtTwoPI)
    assert(torch.min(torch.le(prob,sigmareci*oneDivSqrtTwoPI)).data[0]==1)
    return prob

def mdn_loss_function(out_pi, out_sigma, out_mu, y,reduce=True):
    '''

    :param out_pi: K* batch
    :param out_sigma: K*2
    :param out_mu:  K*2
    :param y: 2*batch or 2
    :return:
    '''
    result = gaussian_distribution(y, out_mu, out_sigma) * out_pi
    result = torch.sum(result, dim=0)
    result = - torch.log(result)
    if reduce:
        return torch.sum(result)
    else:
        return result

def gaussian_grad(x,mu,sigma):
    # gradient of gaussian at x
    # poin*2*batch
    dis = gaussian_distribution(x,mu,sigma)
    tmp = (x- mu)*torch.reciprocal(sigma)
    tmp = torch.sum(tmp*tmp,dim=1)
    grad = (x-mu)*torch.reciprocal(sigma*sigma*2)
    grad = grad*dis.unsqueeze(dim=1).expand_as(grad)*tmp.unsqueeze(dim=1).expand_as(grad)
    return grad

def psuedo_seek_mode(out_pi,out_sigma,out_mu):
    '''

    :param out_pi: K*batch
    :param out_sigma: K*2
    :param out_mu: K*2
    :return:
    mode:2*K
    mode_val: batch * K
    '''
    #use the mean of components as mode
    mode = torch.transpose(out_mu,0,1)#2*mode
    dis = gaussian_distribution(mode,out_mu,out_sigma) #K*mode
    mode_val = torch.matmul(out_pi.transpose(0,1),dis)#batch*mode

    return [mode,mode_val]

def seek_mode(out_pi,out_sigma,out_mu):
    '''

    :param out_pi:
    :param out_sigma:
    :param out_mu:
    :return: mode_val batch*K
    '''
    mode_loss=0

    K = out_pi.data.shape[0]
    batch = out_pi.data.shape[1]
    mu = Variable(out_mu.data) #clears history
    pi = Variable(out_pi.data)
    sigma = Variable(out_sigma.data)
    peaks = torch.transpose(mu,0,1).unsqueeze(dim=2).expand(2,K,batch).contiguous().view(2,-1) #2*(mode*batch)
    pi = pi.unsqueeze(dim=1).expand(K,K,batch).contiguous().view(K,-1) #K*(mode*batch)

    peaks = nn.Parameter(peaks.data,requires_grad=True)
    optimizer = optim.SGD(params=[peaks, ], lr=1e-6)
    for epoch in range(10):
        optimizer.zero_grad()
        mode_loss = mdn_loss_function(pi, sigma, mu, peaks,reduce=False) #mode*batch
        torch.sum(mode_loss).backward()
        optimizer.step()

    mode_val = torch.exp(mode_loss.view(batch,K)*(-1))
    return peaks.view(2,K,batch),mode_val


def get_max_mode(mode, mode_val):
    '''

    :param mode: 2*mode or 2*mode*batch
    :param mode_val: batch*mode
    :return:
    '''
    max_val, max_idx = torch.max(mode_val, dim=1)
    if mode.dim() == 2:
        max_mode = mode[:, max_idx.data]
    else:
        batchn = max_idx.data.shape[0]
        mode_idx = list(zip(max_idx.data, range(batchn)))
        max_mode = torch.stack([mode[:, midx[0], midx[1]] for midx in mode_idx]).transpose(0, 1)  # 2*batch
    return max_mode

