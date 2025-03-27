
"""
Created on Wed Sep 18 20:27:19 2019
@author: MOHAMEDR002
"""
import numpy as np
import torch
import random
from torch.utils.data import TensorDataset, DataLoader
import math
from torch import nn
from torch.utils.data import Dataset
from numba import jit
from torch.autograd import Function
import torch.nn.functional as F
import torch.distributions as dist
import logging
import os
from datetime import datetime
import sys

device = torch.device('cuda')

def _logger(logger_name, level=logging.DEBUG):
    """
    Method to return a custom logger with the given name and level
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    format_string = "%(message)s"
    log_format = logging.Formatter(format_string)
    # Creating and adding the console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    # Creating and adding the file handler
    file_handler = logging.FileHandler(logger_name, mode='a')
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    return logger


def starting_logs(data_type, da_method, exp_log_dir, src_id, tgt_id, run_id):
    log_dir = os.path.join(exp_log_dir, src_id + "_to_" + tgt_id + "_run_" + str(run_id))
    os.makedirs(log_dir, exist_ok=True)
    log_file_name = os.path.join(log_dir, f"logs_{da_method}_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log")
    logger = _logger(log_file_name)
    logger.debug("=" * 45)
    logger.debug(f'Dataset: {data_type}')
    logger.debug(f'Method:  {da_method}')
    logger.debug("=" * 45)
    logger.debug(f'Source: {src_id} ---> Target: {tgt_id}')
    logger.debug(f'Run ID: {run_id}')
    logger.debug("=" * 45)
    return logger, log_dir


def set_requires_grad(model, requires_grad=True):
    for param in model.parameters():
        param.requires_grad = requires_grad
def loop_iterable(iterable):
    while True:
        yield from iterable

def fix_randomness(SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    
def scoring_func(error_arr):
    pos_error_arr = error_arr[error_arr >= 0] 
    neg_error_arr = error_arr[error_arr < 0]
    score = 0 
    for error in neg_error_arr:
        score = math.exp(-(error / 13)) - 1 + score 
    for error in pos_error_arr: 
        score = math.exp(error / 10) - 1 + score

    return score

def roll(x, shift: int, dim: int = -1, fill_pad: int = None):

    if 0 == shift:
        return x

    elif shift < 0:
        shift = -shift
        gap = x.index_select(dim, torch.arange(shift))
        if fill_pad is not None:
            gap = fill_pad * torch.ones_like(gap, device=x.device)
        return torch.cat([x.index_select(dim, torch.arange(shift, x.size(dim))), gap], dim=dim)

    else:
        shift = x.size(dim) - shift
        gap = x.index_select(dim, torch.arange(shift, x.size(dim)))
        if fill_pad is not None:
            gap = fill_pad * torch.ones_like(gap, device=x.device)
        return torch.cat([gap, x.index_select(dim, torch.arange(shift))], dim=dim)
def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)
        # if name=='weight':
        #     nn.init.kaiming_uniform_(param.data)
        # else:
        #     torch.nn.init.zeros_(param.data)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self,yhat,y):
        return torch.sqrt(self.mse(yhat,y))
    
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score - self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss


class MMDLoss(nn.Module):
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None, **kwargs):
        super(MMDLoss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul, kernel_num, fix_sigma):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target):
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            XX = torch.mean(kernels[:batch_size, :batch_size])
            YY = torch.mean(kernels[batch_size:, batch_size:])
            XY = torch.mean(kernels[:batch_size, batch_size:])
            YX = torch.mean(kernels[batch_size:, :batch_size])
            loss = torch.mean(XX + YY - XY - YX)
            return loss

class MMDLoss2(nn.Module):
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None, **kwargs):
        super(MMDLoss2, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul, kernel_num, fix_sigma):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target):
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            XX = torch.mean(kernels[:batch_size, :batch_size])
            YY = torch.mean(kernels[batch_size:, batch_size:])
            XY = torch.mean(kernels[:batch_size, batch_size:])
            YX = torch.mean(kernels[batch_size:, :batch_size])
            loss = torch.mean( - XY - YX)
            return loss

class conditional_MMDLoss(nn.Module):
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None, **kwargs):
        super(conditional_MMDLoss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul, kernel_num, fix_sigma):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss


    def forward(self, source, target, src_y, tgt_y):
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)

            XX = torch.mean(torch.abs(src_y - src_y.view(-1,1)) * kernels[:batch_size, :batch_size])
            YY = torch.mean(torch.abs(tgt_y - tgt_y.view(-1,1)) * kernels[batch_size:, batch_size:])
            XY = torch.mean(torch.abs(tgt_y - src_y.view(-1,1)) * kernels[:batch_size, batch_size:])
            YX = torch.mean(torch.abs(src_y - tgt_y.view(-1,1)) * kernels[batch_size:, :batch_size])
            #print(XX, YY, XY, YX)
            loss = torch.mean(- XY - YX) #XX + YY 
            return loss

class Tri_MMDLoss(nn.Module):
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None, **kwargs):
        super(Tri_MMDLoss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, anchor, pos, neg, kernel_mul, kernel_num, fix_sigma):
        n_samples = int(anchor.size()[0]) + int(pos.size()[0]) + int(neg.size()[0])
        total = torch.cat([anchor, pos, neg], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_anchor, f_of_pos, f_of_neg):
        loss = 0.0
        delta = f_of_anchor.float().mean(0) - f_of_pos.float().mean(0) + f_of_neg.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, anchor, pos, neg):
        if self.kernel_type == 'linear':
            return self.linear_mmd2(anchor, pos, neg)
        elif self.kernel_type == 'rbf':
            anc_batch_size = int(anchor.size()[0])
            neg_batch_size = int(neg.size()[0])
            kernels = self.guassian_kernel(
                anchor, pos, neg, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            AA = torch.mean(kernels[:anc_batch_size, :anc_batch_size])
            PP = torch.mean(kernels[anc_batch_size:anc_batch_size+neg_batch_size, anc_batch_size:anc_batch_size+neg_batch_size])
            NN = torch.mean(kernels[neg_batch_size:, neg_batch_size:])
            AP = torch.mean(kernels[:anc_batch_size, anc_batch_size:anc_batch_size+neg_batch_size])
            AN = torch.mean(kernels[:anc_batch_size, neg_batch_size:])
            PN = torch.mean(kernels[anc_batch_size:anc_batch_size+neg_batch_size, neg_batch_size:])
            PA = torch.mean(kernels[anc_batch_size:anc_batch_size+neg_batch_size, :anc_batch_size])
            NP = torch.mean(kernels[neg_batch_size:, anc_batch_size:anc_batch_size+neg_batch_size])
            NA = torch.mean(kernels[neg_batch_size:, :anc_batch_size])

            loss = torch.mean(- AP - PA) # + AN + NA) #PP - NN 
            return loss
            
def CORAL(source, target, **kwargs):
    d = source.data.shape[1]
    ns, nt = source.data.shape[0], target.data.shape[0]
    # source covariance
    xm = torch.mean(source, 0, keepdim=True) - source
    xc = xm.t() @ xm / (ns - 1)
    #print('coral', xm.shape, xc.shape)
    # target covariance
    xmt = torch.mean(target, 0, keepdim=True) - target
    xct = xmt.t() @ xmt / (nt - 1)

    # frobenius norm between source and target
    loss = torch.mul((xc - xct), (xc - xct))
    loss = torch.sum(loss)# / (4*d*d)
    return loss


def NIG_NLL(y, gamma, v, alpha, beta, w_i_dis_mean, quantile, reduce=True):
    tau_two = 2.0 / (quantile * (1.0 - quantile))
    twoBlambda = 2.0 * 2.0 * beta * (1.0 + tau_two * w_i_dis_mean * v)

    nll = 0.5 * torch.log(torch.tensor(np.pi) / v) \
        - alpha * torch.log(twoBlambda) \
        + (alpha + 0.5) * torch.log(v * (y - gamma) ** 2 + twoBlambda) \
        + torch.lgamma(alpha) \
        - torch.lgamma(alpha + 0.5)

    return torch.mean(nll) if reduce else nll

def KL_NIG(mu1, v1, a1, b1, mu2, v2, a2, b2):
    KL = 0.5 * (a1 - 1) / b1 * (v2 * (mu2 - mu1) ** 2) \
        + 0.5 * v2 / v1 \
        - 0.5 * torch.log(torch.abs(v2) / torch.abs(v1)) \
        - 0.5 + a2 * torch.log(b1 / b2) \
        - (torch.lgamma(a1) - torch.lgamma(a2)) \
        + (a1 - a2) * torch.digamma(a1) \
        - (b1 - b2) * a1 / b1
    return KL

def tilted_loss(q, e):
    return torch.maximum(q * e, (q - 1) * e)

def NIG_Reg(y, gamma, v, alpha, beta, w_i_dis_mean, quantile, omega=0.01, reduce=True, kl=False):
    tau_two = 2.0 / (quantile * (1.0 - quantile))
    #w = beta*(1+v)/(alpha*v)
    error = tilted_loss(quantile, y - gamma)

    if kl:
        kl = KL_NIG(gamma, v, alpha, beta, gamma, omega, 1 + omega, beta)
        reg = error * kl
    else:
        evi = 2 * v + alpha + 1 / beta
        reg = error * evi

    return torch.mean(reg) if reduce else reg

def quant_evi_loss(y_true, gamma, v, alpha, beta, quantile, coeff=1.0, reduce=True):
    theta = (1.0 - 2.0 * quantile) / (quantile * (1.0 - quantile))
    mean_ = beta / (alpha - 1)

    w_i_dis = dist.Exponential(rate=1 / mean_)
    w_i_dis_mean = w_i_dis.mean
    mu = gamma + theta * w_i_dis_mean
    loss_nll = NIG_NLL(y_true, mu, v, alpha, beta, w_i_dis_mean, quantile, reduce=reduce)
    loss_reg = NIG_Reg(y_true, gamma, v, alpha, beta, w_i_dis_mean, quantile, reduce=reduce)
    return loss_nll + coeff * loss_reg

def quant_evi_loss_upt(y_true, gamma, v, alpha, beta, quantile, coeff=1.0, reduce=True):
    with torch.no_grad():
        gamma = gamma.clone()

    loss_one = quant_evi_loss(y_true, gamma.detach(), v, alpha, beta, quantile, coeff=coeff, reduce=False)
    error_loss = tilted_loss(quantile, y_true - gamma)
    reg = 1e-2 * (error_loss + loss_one) # This seems to be the intended variable, not 'reg' from NIG_Reg
    #print(error_loss, loss_one)
    return torch.mean(reg) if reduce else reg

def NIG_NLL_org(y, gamma, v, alpha, beta, reduce=True):
    twoBlambda = 2*beta*(1+v)

    nll = 0.5*torch.log(torch.pi/v)  \
        - alpha*torch.log(twoBlambda)  \
        + (alpha+0.5) * torch.log(v*(y-gamma)**2 + twoBlambda)  \
        + torch.lgamma(alpha)  \
        - torch.lgamma(alpha+0.5)

    return torch.mean(nll) if reduce else nll

def NIG_Reg_org(y, gamma, v, alpha, beta, omega=0.01, reduce=True, kl=False):
    # error = tf.stop_gradient(tf.abs(y-gamma))
    error = torch.abs(y-gamma)

    if kl:
        kl = KL_NIG(gamma, v, alpha, beta, gamma, omega, 1+omega, beta)
        reg = error*kl
    else:
        evi = 2*v+(alpha)
        reg = error*evi

    return torch.mean(reg) if reduce else reg

def NIG_Reg_unreason(y, gamma, v, alpha, beta, omega=0.01, reduce=True, kl=False):
    # error = tf.stop_gradient(tf.abs(y-gamma))
    error = torch.abs(y-gamma)
    w = torch.sqrt(beta*(1+v)/(alpha*v))
    
    if kl:
        kl = KL_NIG(gamma, v, alpha, beta, gamma, omega, 1+omega, beta)
        reg = error*kl
    else:
        evi = 2*v+(alpha)
        reg = error/w*evi

    return torch.mean(reg) if reduce else reg

def Non_saturating_Reg(y, gamma, v, alpha, beta, omega=0.01, reduce=True):
    loss_u = (y-gamma)**2*v*(alpha-1)/(beta*(v+1))
    return torch.mean(loss_u) if reduce else loss_u

def evidential_loss(y_true, gamma, v, alpha, beta, coeff=1.0):
    loss_nll = NIG_NLL_org(y_true, gamma, v, alpha, beta)
    loss_reg = NIG_Reg_org(y_true, gamma, v, alpha, beta)
    return loss_nll + coeff * loss_reg

def evidential_unreason_loss(y_true, gamma, v, alpha, beta, coeff=1.0, reduce=True):
    loss_nll = NIG_NLL_org(y_true, gamma, v, alpha, beta, reduce=reduce)
    loss_reg = NIG_Reg_unreason(y_true, gamma, v, alpha, beta, reduce=reduce)
    #loss_u_reg = Non_saturating_Reg(y_true, gamma, v, alpha, beta, reduce=reduce)
    return loss_nll + coeff * loss_reg# + loss_u_reg

def evidential_unreason_loss2(y_true, gamma, v, alpha, beta, coeff=1.0, reduce=True):
    loss_nll = torch.log(beta/v)
    loss_reg = (y_true - gamma)**2/(beta/v)
    return torch.mean(loss_nll + (1+coeff*v) * loss_reg)

def non_saturating_reg_loss(y_true, gamma, v, alpha, beta):
    return torch.pow(y_true - gamma, 2) * v * (alpha - 1) / beta / (v + 1)
    
class bhattachayra(nn.Module):
    def __init__(self):
        super(bhattachayra, self).__init__()

    def forward(self, mu1, sigma1, mu2, sigma2):
        '''
        Calculate the bhattachayra distance, only suitable for distributions assumed to be normal
        mu1: torch array of reference source means 
        sigma1: torch array of reference source stds
        mu2: torch array of target means 
        sigma2: torch array of target stds
        return: bhattachayra distance between the two normal distributions
        '''
        assert mu1.size() == sigma1.size() == mu2.size() == sigma2.size()
        # Add small amount to prevent explosion to infinity
        sigma1 = sigma1 + 1e-6
        sigma2 = sigma2 + 1e-6
        squares = torch.pow(sigma1,2) + torch.pow(sigma2,2)
        db = 1/4 * ((mu1 - mu2)**2)/squares + 1/2 * torch.log(squares/(2*sigma1*sigma2))
        return db.mean()
    
class bhattachayra_GMM(nn.Module):
    def __init__(self, n_components, ref_mu, ref_sigma, ref_pi):
        #cuda = torch.cuda.is_available()

        super(bhattachayra_GMM, self).__init__() 
        self.n_components = n_components
        if self.n_components not in [1, 2, 3]:
            raise Exception('Only implemented for one, two or three components so far')
        
        self.ref_mu = torch.from_numpy(ref_mu)
        self.ref_sigma = torch.from_numpy(ref_sigma)
        self.ref_pi = torch.from_numpy(ref_pi)
        self.loss = bhattachayra()
        
#        if cuda:
#            self.ref_mu = self.ref_mu.cuda()
#            self.ref_sigma = self.ref_sigma.cuda()
#            self.ref_pi = self.ref_pi.cuda()

        assert(ref_mu.shape[1] == self.n_components)

    def forward(self, mus, sigmas, pis):
        '''
        Calculate the bhattachayra distance summed up, only suitable for distributions assumed to be normal
        mus: torch vector of reference target means
        sigmas: torch vector of reference target stds
        pis: torch vector of reference target pis
        return: bhattachayra distance between the two GMMs
        '''
        mus = mus.squeeze()
        sigmas = sigmas.squeeze()
        pis = pis.squeeze()
        
        if self.n_components == 1:
            mus = mus.view(-1, 1)
            sigmas = sigmas.view(-1, 1)
            pis = pis.view(-1, 1)
            
        assert mus.shape == self.ref_mu.shape
        assert sigmas.shape == self.ref_sigma.shape        
        assert pis.shape == self.ref_pi.shape
        
        if self.n_components == 1:
            dist00 = self.ref_pi[:]* pis[:]*self.loss(self.ref_mu[:], self.ref_sigma[:], mus[:], sigmas[:])
            total = dist00
        if self.n_components == 2:
            dist00 = self.ref_pi[:,0]* pis[:,0]*self.loss(self.ref_mu[:,0], self.ref_sigma[:,0], mus[:,0], sigmas[:,0])
            dist11 = self.ref_pi[:,1]* pis[:,1]*self.loss(self.ref_mu[:,1], self.ref_sigma[:,1], mus[:,1], sigmas[:,1])
            total = dist00  + dist11
        if self.n_components == 3: 
            dist00 = self.ref_pi[:,0]* pis[:,0]*self.loss(self.ref_mu[:,0], self.ref_sigma[:,0], mus[:,0], sigmas[:,0])
            dist11 = self.ref_pi[:,1]* pis[:,1]*self.loss(self.ref_mu[:,1], self.ref_sigma[:,1], mus[:,1], sigmas[:,1])
            dist22 = self.ref_pi[:,2]* pis[:,2]*self.loss(self.ref_mu[:,2], self.ref_sigma[:,2], mus[:,2], sigmas[:,2])
            total = dist00 + dist11 + dist22
        return total.mean()
