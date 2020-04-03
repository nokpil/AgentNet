# coding=utf-8
import argparse
import os
import random
import shutil
import time
import warnings
import sys
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import math
import scipy.interpolate as interp

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau as RLRP
from torch.nn.init import xavier_normal
from torch.nn.parameter import Parameter
from torch.distributions.multivariate_normal import MultivariateNormal

import pickle
import importlib
import itertools
import random
from collections import OrderedDict
from copy import deepcopy

import src.DataStructure as DS

## Data type conversion

def DCN(x):
    return x.data.cpu().numpy()
def CN(x):
    return x.cpu().numpy()
def TTC(x):
    return torch.Tensor(x).cuda()

## Easy plots

def imshow_now(x):
    fig = plt.figure(figsize = (4,4), dpi = 150)
    ax = fig.add_subplot(1,1,1)
    im = ax.imshow(x, cmap = cm.RdBu, aspect = 'auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

def plot_now(x):
    fig = plt.figure(figsize = (4,4), dpi = 150)
    ax = fig.add_subplot(1,1,1)
    for X in x:
        ax.plot(X[0], X[1])

def scatter_now(x):
    fig = plt.figure(figsize = (4,4), dpi = 150)
    ax = fig.add_subplot(1,1,1)
    for X in x:
        ax.scatter(X[0], X[1], s = 2)

## Useful functions

def sign():
    return 1 if np.random.random() < 0.5 else -1

def false_shuffle(length, threshold):
    target = [0 for _ in range(length)]
    fixed = set([])
    for i in range(length):
        possible = set(range(max(0, i-threshold), min(length, i+threshold)))
        true_possible = possible - fixed 
        if true_possible:
            s = np.random.choice(list(true_possible))
        #print(i, true_possible, set.intersection(possible, fixed), s)
        fixed.add(s)
        target[i] = s
    return target

def pdist(z):
    z_norm = (z ** 2).sum(2).view(-1, z.shape[1], 1)
    w_t = torch.transpose(z, 1, 2)
    w_norm = z_norm.view(-1, 1, z.shape[1])
    dist = z_norm + w_norm - 2.0 * torch.bmm(z, w_t)
    dist = torch.clamp(dist, 0., np.inf)
    # return torch.pow(dist, 0.5)
    return dist

def angle_between(v1, v2):
    return np.degrees(np.math.atan2(np.linalg.det([v1, v2]), np.dot(v1, v2)))

def angle_between_vec(x, y):
    x1, y1, vx1, vy1 = x
    x2, y2, vx2, vy2 = y
    v1 = np.array([vx1, vy1])
    v2 = np.array([x2 - x1, y2 - y1])
    if np.sum(v2) == 0:
        v2 = v1
    return angle_between(v1, v2)

def logsumexp(x, dim=None):
    x_max = torch.max(x, dim=dim, keepdim=True)[0]
    z = torch.log(torch.sum(torch.exp(x - x_max), dim=dim, keepdim=True)) + x_max
    return z.sum(dim=dim)

def norm_sigmoid(a, b, x):
    s= 1/(1+np.exp(b*(x-a)))
    return 1*(s-np.min(s))/(np.max(s)-np.min(s)) # normalize function to 0-1

def idx2idx(idx, target):
    f = 0
    cnvt = []
    for i in target:
        f = np.argwhere(idx[f:]==i)[0][0]+f
        cnvt.append(f)
    return cnvt

def interpolate_polyline(polyline, num_points):
    duplicates = []
    for i in range(1, len(polyline)):
        if np.allclose(polyline[i], polyline[i - 1]):
            duplicates.append(i)
    if duplicates:
        polyline = np.delete(polyline, duplicates, axis=0)
    tck, u = interp.splprep(polyline.T, s=0)
    u = np.linspace(0.0, 1.0, num_points)
    return np.column_stack(interp.splev(u, tck))

def summary(ob):
    print('x    | max :' + str(np.max(ob['x'])) + ', min : ' + str(np.min(ob['x'])))
    print('y    | max :' + str(np.max(ob['y'])) + ', min : ' + str(np.min(ob['y'])))
    print('z    | max :' + str(np.max(ob['z'])) + ', min : ' + str(np.min(ob['z'])))
    print('vx   | max :' + str(np.max(ob['vx'])) + ', min : ' + str(np.min(ob['vx'])))
    print('vy   | max :' + str(np.max(ob['vy'])) + ', min : ' + str(np.min(ob['vy'])))
    print('vz   | max :' + str(np.max(ob['vz'])) + ', min : ' + str(np.min(ob['vz'])))

def autocorr(id_list):
    from pandas.plotting import autocorrelation_plot
    from pandas.plotting import lag_plot

    fig_list = []

    for i in range(len(id_list)):
        d = autocorrelation_plot(id_list[i][['vx', 'vy', 'vz']])
        x = d.axes.lines[5]
        data.append(x.get_xydata())

    fig = plt.figure(figsize=(18, 12))
    ax = fig.add_subplot(111)
    for i in range(len(id_list)):
        ax.plot(data[i][:, 0], np.abs(data[i][:, 1]), c='fuchsia', alpha=0.01, lw=2)

    ax.set_xlim(0, 1000)

    fig_list.append(fig)

    fig = plt.figure(figsize=(18, 12))
    ax = fig.add_subplot(111)
    for i in range(len(id_list)):
        for j in range(1, len(data[i][:, 1])):
            if data[i][:, 1][j] * data[i][:, 1][j - 1] < 0:
                break
        ax.plot(data[i][:j + 1, 0], np.abs(data[i][:j + 1, 1]), c='fuchsia', alpha=0.01, lw=2)

    ax.set_xlim(0, 500)

    fig_list.append(fig)

    return fig_list

def trajectory(id_list, index_list, time, interval):
    fig = plt.figure(figsize=(12, 12))
    ax = fig.gca(projection='3d')

    for index in index_list:
        if time > 0:
            data = id_list[index][:time:interval]
        else:
            data = id_list[index][::interval]
        ax.plot(data['x'].values, data['y'].values, data['z'].values, lw=5, c='red')
        ax.quiver(data['x'].values, data['y'].values, data['z'].values, data['vx'].values, data['vy'].values,
                  data['vz'].values, color='blueviolet', length=0.1, normalize=True)

# Pytorch Data modification

def ns(loader):
    new_loader = copy.deepcopy(loader)
    data_source = new_loader.sampler.data_source
    new_loader.batch_sampler.sampler = torch.utils.data.sampler.SequentialSampler(data_source)
    return new_loader


class DistributedSampler_LSTM(torch.utils.data.Sampler):

    def __init__(self, dataset, num_replicas=None, rank=None):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        # deterministically shuffle based on epoch
        indices = false_shuffle(len(self.dataset), 10)

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        #indices = indices[self.rank*self.num_samples : (self.rank+1)*self.num_samples]
        indices = indices[self.rank:self.total_size:self.num_replicas]
        #print(len(indices), self.num_samples)
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

def data_cat(train_set):
    train_data_list = []
    train_labels_list = []

    for i in range(len(train_set.train_data)):
        data = train_set.train_data[i]
        labels = train_set.train_labels[i]
        mask = torch.tensor([True if labels[j, 0] == 1 else False for j in range(labels.shape[0])])
        intp_data = data[mask, :]
        labels = labels[mask, :]
        train_data_list.append(DCN(intp_data))
        train_labels_list.append(DCN(labels))

    train_data = np.concatenate(train_data_list)
    train_labels = np.concatenate(train_labels_list)

    return train_data, train_labels

def make_label_mask(labels, boolean):
    if boolean:
        return torch.tensor([True if labels[j, 0] == 1 else False for j in range(labels.shape[0])])
    else:
        return torch.tensor([1. if labels[j, 0] == 1 else -1. for j in range(labels.shape[0])])

def masked_pair(data, label, i, device):
    mask = label_mask(label[i], boolean=True)
    return mask, (data[i, mask, :].to(device), label[i, mask, 1:].to(device))


def group_reattach(old_state, new_state):
    return torch.cat((new_state, old_state[:, :, -1].unsqueeze(-1)), dim=-1)


## Pytorch Batched Calculations

def batchedDot(a, b):
    return torch.matmul(a.view([*a.shape[:-1], 1, a.shape[-1]]), b.view([*b.shape, 1])).squeeze(-1).squeeze(-1)

def batchedInv(batchedTensor):
        if np.prod(batchedTensor.shape[:-2]) >= 256 * 256 - 1:
            chunk_num = int(np.prod(batchedTensor.shape[1:-2]))
            if chunk_num >= (256*256 - 1):
                print("TOO BIG TENSOR")
            max_split = (256 * 256 - 1)//chunk_num
            temp = []
            for t in torch.split(batchedTensor, max_split):
                temp.append(torch.inverse(t))
            return torch.cat(temp)
        else:
            return torch.inverse(batchedTensor)

def batchedDet(batchedTensor):
        if np.prod(batchedTensor.shape[:-2]) >= 256 * 256 - 1:
            chunk_num = int(np.prod(batchedTensor.shape[1:-2]))
            if chunk_num >= (256*256 - 1):
                print("TOO BIG TENSOR")
            max_split = (256 * 256 - 1)//chunk_num
            temp = []
            for t in torch.split(batchedTensor, max_split):
                temp.append(torch.det(t))
            return torch.cat(temp)
        else:
            return torch.det(batchedTensor)

def batchedDet_old(batchedTensor):
    jitter = 1e-6
    if np.prod(batchedTensor.shape[:-2]) >= 256 * 256 - 1:
        chunk_num = int(np.prod(batchedTensor.shape[1:-2]))
        if chunk_num >= (256*256 - 1):
            print("TOO BIG TENSOR")
        max_split = (256 * 256 - 1)//chunk_num
        det_list = []
        for t in torch.split(batchedTensor, max_split):
            temp.append(torch.prod(torch.diagonal(torch.cholesky(t), dim1=-2, dim2=-1), dim = -1)**2)
        return torch.cat(temp)
    else:
        return torch.prod(torch.diagonal(torch.cholesky(batchedTensor), dim1=-2, dim2=-1), dim = -1)**2


## Pytorch Loss

def KLGaussianGaussian(mu1, sig1, mu2, sig2, keep_dims=0):
    """
    Re-parameterized formula for KL
    between Gaussian predicted by encoder and Gaussian dist.
    Parameters
    ----------
    mu1  : FullyConnected (Linear)
    sig1 : FullyConnected (Softplus)
    mu2  : FullyConnected (Linear)
    sig2 : FullyConnected (Softplus)
    """
    if keep_dims:
        kl = 0.5 * (2 * torch.log(sig2) - 2 * torch.log(sig1) +
                    (sig1 ** 2 + (mu1 - mu2) ** 2) / sig2 ** 2 - 1)
    else:
        kl = torch.sum(0.5 * (2 * torch.log(sig2) - 2 * torch.log(sig1) +
                              (sig1 ** 2 + (mu1 - mu2) ** 2) /
                              sig2 ** 2 - 1), dim=-1)

    return kl

def bivariate_normal(x, mu, sig, corr):
    cv = np.array([sig[0] ** 2, sig[0] * sig[1] * corr[0],
                   sig[0] * sig[1] * corr[0], sig[1] ** 2]).reshape(2, 2)
    #print(mu)
    #print(cv)
    xmu = np.expand_dims(x - mu, axis=-1)

    z = np.squeeze(xmu.transpose(0, 1, 3, 2) @ np.linalg.inv(cv) @ xmu)
    denom = 2 * np.pi * np.sqrt(np.linalg.det(cv))

    return np.exp(-z / 2) / denom

def trivariate_normal(x, mu, sig, corr):
    cv = np.array([sig[0] ** 2, sig[0] * sig[1] * corr[0], sig[0] * sig[2] * corr[1],
                   sig[0] * sig[1] * corr[0], sig[1] ** 2, sig[1] * sig[2] * corr[2],
                   sig[0] * sig[2] * corr[1], sig[1] * sig[2] * corr[2], sig[2] ** 2]).reshape(3, 3)

    xmu = np.expand_dims(x - mu, axis=-1)
    z = np.squeeze(xmu.transpose(0, 1, 2, 4, 3) @ np.linalg.inv(cv) @ xmu)
    denom = np.power(2 * np.pi, 3 / 2) * np.sqrt(np.linalg.det(cv))

    return np.exp(-z / 2) / denom

def quadvariate_normal(x, mu, sig, corr):
    cv = np.array([sig[0] ** 2, sig[0] * sig[1] * corr[0], sig[0] * sig[2] * corr[1], sig[0] * sig[3] * corr[2],
                   sig[0] * sig[1] * corr[0], sig[1] ** 2, sig[1] * sig[2] * corr[3], sig[1] * sig[3] * corr[4],
                   sig[0] * sig[2] * corr[1], sig[1] * sig[2] * corr[3], sig[2] ** 2, sig[2] * sig[3] * corr[5],
                   sig[0] * sig[3] * corr[2], sig[1] * sig[3] * corr[4], sig[2] * sig[3] * corr[5], sig[3] ** 2]).reshape(3, 3)

    xmu = np.expand_dims(x - mu, axis=-1)
    z = np.squeeze(xmu.transpose(0, 1, 2, 3, 5, 4) @ np.linalg.inv(cv) @ xmu)
    denom = np.power(2 * np.pi, 4 / 2) * np.sqrt(np.linalg.det(cv))

    return np.exp(-z / 2) / denom


def hexavariate_normal(x, mu, sig, corr):
    cv = np.array([sig[0] ** 2, sig[0] * sig[1] * corr[0], sig[0] * sig[2] * corr[1], sig[0] * sig[3] * corr[2],
                   sig[0] * sig[4] * corr[3], sig[0] * sig[5] * corr[4],
                   sig[0] * sig[1] * corr[0], sig[1] ** 2, sig[1] * sig[2] * corr[5], sig[1] * sig[3] * corr[6],
                   sig[1] * sig[4] * corr[7], sig[1] * sig[5] * corr[8],
                   sig[0] * sig[2] * corr[1], sig[1] * sig[2] * corr[5], sig[2] ** 2, sig[2] * sig[3] * corr[9],
                   sig[2] * sig[4] * corr[10], sig[2] * sig[5] * corr[11],
                   sig[0] * sig[3] * corr[2], sig[1] * sig[3] * corr[6], sig[2] * sig[3] * corr[9], sig[3] ** 2,
                   sig[3] * sig[4] * corr[12], sig[3] * sig[5] * corr[13],
                   sig[0] * sig[4] * corr[3], sig[1] * sig[4] * corr[7], sig[2] * sig[4] * corr[10],
                   sig[3] * sig[4] * corr[12], sig[4] ** 2, sig[4] * sig[5] * corr[14],
                   sig[0] * sig[5] * corr[4], sig[1] * sig[5] * corr[8], sig[2] * sig[5] * corr[11],
                   sig[3] * sig[5] * corr[13], sig[4] * sig[5] * corr[14], sig[5] ** 2]).reshape(6, 6)

    xmu = np.expand_dims(x - mu, axis=-1)
    z = np.squeeze(xmu.transpose(0, 1, 2, 3, 4, 5, 7, 6) @ np.linalg.inv(cv) @ xmu)
    denom = np.power(2 * np.pi, 6 / 2) * np.sqrt(np.linalg.det(cv))
    return np.exp(-z / 2) / denom


def corr_bivariate(x, mu, sig, corr, coef):
    assert mu.shape[0] == sig.shape[0] == corr.shape[0] == coef.shape[0]
    z = np.zeros_like(x[:, :, 0])
    for i in range(coef.shape[0]):
        z += coef[i] * bivariate_normal(x, mu[i], sig[i], corr[i])
    return z

def corr_trivariate(x, mu, sig, corr, coef):
    assert mu.shape[0] == sig.shape[0] == corr.shape[0] == coef.shape[0]
    z = np.zeros_like(x[:, :, :, 0])
    for i in range(coef.shape[0]):
        z += coef[i] * trivariate_normal(x, mu[i], sig[i], corr[i])
    return z

def corr_quadvariate(x, mu, sig, corr, coef):
    assert mu.shape[0] == sig.shape[0] == corr.shape[0] == coef.shape[0]
    z = np.zeros_like(x[:, :, :, :, 0])
    for i in range(coef.shape[0]):
        z += coef[i] * quadvariate_normal(x, mu[i], sig[i], corr[i])
    return z

def corr_hexavariate(x, mu, sig, corr, coef):
    assert mu.shape[0] == sig.shape[0] == corr.shape[0] == coef.shape[0]
    z = np.zeros_like(x[:, :, :, :, :, :, 0])
    for i in range(coef.shape[0]):
        z += coef[i] * hexavariate_normal(x, mu[i], sig[i], corr[i])
    return z

class BiGMM():
    def __init__(self):
        pass
    def __call__(self, y, mu ,sig, corr, coef, loss_out = True, cv_out = False):
        y = y.unsqueeze(-2)
        corr = corr.squeeze(-1)
        cv = torch.stack((sig[:, :, :, 0] ** 2, sig[:, :, :, 0] * sig[:, :, :, 1] * corr,
                        sig[:, :, :, 0] * sig[:, :, :, 1] * corr, sig[:, :, :, 1] ** 2),
                       dim=-1).reshape(sig.shape[0], sig.shape[1], sig.shape[2], 2, 2)
        inv_cv = batchedInv(cv)
        if cv_out and not loss_out:
            return cv
        else:
            if torch.sum(torch.isnan(inv_cv)) > 0:
                print(sig[0,0,:], corr[0,0,:])

            xmu = (y - mu).unsqueeze(-1)

            nll = 0.5 * (torch.log(cv[:, :, :, 0, 0] * cv[:, :, :, 1, 1] - cv[:, :, :, 1, 0] * cv[:, :, :, 0, 1])
                        + torch.matmul(torch.matmul(xmu.transpose(-1, -2), inv_cv), xmu).squeeze(-1).squeeze(-1)).squeeze(-1)

        if loss_out and cv_out:
            return nll, cv
        else:
            return nll

class BiGMM_old():
    def __init__(self):
        pass
    def __call__(self, y, mu ,sig, corr, coef, loss_out = True, cv_out = False):
        y = y.unsqueeze(-2)
        corr = corr.squeeze(-1)
        cv = torch.stack((sig[:, :, :, 0] ** 2, sig[:, :, :, 0] * sig[:, :, :, 1] * corr,
                        sig[:, :, :, 0] * sig[:, :, :, 1] * corr, sig[:, :, :, 1] ** 2),
                       dim=-1).reshape(sig.shape[0], sig.shape[1], sig.shape[2], 2, 2)
        inv_cv = batchedInv(cv)
        if cv_out and not loss_out:
            return cv
        else:
            if torch.sum(torch.isnan(inv_cv)) > 0:
                print(sig[0,0,:], corr[0,0,:])

            xmu = (y - mu).unsqueeze(-1)

            terms = -0.5 * (torch.log(cv[:, :, :, 0, 0] * cv[:, :, :, 1, 1] - cv[:, :, :, 1, 0] * cv[:, :, :, 0, 1])
                        + torch.matmul(torch.matmul(xmu.transpose(-1, -2), inv_cv), xmu).squeeze(-1).squeeze(-1)
                            + torch.log(torch.tensor(2 * np.pi)))

            nll = -torch.logsumexp(torch.log(coef) + terms, dim=-1)

        if loss_out and cv_out:
            return nll, cv
        else:
            return nll

class TriGMM():
    def __init__(self):
        pass
    def __call__(self, y, mu, sig, corr, coef, loss_out = True, cv_out = False):

        y = y.unsqueeze(-2)
        zeros = torch.zeros_like(sig[:,:,:,0])
        cv = torch.stack((sig[:, :, :, 0] ** 2, sig[:, :, :, 0] * sig[:, :, :, 1] * corr[:, :, :, 0], sig[:, :, :, 0] * sig[:, :, :, 2] * corr[:, :, :, 1],
                sig[:, :, :, 0] * sig[:, :, :, 1] * corr[:, :, :, 0], sig[:, :, :, 1] ** 2, sig[:, :, :, 1] * sig[:, :, :, 2] * corr[:, :, :, 2],
                sig[:, :, :, 0] * sig[:, :, :, 2] * corr[:, :, :, 1], sig[:, :, :, 1] * sig[:, :, :, 2] * corr[:, :, :, 2], sig[:, :, :, 2] ** 2), dim = -1).reshape(sig.shape[0], sig.shape[1], sig.shape[2], 3, 3)

        inv_cv = batchedInv(cv)
        #inv_cv = torch.tensor(np.linalg.inv(DCN(cv))).to(y.device)
        #inv_cv = torch.inverse(cv)
        if torch.sum(torch.isnan(inv_cv)) > 0:
            print('inv_cv')
            print(sig[0, 0, :], corr[0, 0, :])
        if cv_out and not loss_out:
            return cv
        else:
            xmu = (y - mu).unsqueeze(-1)
            nll = 0.5 * (torch.logdet(cv) + torch.matmul(torch.matmul(xmu.transpose(-1, -2), inv_cv), xmu).squeeze(-1).squeeze(-1)).squeeze(-1)
        if loss_out and cv_out:
            return nll, cv
        else:
            return nll


class TriGMM_old():
    def __init__(self):
        pass
    def __call__(self, y, mu, sig, corr, coef, loss_out = True, cv_out = False):

        y = y.unsqueeze(-2)
        zeros = torch.zeros_like(sig[:,:,:,0])
        cv = torch.stack((sig[:, :, :, 0] ** 2, sig[:, :, :, 0] * sig[:, :, :, 1] * corr[:, :, :, 0], sig[:, :, :, 0] * sig[:, :, :, 2] * corr[:, :, :, 1],
                sig[:, :, :, 0] * sig[:, :, :, 1] * corr[:, :, :, 0], sig[:, :, :, 1] ** 2, sig[:, :, :, 1] * sig[:, :, :, 2] * corr[:, :, :, 2],
                sig[:, :, :, 0] * sig[:, :, :, 2] * corr[:, :, :, 1], sig[:, :, :, 1] * sig[:, :, :, 2] * corr[:, :, :, 2], sig[:, :, :, 2] ** 2), dim = -1).reshape(sig.shape[0], sig.shape[1], sig.shape[2], 3, 3)

        #inv_cv = batchedInv(cv)
        inv_cv = torch.tensor(np.linalg.inv(DCN(cv))).to(y.device)

        #inv_cv = torch.inverse(cv)
        mode_num = coef.shape[-1]
        if torch.sum(torch.isnan(inv_cv)) > 0:
            print('inv_cv')
            print(sig[0, 0, :], corr[0, 0, :])
        if cv_out and not loss_out:
            return cv
        else:
            xmu = (y - mu).unsqueeze(-1)
            #det = torch.abs(cv[:, :, :, 0, 0]*(cv[:, :, :, 1, 1]*cv[:, :, :, 2, 2]-cv[:, :, :, 1, 2]*cv[:, :, :, 2, 1])-cv[:, :, :, 0, 1]*(cv[:, :, :, 1, 0]*cv[:, :, :, 2, 2]-cv[:, :, :, 1, 2]*cv[:, :, :, 2, 0])+cv[:, :, :, 0, 2]*(cv[:, :, :, 1, 0]*cv[:, :, :, 2, 1]-cv[:, :, :, 1, 1]*cv[:, :, :, 2, 0]))
            #terms = -0.5 * (torch.log(det) + torch.matmul(torch.matmul(xmu.transpose(-1, -2), inv_cv), xmu).squeeze(-1).squeeze(-1) + torch.log(torch.tensor(2 * np.pi)))
            terms = -0.5 * (torch.logdet(cv) + torch.matmul(torch.matmul(xmu.transpose(-1, -2), inv_cv), xmu).squeeze(-1).squeeze(-1) + torch.log(torch.tensor(2 * np.pi)))
            nll = -torch.logsumexp(torch.log(coef) + terms, dim = -1)
            #print(nll.shape, cv.shape)

        if loss_out and cv_out:
            return nll, cv
        else:
            return nll


class QuadGMM():
    def __init__(self):
        pass
    def __call__(self, y, mu, sig, corr, coef, loss = True, cv = False):

        y = y.unsqueeze(-2)
        zeros = torch.zeros_like(sig[:,:,:,0])
        L = torch.stack(
            (sig[:,:,:,0], corr[:,:,:,0], corr[:,:,:,1], corr[:,:,:,2],
             zeros, sig[:,:,:,1], corr[:,:,:,3], corr[:,:,:,4],
             zeros, zeros, sig[:,:,:,2], corr[:,:,:,5],
             zeros, zeros, zeros, sig[:,:,:,3]),
            dim=-1).reshape(sig.shape[0], sig.shape[1], sig.shape[2], 4, 4)
        inv_cv = torch.matmul(L.transpose(-1, -2), L)
        if cv and not loss:
            return batchedInv(inv_cv)
        else:
            log_det = -2 * torch.sum(torch.log(torch.diagonal(L, dim1=-2, dim2=-1)), dim=-1)
            if torch.sum(torch.isnan(log_det)) > 0:
                print(sig[0, 0, :], corr[0, 0, :])
            #print(y.shape, mu.shape)
            xmu = (y - mu).unsqueeze(-1)
            
            terms = -0.5 * (log_det
                            + torch.matmul(torch.matmul(xmu.transpose(-1, -2), inv_cv), xmu).squeeze(-1).squeeze(-1)
                            + torch.log(torch.tensor(2 * np.pi)))

            nll = -torch.logsumexp(torch.log(coef) + terms, dim=-1)

        if loss and cv:
            return nll, batchedInv(inv_cv)
        else:
            return nll

class HexaGMM():
    def __init__(self):
        pass
    def __call__(self, y, mu, sig, corr, coef, loss = True, cv = False):
        y = y.unsqueeze(-2)
        zeros = torch.zeros_like(sig[:,:,:,0])
        L = torch.stack((sig[:,:,:,0], corr[:,:,:,0], corr[:,:,:,1], corr[:,:,:,2], corr[:,:,:,3], corr[:,:,:,4],
                          zeros, sig[:,:,:,1], corr[:,:,:,5], corr[:,:,:,6], corr[:,:,:,7] , corr[:,:,:,8],
                         zeros, zeros, sig[:,:,:,2], corr[:,:,:,9], corr[:,:,:,10], corr[:,:,:,11],
                         zeros, zeros, zeros, sig[:,:,:,3], corr[:,:,:,12], corr[:,:,:,13],
                         zeros, zeros, zeros, zeros, sig[:,:,:,4], corr[:,:,:,14],
                         zeros, zeros, zeros, zeros, zeros, sig[:,:,:,5]),
                       dim=-1).reshape(sig.shape[0], sig.shape[1], sig.shape[2], 6, 6)
        inv_cv = torch.matmul(L.transpose(-1, -2), L)
        if cv and not loss:
            return batchedInv(inv_cv)
        else:

            log_det = -2*torch.sum(torch.log(torch.diagonal(L, dim1=-2, dim2 = -1)), dim = -1)
            if torch.sum(torch.isnan(log_det)) > 0:
                print(sig[0,0,:], corr[0,0,:])
            #print(y.shape, mu.shape)
            xmu = (y - mu).unsqueeze(-1)

            terms = -0.5 * (log_det
            + torch.matmul(torch.matmul(xmu.transpose(-1, -2), inv_cv), xmu).squeeze(-1).squeeze(-1)
            + torch.log(torch.tensor(2*np.pi)))

            nll = -torch.logsumexp(torch.log(coef) + terms, dim=-1)

        if loss and cv:
            return nll, batchedInv(inv_cv)
        else :
            return nll


def gmm_criterion(D_s):
    criterion = None
    if  D_s== 2:
        criterion = BiGMM()
    elif D_s == 3:
        criterion = TriGMM()
    elif D_s == 4:
        criterion = QuadGMM()
    elif D_s == 6:
        criterion = HexaGMM()
    else:
        print('NOT IMPLEMENTED : GMM')
    return criterion

class gmm_sample():
    def __init__(self, D_s):
        self.D_s = D_s
    def __call__(self, mu, L):
        original_shape = mu.shape
        mu = mu.view(-1, self.D_s)
        L = L.view(-1, self.D_s, self.D_s)
        try:
            distrib = torch.distributions.MultivariateNormal(loc=mu, covariance_matrix=L)
            sampled_mu = distrib.rsample()
            return sampled_mu.view(original_shape)
        except Exception as e:
            print(e)
            return None
        
## Utility classes

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

if __name__ == '__main__':
    print('hello!')
    main()

## Neural Network Blocks

def MLP_layers(cfg, D_Agent, nl_type='RL', batch_norm=False, drop_out=False):
    layers = []
    nl_dict = {'RL': nn.ReLU(), 'TH': nn.Tanh(), 'LR': nn.LeakyReLU(0.2)}
    nl = nl_dict[nl_type]

    for i in range(1, len(cfg)):
        if i != len(cfg) - 1:
            layers += [('FC' + str(i) + '0', nn.Linear(cfg[i - 1], cfg[i]))]
            if batch_norm:
                layers += [('BN' + str(i) + '0', nn.BatchNorm1d(D_agent))]
            if drop_out:
                layers += [('DO' + str(i) + '0', nn.Dropout(p=0.5))]
            layers += [(nl_type + str(i) + '0', nl)]
        else:
            layers += [('FC' + str(i) + '0', nn.Linear(cfg[i - 1], cfg[i]))]
            layers[-1][-1].bias.requires_grad = False

    return nn.Sequential(OrderedDict(layers))

def Res_layers(cfg, D_Agent, nl_type='RL', batch_norm=False, drop_out=False):
    meta_layers = []
    nl_dict = {'RL': nn.ReLU(), 'TH': nn.Tanh(), 'LR': nn.LeakyReLU(0.2)}
    nl = nl_dict[nl_type]

    for i in range(1, len(cfg)):
        layers = []
        for j in range(2):
            layers += [('FC' + str(i + 1) + str(j), nn.Linear(cfg[i], cfg[i]))]
            if batch_norm:
                layers += [('BN' + str(i + 1) + str(j), nn.BatchNorm1d(D_agent))]
            if drop_out:
                layers += [('DO' + str(i + 1) + str(j), nn.Dropout(p=0.5))]
            if j == 0:
                layers += [(nl_type + str(i + 1) + '0', nl)]

        meta_layers.append(nn.Sequential(OrderedDict(layers)))

    return nn.Sequential(*meta_layers)

class MLP_Block(nn.Module):
    def __init__(self, cfg, D_agent, nl_type='RL', batch_norm=False, drop_out=False):
        super(MLP_Block, self).__init__()

        self.FC = MLP_layers(cfg, D_agent, nl_type, batch_norm, drop_out)

    def forward(self, x):
        return self.FC(x)

class Res_Block(nn.Module):
    def __init__(self, cfg, D_agent, nl_type='RL', batch_norm=False, drop_out=False):
        super(Res_Block, self).__init__()

        nl_dict = {'RL': nn.ReLU(), 'TH': nn.Tanh(), 'LR': nn.LeakyReLU(0.2)}
        self.nl = nl_dict[nl_type]
        self.cfg = cfg

        self.FC1 = MLP_layers(cfg[:2], D_agent, nl_type, batch_norm=False, drop_out=False)
        self.RS = Res_layers(cfg[1:-1], D_agent, nl_type, batch_norm, drop_out)
        self.FC2 = MLP_layers(cfg[-2:], D_agent, nl_type, batch_norm=False, drop_out=False)

    def forward(self, x):
        x = self.FC1(x)
        for m in self.RS.children():
            x = self.nl(m(x) + x)
        x = self.FC2(x)
        return x

def cfg_Block(block_type, cfg, D_agent, nl_type='RL', batch_norm=False, drop_out=False):
    if block_type == 'mlp':
        block = MLP_Block(cfg, D_agent, nl_type, batch_norm, drop_out)
    elif block_type =='res':
        block = Res_Block(cfg, D_agent, nl_type, batch_norm, drop_out)
    else:
        print("NOT IMPLEMENTED : cfg_Block")
    return block

## Architecture

class Module_VAINS(nn.Module):
    def __init__(self, cfg_enc, cfg_com, cfg_att, cfg_dec, cfg_mu, cfg_sig, cfg_corr, cfg_coef, D_att, D_att_num, D_agent, block_type, att_type, dropout):
        super(Module_VAINS, self).__init__()

        self.D_att = D_att
        self.D_att_num = D_att_num
        self.D_agent = D_agent
        self.block_type = block_type
        self.att_type = att_type
        self.dropout = dropout

        self.enc = cfg_Block(block_type, cfg_enc, D_agent, 'RL', False, False)
        self.com = cfg_Block(block_type, cfg_com, D_agent, 'RL', False, False)
        self.att = cfg_Block(block_type, cfg_att, D_agent, 'RL', False, False)

        self.distort = None
        if self.att_type == 'distort':
            self.distort = nn.Sequential(OrderedDict([
                ('L1', nn.Linear(1, 10)),
                ('T1', nn.Tanh()),
                ('L2', nn.Linear(10, 1, bias=False))
            ]))

        self.dec = cfg_Block(block_type, cfg_dec, D_agent, 'RL', False, False)
        self.mu_dec = cfg_Block(block_type, cfg_mu, D_agent, 'RL', False, False)
        self.sig_dec = cfg_Block(block_type, cfg_sig, D_agent, 'RL', False, False)
        self.corr_dec = cfg_Block(block_type, cfg_corr, D_agent, 'RL', False, False)
        self.coef_dec = cfg_Block(block_type, cfg_coef, D_agent, 'RL', False, False)

        if block_type == 'mlp':
            self.D_k = self.coef_dec.FC[-1].out_features
            self.D_s = int(self.mu_dec.FC[-1].out_features / self.D_k)
        elif block_type == 'res':
            self.D_k = self.coef_dec.FC2[-1].out_features
            self.D_s = int(self.mu_dec.FC2[-1].out_features / self.D_k)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if type(m.bias) != type(None):
                    m.bias.data.zero_()

    def forward(self, x):
        softmax = nn.Softmax(dim=-1)

        e_s = self.enc(x)
        e_c = self.com(x)
        a = self.att(x)

        mask1 = torch.zeros(x.shape[0], x.shape[1], x.shape[1]).to(x.device)
        for i in range(x.shape[0]):
            m = (x[i,:,-1] != 1).nonzero().squeeze()
            if m.dim() != 1:
                print(x[i,:,-1])
                print(m)
                print('wrong')  #
            for j in m:
                mask1[i][j, :] = 1
                mask1[i][:, j] = 1

        mask1 = (mask1 + torch.eye(x.shape[1], x.shape[1]).to(x.device)) * 10000
        mask2 = (1 - torch.eye(x.shape[1], x.shape[1])).to(x.device)
        c = None
        if self.att_type == 'single':
            w = F.dropout(softmax(-torch.add(pdist(a), mask1)), p = self.dropout, training = self.training)
            p = torch.bmm(torch.mul(w, mask2), e_c)
            c = torch.cat((p, e_s), dim=-1)

        elif self.att_type == 'multi':
            p_list = [e_s]
            for i in range(self.D_att_num):
                p_list.append(
                    torch.bmm(torch.mul(F.dropout(softmax(-torch.add(pdist(a[:, :, self.D_att * i : self.D_att * (i + 1)]), mask1)), p = self.dropout, training = self.training), mask2),
                              e_c))
            c = torch.cat(p_list, dim=-1)

        elif self.att_type == 'distort':
            q1 = pdist(a).unsqueeze(-1)
            q2 = (self.distort(q1) + q1).squeeze() # Skip-connection
            w = F.dropout(softmax(-torch.add(q2, mask1)), p = self.dropout, training = self.training)
            p = torch.bmm(torch.mul(w, mask2), e_c)
            c = torch.cat((p, e_s), dim=-1)
        else:
            print('NOT IMPLEMENTED : ATTENTION FORWARD')

        d = self.dec(c)

        mu = self.mu_dec(d)
        sig = self.sig_dec(d)/10. # divided by 10.0 due to numeircal stability
        corr = self.corr_dec(d)/10.
        coef = torch.softmax(self.coef_dec(d), dim=-1)  # sum(coef) = 1

        if self.D_s == 2:
            mu = mu.reshape(mu.shape[0], mu.shape[1], self.D_k, 2)
            sig = F.softplus(sig.reshape(sig.shape[0], sig.shape[1], self.D_k, 2))
            corr = F.softsign(corr.reshape(corr.shape[0], corr.shape[1], self.D_k, 1))

        elif self.D_s == 3:
            mu = mu.reshape(mu.shape[0], mu.shape[1], self.D_k, 3)
            sig = F.softplus(sig.reshape(sig.shape[0], sig.shape[1], self.D_k, 3))
            corr = F.softsign(corr.reshape(corr.shape[0], corr.shape[1], self.D_k, 3))

        elif self.D_s == 6:
            mu = mu.reshape(mu.shape[0], mu.shape[1], self.D_k, 6)
            sig = F.softplus(sig.reshape(sig.shape[0], sig.shape[1], self.D_k, 6))
            corr = F.softsign(corr.reshape(corr.shape[0], corr.shape[1], self.D_k, 15))
        else:
            print("NOT IMPLEMENTED : D_s reshaping")

        return mu, sig, corr, coef

class Module_MLP(nn.Module):
    def __init__(self, cfg_dec, cfg_mu, cfg_sig, cfg_corr, cfg_coef, D_agent, block_type):
        super(Module_MLP, self).__init__()

        self.D_agent = D_agent
        self.block_type = block_type

        self.dec = cfg_Block(block_type, cfg_dec, D_agent, 'RL', False, False)
        self.mu_dec = cfg_Block(block_type, cfg_mu, D_agent, 'RL', False, False)
        self.sig_dec = cfg_Block(block_type, cfg_sig, D_agent, 'RL', False, False)
        self.corr_dec = cfg_Block(block_type, cfg_corr, D_agent, 'RL', False, False)
        self.coef_dec = cfg_Block(block_type, cfg_coef, D_agent, 'RL', False, False)

        if block_type == 'mlp':
            self.D_k = self.coef_dec.FC[-1].out_features
            self.D_s = int(self.mu_dec.FC[-1].out_features / self.D_k)
        elif block_type == 'res':
            self.D_k = self.coef_dec.FC2[-1].out_features
            self.D_s = int(self.mu_dec.FC2[-1].out_features / self.D_k)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if type(m.bias) != type(None):
                    m.bias.data.zero_()

    def forward(self, x):

        d = self.dec(x)

        mu = self.mu_dec(d)
        sig = self.sig_dec(d) / 10.
        corr = self.corr_dec(d) / 10.
        coef = torch.softmax(self.coef_dec(d), dim=-1)  # sum(coef) = 1

        if self.D_s == 2:
            mu = mu.reshape(mu.shape[0], mu.shape[1], self.D_k, 2)
            sig = F.softplus(sig.reshape(sig.shape[0], sig.shape[1], self.D_k, 2))
            corr = F.softsign(corr.reshape(corr.shape[0], corr.shape[1], self.D_k, 1))

        elif self.D_s == 3:
            mu = mu.reshape(mu.shape[0], mu.shape[1], self.D_k, 3)
            sig = F.softplus(sig.reshape(sig.shape[0], sig.shape[1], self.D_k, 3))
            corr = F.softsign(corr.reshape(corr.shape[0], corr.shape[1], self.D_k, 3))

        elif self.D_s == 6:
            mu = mu.reshape(mu.shape[0], mu.shape[1], self.D_k, 6)
            sig = F.softplus(sig.reshape(sig.shape[0], sig.shape[1], self.D_k, 6))
            corr = F.softsign(corr.reshape(corr.shape[0], corr.shape[1], self.D_k, 15))
        else:
            print("NOT IMPLEMENTED : D_s reshaping")

        return mu, sig, corr, coef

class Module_MLP_LSTM(nn.Module):
    def __init__(self, cfg_init, cfg_lstm, cfg_dec, cfg_mu, cfg_sig, cfg_corr, cfg_coef, D_agent, block_type, eval_type):
        super(Module_MLP_LSTM, self).__init__()

        self.D_agent = D_agent
        self.block_type = block_type
        self.eval_type = eval_type

        self.init_hidden = cfg_Block(block_type, cfg_init, D_agent, 'RL', False, False)
        self.init_cell = cfg_Block(block_type, cfg_init, D_agent, 'RL', False, False)
        self.lstm = nn.LSTMCell(cfg_lstm[0], cfg_lstm[1])

        self.dec1 = cfg_Block(block_type, cfg_dec, D_agent, 'RL', False, False)
        self.mu_dec1 = cfg_Block(block_type, cfg_mu, D_agent, 'RL', False, False)
        self.sig_dec1 = cfg_Block(block_type, cfg_sig, D_agent, 'RL', False, False)
        self.corr_dec1 = cfg_Block(block_type, cfg_corr, D_agent, 'RL', False, False)
        self.coef_dec1 = cfg_Block(block_type, cfg_coef, D_agent, 'RL', False, False)

        self.dec2 = cfg_Block(block_type, cfg_dec, D_agent, 'RL', False, False)
        self.mu_dec2 = cfg_Block(block_type, cfg_mu, D_agent, 'RL', False, False)
        self.sig_dec2 = cfg_Block(block_type, cfg_sig, D_agent, 'RL', False, False)
        self.corr_dec2 = cfg_Block(block_type, cfg_corr, D_agent, 'RL', False, False)
        self.coef_dec2 = cfg_Block(block_type, cfg_coef, D_agent, 'RL', False, False)

        if block_type == 'mlp':
            self.D_k = self.coef_dec1.FC[-1].out_features
            self.D_s = int(self.mu_dec1.FC[-1].out_features / self.D_k)
        elif block_type == 'res':
            self.D_k = self.coef_dec1.FC2[-1].out_features
            self.D_s = int(self.mu_dec1.FC2[-1].out_features / self.D_k)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if type(m.bias) != type(None):
                    m.bias.data.zero_()

    def initialize(self, x, new_hidden, hidden = None):
        x = x.view(x.shape[0] * x.shape[1], -1)
        init_mask = (new_hidden[:, :, 0] > 0).flatten().unsqueeze(-1).float()
        h = self.init_hidden(x)
        c = self.init_cell(x)
        if hidden is None:
            hidden = (torch.zeros_like(h), torch.zeros_like(c))
        return (hidden[0]*(1-init_mask)) + h*init_mask, (hidden[1]*(1-init_mask))+c*init_mask

    def forward(self, x, hidden):

        batch_num = x.shape[0]
        agent_num = x.shape[1]
        x = x.view(batch_num*agent_num, -1)
        hidden = self.lstm(x, hidden)
        c = hidden[0].view(batch_num, agent_num, -1)

        d1 = self.dec1(c)
        d2 = self.dec2(c)

        mu1 = self.mu_dec1(d1)
        sig1 = self.sig_dec1(d1)   
        corr1 = self.corr_dec1(d1) 
        coef1 = torch.softmax(self.coef_dec1(d1), dim=-1)  # sum(coef) = 1

        mu1 = mu1.reshape(mu1.shape[0], mu1.shape[1], self.D_k, 3)
        sig1 = F.softplus(sig1.reshape(sig1.shape[0], sig1.shape[1], self.D_k, 3))
        corr1 = F.softsign(corr1.reshape(corr1.shape[0], corr1.shape[1], self.D_k, 3))

        mu2 = self.mu_dec2(d2)
        sig2 = self.sig_dec2(d2)   
        corr2 = self.corr_dec2(d2) 
        coef2 = torch.softmax(self.coef_dec2(d2), dim=-1)  # sum(coef) = 1
        
        mu2 = mu2.reshape(mu2.shape[0], mu2.shape[1], self.D_k, 3)
        sig2 = F.softplus(sig2.reshape(sig2.shape[0], sig2.shape[1], self.D_k, 3))
        corr2 = F.softsign(corr2.reshape(corr2.shape[0], corr2.shape[1], self.D_k, 3))

        return (mu1, sig1, corr1, coef1), (mu2, sig2, corr2, coef2), hidden

        return mu, sig, corr, coef, hidden

class Module_MLP_LSTM_single(nn.Module):
    def __init__(self, cfg_init, cfg_lstm, cfg_dec, cfg_mu, cfg_sig, cfg_corr, cfg_coef, D_agent, block_type, eval_type):
        super(Module_MLP_LSTM_single, self).__init__()

        self.D_agent = D_agent
        self.block_type = block_type
        self.eval_type = eval_type

        self.init_hidden = cfg_Block(block_type, cfg_init, D_agent, 'RL', False, False)
        self.init_cell = cfg_Block(block_type, cfg_init, D_agent, 'RL', False, False)
        self.lstm = nn.LSTMCell(cfg_lstm[0], cfg_lstm[1])

        self.dec1 = cfg_Block(block_type, cfg_dec, D_agent, 'RL', False, False)
        self.mu_dec1 = cfg_Block(block_type, cfg_mu, D_agent, 'RL', False, False)
        self.sig_dec1 = cfg_Block(block_type, cfg_sig, D_agent, 'RL', False, False)
        self.corr_dec1 = cfg_Block(block_type, cfg_corr, D_agent, 'RL', False, False)

        self.dec2 = cfg_Block(block_type, cfg_dec, D_agent, 'RL', False, False)
        self.mu_dec2 = cfg_Block(block_type, cfg_mu, D_agent, 'RL', False, False)
        self.sig_dec2 = cfg_Block(block_type, cfg_sig, D_agent, 'RL', False, False)
        self.corr_dec2 = cfg_Block(block_type, cfg_corr, D_agent, 'RL', False, False)

        if block_type == 'mlp':
            self.D_k = 1
            self.D_s = int(self.mu_dec1.FC[-1].out_features / self.D_k)
        elif block_type == 'res':
            self.D_k = 1
            self.D_s = int(self.mu_dec1.FC2[-1].out_features / self.D_k)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if type(m.bias) != type(None):
                    m.bias.data.zero_()

    def initialize(self, x, new_hidden, hidden = None):
        x = x.view(x.shape[0] * x.shape[1], -1)
        init_mask = (new_hidden[:, :, 0] > 0).flatten().unsqueeze(-1).float()
        h = self.init_hidden(x)
        c = self.init_cell(x)
        if hidden is None:
            hidden = (torch.zeros_like(h), torch.zeros_like(c))
        return (hidden[0]*(1-init_mask)) + h*init_mask, (hidden[1]*(1-init_mask))+c*init_mask

    def forward(self, x, hidden):

        batch_num = x.shape[0]
        agent_num = x.shape[1]
        x = x.view(batch_num*agent_num, -1)
        hidden = self.lstm(x, hidden)
        c = hidden[0].view(batch_num, agent_num, -1)

        d1 = self.dec1(c)
        d2 = self.dec2(c)

        mu1 = self.mu_dec1(d1)
        sig1 = self.sig_dec1(d1)   
        corr1 = self.corr_dec1(d1) 

        mu1 = mu1.reshape(mu1.shape[0], mu1.shape[1], self.D_k, 3)
        sig1 = F.softplus(sig1.reshape(sig1.shape[0], sig1.shape[1], self.D_k, 3))
        corr1 = F.softsign(corr1.reshape(corr1.shape[0], corr1.shape[1], self.D_k, 3))

        mu2 = self.mu_dec2(d2)
        sig2 = self.sig_dec2(d2)   
        corr2 = self.corr_dec2(d2) 
        
        mu2 = mu2.reshape(mu2.shape[0], mu2.shape[1], self.D_k, 3)
        sig2 = F.softplus(sig2.reshape(sig2.shape[0], sig2.shape[1], self.D_k, 3))
        corr2 = F.softsign(corr2.reshape(corr2.shape[0], corr2.shape[1], self.D_k, 3))

        return (mu1, sig1, corr1), (mu2, sig2, corr2), hidden

        return mu, sig, corr, coef, hidden

class Module_MLP_AOUP(nn.Module):
    def __init__(self, cfg_init, cfg_lstm, cfg_dec, cfg_mu, cfg_sig, cfg_corr, cfg_coef, D_agent, block_type, eval_type):
        super(Module_MLP_AOUP, self).__init__()

        self.D_agent = D_agent
        self.block_type = block_type
        self.eval_type = eval_type

        self.init_hidden = cfg_Block(block_type, cfg_init, D_agent, 'RL', False, False)
        self.init_cell = cfg_Block(block_type, cfg_init, D_agent, 'RL', False, False)
        self.lstm = nn.LSTMCell(cfg_lstm[0], cfg_lstm[1])

        self.dec1 = cfg_Block(block_type, cfg_dec, D_agent, 'RL', False, False)
        self.mu_dec1 = cfg_Block(block_type, cfg_mu, D_agent, 'RL', False, False)
        self.sig_dec1 = cfg_Block(block_type, cfg_sig, D_agent, 'RL', False, False)
        self.corr_dec1 = cfg_Block(block_type, cfg_corr, D_agent, 'RL', False, False)
        self.coef_dec1 = cfg_Block(block_type, cfg_coef, D_agent, 'RL', False, False)

        self.dec2 = cfg_Block(block_type, cfg_dec, D_agent, 'RL', False, False)
        self.mu_dec2 = cfg_Block(block_type, cfg_mu, D_agent, 'RL', False, False)
        self.sig_dec2 = cfg_Block(block_type, cfg_sig, D_agent, 'RL', False, False)
        self.corr_dec2 = cfg_Block(block_type, cfg_corr, D_agent, 'RL', False, False)
        self.coef_dec2 = cfg_Block(block_type, cfg_coef, D_agent, 'RL', False, False)

        if block_type == 'mlp':
            self.D_k = self.coef_dec1.FC[-1].out_features
            self.D_s = int(self.mu_dec1.FC[-1].out_features / self.D_k)
        elif block_type == 'res':
            self.D_k = self.coef_dec1.FC2[-1].out_features
            self.D_s = int(self.mu_dec1.FC2[-1].out_features / self.D_k)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if type(m.bias) != type(None):
                    m.bias.data.zero_()

    def initialize(self, x):
        batch_num = x.shape[0]
        agent_num = x.shape[1]
        x = x.reshape(x.shape[0] * x.shape[1], -1)
        h = self.init_hidden(x)
        c = self.init_cell(x)
        return (h.reshape(batch_num, agent_num, -1), c.reshape(batch_num, agent_num, -1))

    def forward(self, x, hidden, cell, R_const, test = False):
        #print(x.shape, hidden[0].shape, hidden[1].shape)
        softmax = nn.Softmax(dim=-1)

        batch_num = x.shape[0]
        agent_num = x.shape[1]
        x = x.reshape(batch_num * agent_num, -1)
        hidden = hidden.view(batch_num * agent_num, -1)
        cell = cell.view(batch_num * agent_num, -1)

        hidden, cell = self.lstm(x, (hidden, cell))
        if test:
            b = hidden.view(batch_num, agent_num, -1)
            x = x.view(batch_num, agent_num, -1)

            d1 = self.dec1(b)
            d2 = self.dec2(b)

            mu1 = self.mu_dec1(d1)
            sig1 = self.sig_dec1(d1)   
            corr1 = self.corr_dec1(d1) 
            coef1 = torch.softmax(self.coef_dec1(d1), dim=-1)  # sum(coef) = 1

            mu1 = mu1.reshape(mu1.shape[0], mu1.shape[1], self.D_k, 2)
            sig1 = F.softplus(sig1.reshape(sig1.shape[0], sig1.shape[1], self.D_k, 2))
            corr1 = F.softsign(corr1.reshape(corr1.shape[0], corr1.shape[1], self.D_k, 1))

            mu2 = self.mu_dec2(d2)
            sig2 = self.sig_dec2(d2)   
            corr2 = self.corr_dec2(d2) 
            coef2 = torch.softmax(self.coef_dec2(d2), dim=-1)  # sum(coef) = 1
            
            mu2 = mu2.reshape(mu2.shape[0], mu2.shape[1], self.D_k, 2)
            sig2 = F.softplus(sig2.reshape(sig2.shape[0], sig2.shape[1], self.D_k, 2))
            corr2 = F.softsign(corr2.reshape(corr2.shape[0], corr2.shape[1], self.D_k, 1))

            return (mu1, sig1, corr1, coef1), (mu2, sig2, corr2, coef2), hidden.view(batch_num, agent_num, -1), cell.view(batch_num, agent_num, -1)
        else:
            return hidden.view(batch_num, agent_num, -1), cell.view(batch_num, agent_num, -1)

class Module_GAT_DET(nn.Module):
    def __init__(self, cfg_enc, cfg_att, cfg_dec, D_att, D_att_num, D_agent, block_type, att_type, dropout):
        super(Module_GAT_DET, self).__init__()

        self.D_att = D_att
        self.D_att_num = D_att_num
        self.D_agent = D_agent
        self.block_type = block_type
        self.att_type = att_type
        self.dropout = dropout

        self.key = cfg_Block(block_type, cfg_enc, D_agent, 'RL', False, False)
        self.query = cfg_Block(block_type, cfg_enc, D_agent, 'RL', False, False)
        self.value = cfg_Block(block_type, cfg_enc, D_agent, 'RL', False, False)
        if self.att_type == 'gat':
            self.att = cfg_Block(block_type, cfg_att, D_agent, 'RL', False, False)
        elif self.att_type == 'add':
            self.att1 = cfg_Block(block_type, cfg_att, D_agent, 'RL', False, False)
            self.att2 = cfg_Block(block_type, cfg_att, D_agent, 'RL', False, False)
            self.att3 = cfg_Block(block_type, cfg_att, D_agent, 'RL', False, False)
        elif self.att_type == 'mul':
            self.att = None
        else:
            print('NOT IMPLEMENTED : gat att block creation')

        self.dec = cfg_Block(block_type, cfg_dec, D_agent, 'RL', False, False)
        self.classifier = nn.Module()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if type(m.bias) != type(None):
                    m.bias.data.zero_()

    def forward(self, x):
        softmax = nn.Softmax(dim=-1)

        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        c = None
        p_list = [v]

        mask_const = 10000
        
        mask = torch.zeros(x.shape[0], x.shape[1], x.shape[1]).to(x.device)
        mask = mask+torch.eye(x.shape[1], x.shape[1]).to(x.device)*mask_const

        if self.att_type == 'gat':
            for i in range(self.D_att_num):
                ky = k[:, :, self.D_att * i: self.D_att * (i + 1)]
                kz = [ky for _ in range(ky.shape[1])]
                qy = q[:, :, self.D_att * i: self.D_att * (i + 1)]
                qz = [qy for _ in range(qy.shape[1])]
                z = torch.cat((torch.stack(kz, dim=-2), torch.stack(qz, dim=-3)), -1)
                w = F.dropout(softmax(torch.add(F.leaky_relu(self.att(z), 0.2).squeeze(-1), -mask)), p = self.dropout, training = self.training)
                p_list.append(torch.bmm(w, v[:, :, self.D_att * i: self.D_att * (i + 1)]))

        elif self.att_type == 'add':
            for i in range(self.D_att_num):
                ky = k[:, :, self.D_att * i: self.D_att * (i + 1)]
                kz = torch.stack([ky for _ in range(ky.shape[1])], dim = -2)
                qy = q[:, :, self.D_att * i: self.D_att * (i + 1)]
                qz = torch.stack([qy for _ in range(qy.shape[1])], dim = -3)
                f = self.att3(F.tanh(self.att1(kz)+self.att2(qz)))
                w = F.dropout(softmax(torch.add(f, -mask)), p=self.dropout, training=self.training)
                p_list.append(torch.bmm(w, v[:, :, self.D_att * i: self.D_att * (i + 1)]))

        elif self.att_type == 'mul':
            for i in range(self.D_att_num):
                ky = k[:, :, self.D_att * i: self.D_att * (i + 1)]
                kz = torch.stack([ky for _ in range(ky.shape[1])], dim = -2)
                qy = q[:, :, self.D_att * i: self.D_att * (i + 1)]
                qz = torch.stack([qy for _ in range(qy.shape[1])], dim = -3)
                w = F.dropout(softmax(torch.add(batchedDot(kz, qz)/np.sqrt(self.D_att), -mask)), p=self.dropout, training=self.training)
                p_list.append(torch.bmm(w, v[:, :, self.D_att * i: self.D_att * (i + 1)]))
        else:
            print('NOT IMPLEMENTED : GAT attention type')
        c = torch.cat(p_list, dim=-1)
        d = self.dec(c)

        return d
    
class Module_GAT(nn.Module):
    def __init__(self, cfg_enc, cfg_att, cfg_dec, cfg_mu, cfg_sig, cfg_corr, cfg_coef, D_att, D_att_num, D_agent, block_type, att_type, dropout):
        super(Module_GAT, self).__init__()

        self.D_att = D_att
        self.D_att_num = D_att_num
        self.D_agent = D_agent
        self.block_type = block_type
        self.att_type = att_type
        self.dropout = dropout

        self.key = cfg_Block(block_type, cfg_enc, D_agent, 'RL', False, False)
        self.query = cfg_Block(block_type, cfg_enc, D_agent, 'RL', False, False)
        self.value = cfg_Block(block_type, cfg_enc, D_agent, 'RL', False, False)
        if self.att_type == 'gat':
            self.att = cfg_Block(block_type, cfg_att, D_agent, 'RL', False, False)
        elif self.att_type == 'add':
            self.att1 = cfg_Block(block_type, [D_att, D_att], D_agent, 'RL', False, False)
            self.att2 = cfg_Block(block_type, [D_att, D_att], D_agent, 'RL', False, False)
            self.att3 = cfg_Block(block_type, [D_att, 1], D_agent, 'RL', False, False)
        elif self.att_type == 'mul':
            self.att = None
        else:
            print('NOT IMPLEMENTED : gat att block creation')

        self.dec = cfg_Block(block_type, cfg_dec, D_agent, 'RL', False, False)
        self.mu_dec = cfg_Block(block_type, cfg_mu, D_agent, 'RL', False, False)
        self.sig_dec = cfg_Block(block_type, cfg_sig, D_agent, 'RL', False, False)
        self.corr_dec = cfg_Block(block_type, cfg_corr, D_agent, 'RL', False, False)
        self.coef_dec = cfg_Block(block_type, cfg_coef, D_agent, 'RL', False, False)

        if block_type == 'mlp':
            self.D_k = self.coef_dec.FC[-1].out_features
            self.D_s = int(self.mu_dec.FC[-1].out_features / self.D_k)
        elif block_type == 'res':
            self.D_k = self.coef_dec.FC2[-1].out_features
            self.D_s = int(self.mu_dec.FC2[-1].out_features / self.D_k)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if type(m.bias) != type(None):
                    m.bias.data.zero_()

    def forward(self, x):
        softmax = nn.Softmax(dim=-1)

        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        c = None
        p_list = [v]

        mask_const = 10000
        mask = torch.zeros(x.shape[0], x.shape[1], x.shape[1]).to(x.device)
        for i in range(x.shape[0]):
            m = (x[i, :, -1] != 1).nonzero().squeeze()
            if m.dim() != 1:
                print(x[i, :, -1])
                print(m)
                print('wrong')  #
            for j in m:
                mask[i][j, :] = mask_const
                mask[i][:, j] = mask_const

        mask = mask+torch.eye(x.shape[1], x.shape[1]).to(x.device)*mask_const

        if self.att_type == 'gat':
            for i in range(self.D_att_num):
                ky = k[:, :, self.D_att * i: self.D_att * (i + 1)]
                kz = [ky for _ in range(ky.shape[1])]
                qy = q[:, :, self.D_att * i: self.D_att * (i + 1)]
                qz = [qy for _ in range(qy.shape[1])]
                z = torch.cat((torch.stack(kz, dim=-2), torch.stack(qz, dim=-3)), -1)
                w = F.dropout(softmax(torch.add(F.leaky_relu(self.att(z), 0.2).squeeze(-1), -mask)), p = self.dropout, training = self.training)
                p_list.append(torch.bmm(w, v[:, :, self.D_att * i: self.D_att * (i + 1)]))
                #p_list.append(torch.bmm(w,v))

        elif self.att_type == 'add':
            for i in range(self.D_att_num):
                ky = k[:, :, self.D_att * i: self.D_att * (i + 1)]
                kz = torch.stack([ky for _ in range(ky.shape[1])], dim = -2)
                qy = q[:, :, self.D_att * i: self.D_att * (i + 1)]
                qz = torch.stack([qy for _ in range(qy.shape[1])], dim = -3)
                f = self.att3(self.att1(kz)+self.att2(qz)).squeeze(-1)
                w = F.dropout(softmax(torch.add(f, -mask)), p=self.dropout, training=self.training)
                p_list.append(torch.bmm(w, v[:, :, self.D_att * i: self.D_att * (i + 1)]))

        elif self.att_type == 'mul':
            for i in range(self.D_att_num):
                ky = k[:, :, self.D_att * i: self.D_att * (i + 1)]
                kz = torch.stack([ky for _ in range(ky.shape[1])], dim = -2)
                qy = q[:, :, self.D_att * i: self.D_att * (i + 1)]
                qz = torch.stack([qy for _ in range(qy.shape[1])], dim = -3)
                w = F.dropout(softmax(torch.add(batchedDot(kz, qz)/np.sqrt(self.D_att), -mask)), p=self.dropout, training=self.training)
                p_list.append(torch.bmm(w, v[:, :, self.D_att * i: self.D_att * (i + 1)]))
        else:
            print('NOT IMPLEMENTED : GAT attention type')

        c = torch.cat(p_list, dim=-1)
        d = self.dec(c)

        mu = self.mu_dec(d)
        sig = self.sig_dec(d) / 10.0  # divided by 10.0 due to numeircal stability
        corr = self.corr_dec(d) /10.0
        coef = torch.softmax(self.coef_dec(d), dim=-1)  # sum(coef) = 1

        if self.D_s == 2:
            mu = mu.reshape(mu.shape[0], mu.shape[1], self.D_k, 2)
            sig = F.softplus(sig.reshape(sig.shape[0], sig.shape[1], self.D_k, 2))
            corr = F.softsign(corr.reshape(corr.shape[0], corr.shape[1], self.D_k, 1))

        elif self.D_s == 3:
            mu = mu.reshape(mu.shape[0], mu.shape[1], self.D_k, 3)
            sig = F.softplus(sig.reshape(sig.shape[0], sig.shape[1], self.D_k, 3))
            corr = F.softsign(corr.reshape(corr.shape[0], corr.shape[1], self.D_k, 3))

        elif self.D_s == 6:
            mu = mu.reshape(mu.shape[0], mu.shape[1], self.D_k, 6)
            sig = F.softplus(sig.reshape(sig.shape[0], sig.shape[1], self.D_k, 6))
            corr = F.softsign(corr.reshape(corr.shape[0], corr.shape[1], self.D_k, 15))
        else:
            print("NOT IMPLEMENTED : D_s reshaping")

        return mu, sig, corr, coef

class Module_GAT_ns(nn.Module):
    def __init__(self, cfg_enc, cfg_att, cfg_dec, cfg_mu, cfg_sig, cfg_corr, cfg_coef, D_att, D_att_num, D_agent, block_type, att_type, dropout):
        super(Module_GAT_ns, self).__init__()

        self.D_att = D_att
        self.D_att_num = D_att_num
        self.D_agent = D_agent
        self.block_type = block_type
        self.att_type = att_type
        self.dropout = dropout

        self.key = cfg_Block(block_type, cfg_enc, D_agent, 'RL', False, False)
        self.query = cfg_Block(block_type, cfg_enc, D_agent, 'RL', False, False)
        self.value = cfg_Block(block_type, cfg_enc, D_agent, 'RL', False, False)
        if self.att_type == 'gat':
            self.att = cfg_Block(block_type, cfg_att, D_agent, 'RL', False, False)
        elif self.att_type == 'add':
            self.att1 = cfg_Block(block_type, [D_att, D_att], D_agent, 'RL', False, False)
            self.att2 = cfg_Block(block_type, [D_att, D_att], D_agent, 'RL', False, False)
            self.att3 = cfg_Block(block_type, [D_att, 1], D_agent, 'RL', False, False)
        elif self.att_type == 'mul':
            self.att = None
        else:
            print('NOT IMPLEMENTED : gat att block creation')

        self.dec = cfg_Block(block_type, cfg_dec, D_agent, 'RL', False, False)
        self.mu_dec = cfg_Block(block_type, cfg_mu, D_agent, 'RL', False, False)
        self.sig_dec = cfg_Block(block_type, cfg_sig, D_agent, 'RL', False, False)
        self.corr_dec = cfg_Block(block_type, cfg_corr, D_agent, 'RL', False, False)
        self.coef_dec = cfg_Block(block_type, cfg_coef, D_agent, 'RL', False, False)

        if block_type == 'mlp':
            self.D_k = self.coef_dec.FC[-1].out_features
            self.D_s = int(self.mu_dec.FC[-1].out_features / self.D_k)
        elif block_type == 'res':
            self.D_k = self.coef_dec.FC2[-1].out_features
            self.D_s = int(self.mu_dec.FC2[-1].out_features / self.D_k)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if type(m.bias) != type(None):
                    m.bias.data.zero_()

    def forward(self, x):
        softmax = nn.Softmax(dim=-1)

        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        c = None
        p_list = [v]

        mask_const = 10000
        mask = torch.zeros(x.shape[0], x.shape[1], x.shape[1]).to(x.device)
        for i in range(x.shape[0]):
            m = (x[i, :, -1] != 1).nonzero().squeeze()
            if m.dim() != 1:
                print(x[i, :, -1])
                print(m)
                print('wrong')  #
            for j in m:
                mask[i][j, :] = mask_const
                mask[i][:, j] = mask_const

        mask = mask+torch.eye(x.shape[1], x.shape[1]).to(x.device)*mask_const

        if self.att_type == 'gat':
            for i in range(self.D_att_num):
                ky = k[:, :, self.D_att * i: self.D_att * (i + 1)]
                kz = [ky for _ in range(ky.shape[1])]
                qy = q[:, :, self.D_att * i: self.D_att * (i + 1)]
                qz = [qy for _ in range(qy.shape[1])]
                z = torch.cat((torch.stack(kz, dim=-2), torch.stack(qz, dim=-3)), -1)
                w = torch.sigmoid(torch.add(F.leaky_relu(self.att(z), 0.2).squeeze(-1), -mask))/self.D_agent
                p_list.append(torch.bmm(w, v[:, :, self.D_att * i: self.D_att * (i + 1)]))
                #p_list.append(torch.bmm(w,v))

        elif self.att_type == 'add':
            for i in range(self.D_att_num):
                ky = k[:, :, self.D_att * i: self.D_att * (i + 1)]
                kz = torch.stack([ky for _ in range(ky.shape[1])], dim = -2)
                qy = q[:, :, self.D_att * i: self.D_att * (i + 1)]
                qz = torch.stack([qy for _ in range(qy.shape[1])], dim = -3)
                f = self.att3(self.att1(kz)+self.att2(qz)).squeeze(-1)
                w = F.dropout(softmax(torch.add(f, -mask)), p=self.dropout, training=self.training)
                p_list.append(torch.bmm(w, v[:, :, self.D_att * i: self.D_att * (i + 1)]))

        elif self.att_type == 'mul':
            for i in range(self.D_att_num):
                ky = k[:, :, self.D_att * i: self.D_att * (i + 1)]
                kz = torch.stack([ky for _ in range(ky.shape[1])], dim = -2)
                qy = q[:, :, self.D_att * i: self.D_att * (i + 1)]
                qz = torch.stack([qy for _ in range(qy.shape[1])], dim = -3)
                w = F.dropout(softmax(torch.add(batchedDot(kz, qz)/np.sqrt(self.D_att), -mask)), p=self.dropout, training=self.training)
                p_list.append(torch.bmm(w, v[:, :, self.D_att * i: self.D_att * (i + 1)]))
        else:
            print('NOT IMPLEMENTED : GAT attention type')

        c = torch.cat(p_list, dim=-1)
        d = self.dec(c)

        mu = self.mu_dec(d)
        sig = self.sig_dec(d) / 10.0  # divided by 10.0 due to numeircal stability
        corr = self.corr_dec(d) /10.0
        coef = torch.softmax(self.coef_dec(d), dim=-1)  # sum(coef) = 1

        if self.D_s == 2:
            mu = mu.reshape(mu.shape[0], mu.shape[1], self.D_k, 2)
            sig = F.softplus(sig.reshape(sig.shape[0], sig.shape[1], self.D_k, 2))
            corr = F.softsign(corr.reshape(corr.shape[0], corr.shape[1], self.D_k, 1))

        elif self.D_s == 3:
            mu = mu.reshape(mu.shape[0], mu.shape[1], self.D_k, 3)
            sig = F.softplus(sig.reshape(sig.shape[0], sig.shape[1], self.D_k, 3))
            corr = F.softsign(corr.reshape(corr.shape[0], corr.shape[1], self.D_k, 3))

        elif self.D_s == 6:
            mu = mu.reshape(mu.shape[0], mu.shape[1], self.D_k, 6)
            sig = F.softplus(sig.reshape(sig.shape[0], sig.shape[1], self.D_k, 6))
            corr = F.softsign(corr.reshape(corr.shape[0], corr.shape[1], self.D_k, 15))
        else:
            print("NOT IMPLEMENTED : D_s reshaping")

        return mu, sig, corr, coef

class Module_GAT_new(nn.Module):
    def __init__(self, cfg_enc, cfg_att, cfg_dec, cfg_mu, cfg_sig, cfg_corr, cfg_coef, D_att, D_att_num, D_agent, block_type, att_type, dropout):
        super(Module_GAT_new, self).__init__()

        self.D_att = D_att
        self.D_att_num = D_att_num
        self.D_agent = D_agent
        self.block_type = block_type
        self.att_type = att_type
        self.dropout = dropout

        self.key = cfg_Block(block_type, cfg_enc, D_agent, 'RL', False, False)
        self.query = cfg_Block(block_type, cfg_enc, D_agent, 'RL', False, False)
        self.value = cfg_Block(block_type, cfg_enc, D_agent, 'RL', False, False)
        if self.att_type == 'gat':
            self.att = cfg_Block(block_type, cfg_att, D_agent, 'RL', False, False)
        elif self.att_type == 'add':
            self.att1 = cfg_Block(block_type, [D_att, D_att], D_agent, 'RL', False, False)
            self.att2 = cfg_Block(block_type, [D_att, D_att], D_agent, 'RL', False, False)
            self.att3 = cfg_Block(block_type, [D_att, 1], D_agent, 'RL', False, False)
        elif self.att_type == 'mul':
            self.att = None
        else:
            print('NOT IMPLEMENTED : gat att block creation')

        # Attention aggregation module
        self.att_agg1 = nn.Linear(D_agent, D_agent)
        self.att_agg2 = nn.Linear(D_agent, D_agent)

        self.dec = cfg_Block(block_type, cfg_dec, D_agent, 'RL', False, False)
        self.mu_dec = cfg_Block(block_type, cfg_mu, D_agent, 'RL', False, False)
        self.sig_dec = cfg_Block(block_type, cfg_sig, D_agent, 'RL', False, False)
        self.corr_dec = cfg_Block(block_type, cfg_corr, D_agent, 'RL', False, False)
        self.coef_dec = cfg_Block(block_type, cfg_coef, D_agent, 'RL', False, False)

        if block_type == 'mlp':
            self.D_k = self.coef_dec.FC[-1].out_features
            self.D_s = int(self.mu_dec.FC[-1].out_features / self.D_k)
        elif block_type == 'res':
            self.D_k = self.coef_dec.FC2[-1].out_features
            self.D_s = int(self.mu_dec.FC2[-1].out_features / self.D_k)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if type(m.bias) != type(None):
                    m.bias.data.zero_()

    def forward(self, x):
        softmax = nn.Softmax(dim=-1)

        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        c = None
        p_list = [v]

        mask_const = 10000
        mask = torch.zeros(x.shape[0], x.shape[1], x.shape[1]).to(x.device)
        for i in range(x.shape[0]):
            m = (x[i, :, -1] != 1).nonzero().squeeze()
            if m.dim() != 1:
                print(x[i, :, -1])
                print(m)
                print('wrong')  #
            for j in m:
                mask[i][j, :] = mask_const
                mask[i][:, j] = mask_const

        mask = mask+torch.eye(x.shape[1], x.shape[1]).to(x.device)*mask_const

        if self.att_type == 'gat':
            for i in range(self.D_att_num):
                ky = k[:, :, self.D_att * i: self.D_att * (i + 1)]
                kz = [ky for _ in range(ky.shape[1])]
                qy = q[:, :, self.D_att * i: self.D_att * (i + 1)]
                qz = [qy for _ in range(qy.shape[1])]
                z = torch.cat((torch.stack(kz, dim=-2), torch.stack(qz, dim=-3)), -1)
                w = F.leaky_relu(self.att(z), 0.2).squeeze(-1)
                w = w + self.att_agg2(F.leaky_relu(self.att_agg1(w), 0.2))
                w = torch.sigmoid(torch.add(w, -mask))/self.D_agent
                p_list.append(torch.bmm(w, v[:, :, self.D_att * i: self.D_att * (i + 1)]))
                #p_list.append(torch.bmm(w,v))

        elif self.att_type == 'add':
            for i in range(self.D_att_num):
                ky = k[:, :, self.D_att * i: self.D_att * (i + 1)]
                kz = torch.stack([ky for _ in range(ky.shape[1])], dim = -2)
                qy = q[:, :, self.D_att * i: self.D_att * (i + 1)]
                qz = torch.stack([qy for _ in range(qy.shape[1])], dim = -3)
                f = self.att3(self.att1(kz)+self.att2(qz)).squeeze(-1)
                w = F.dropout(softmax(torch.add(f, -mask)), p=self.dropout, training=self.training)
                p_list.append(torch.bmm(w, v[:, :, self.D_att * i: self.D_att * (i + 1)]))

        elif self.att_type == 'mul':
            for i in range(self.D_att_num):
                ky = k[:, :, self.D_att * i: self.D_att * (i + 1)]
                kz = torch.stack([ky for _ in range(ky.shape[1])], dim = -2)
                qy = q[:, :, self.D_att * i: self.D_att * (i + 1)]
                qz = torch.stack([qy for _ in range(qy.shape[1])], dim = -3)
                w = F.dropout(softmax(torch.add(batchedDot(kz, qz)/np.sqrt(self.D_att), -mask)), p=self.dropout, training=self.training)
                p_list.append(torch.bmm(w, v[:, :, self.D_att * i: self.D_att * (i + 1)]))
        else:
            print('NOT IMPLEMENTED : GAT attention type')

        c = torch.cat(p_list, dim=-1)
        d = self.dec(c)

        mu = self.mu_dec(d)
        sig = self.sig_dec(d) / 10.0  # divided by 10.0 due to numeircal stability
        corr = self.corr_dec(d) /10.0
        coef = torch.softmax(self.coef_dec(d), dim=-1)  # sum(coef) = 1

        if self.D_s == 2:
            mu = mu.reshape(mu.shape[0], mu.shape[1], self.D_k, 2)
            sig = F.softplus(sig.reshape(sig.shape[0], sig.shape[1], self.D_k, 2))
            corr = F.softsign(corr.reshape(corr.shape[0], corr.shape[1], self.D_k, 1))

        elif self.D_s == 3:
            mu = mu.reshape(mu.shape[0], mu.shape[1], self.D_k, 3)
            sig = F.softplus(sig.reshape(sig.shape[0], sig.shape[1], self.D_k, 3))
            corr = F.softsign(corr.reshape(corr.shape[0], corr.shape[1], self.D_k, 3))

        elif self.D_s == 6:
            mu = mu.reshape(mu.shape[0], mu.shape[1], self.D_k, 6)
            sig = F.softplus(sig.reshape(sig.shape[0], sig.shape[1], self.D_k, 6))
            corr = F.softsign(corr.reshape(corr.shape[0], corr.shape[1], self.D_k, 15))
        else:
            print("NOT IMPLEMENTED : D_s reshaping")

        return mu, sig, corr, coef

class Module_GAT_AOUP(nn.Module):
    def __init__(self, cfg_init, cfg_lstm, cfg_enc, cfg_att, cfg_dec, cfg_mu, cfg_sig, cfg_corr, cfg_coef, D_att, D_att_num, D_agent, block_type, att_type, dropout):
        super(Module_GAT_AOUP, self).__init__()

        self.D_att = D_att
        self.D_att_num = D_att_num
        self.D_agent = D_agent
        self.block_type = block_type
        self.att_type = att_type
        self.dropout = dropout

        self.init_hidden = cfg_Block(block_type, cfg_init, D_agent, 'RL', False, False)
        self.init_cell = cfg_Block(block_type, cfg_init, D_agent, 'RL', False, False)
        self.lstm = nn.LSTMCell(cfg_lstm[0], cfg_lstm[1])

        self.key = cfg_Block(block_type, cfg_enc, D_agent, 'RL', False, False)
        self.query = cfg_Block(block_type, cfg_enc, D_agent, 'RL', False, False)
        self.value = cfg_Block(block_type, cfg_enc, D_agent, 'RL', False, False)
        if self.att_type == 'gat':
            self.att = cfg_Block(block_type, cfg_att, D_agent, 'RL', False, False)
        elif self.att_type == 'add':
            self.att1 = cfg_Block(block_type, [D_att, D_att], D_agent, 'RL', False, False)
            self.att2 = cfg_Block(block_type, [D_att, D_att], D_agent, 'RL', False, False)
            self.att3 = cfg_Block(block_type, [D_att, 1], D_agent, 'RL', False, False)
        elif self.att_type == 'mul':
            self.att = None
        else:
            print('NOT IMPLEMENTED : gat att block creation')

        self.dec1 = cfg_Block(block_type, cfg_dec, D_agent, 'RL', False, False)
        self.mu_dec1 = cfg_Block(block_type, cfg_mu, D_agent, 'RL', False, False)
        self.sig_dec1 = cfg_Block(block_type, cfg_sig, D_agent, 'RL', False, False)
        self.corr_dec1 = cfg_Block(block_type, cfg_corr, D_agent, 'RL', False, False)
        self.coef_dec1 = cfg_Block(block_type, cfg_coef, D_agent, 'RL', False, False)

        self.dec2 = cfg_Block(block_type, cfg_dec, D_agent, 'RL', False, False)
        self.mu_dec2 = cfg_Block(block_type, cfg_mu, D_agent, 'RL', False, False)
        self.sig_dec2 = cfg_Block(block_type, cfg_sig, D_agent, 'RL', False, False)
        self.corr_dec2 = cfg_Block(block_type, cfg_corr, D_agent, 'RL', False, False)
        self.coef_dec2 = cfg_Block(block_type, cfg_coef, D_agent, 'RL', False, False)

        if block_type == 'mlp':
            self.D_k = self.coef_dec1.FC[-1].out_features
            self.D_s = int(self.mu_dec1.FC[-1].out_features / self.D_k)
        elif block_type == 'res':
            self.D_k = self.coef_dec1.FC2[-1].out_features
            self.D_s = int(self.mu_dec1.FC2[-1].out_features / self.D_k)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if type(m.bias) != type(None):
                    m.bias.data.zero_()

    def initialize(self, x):
        batch_num = x.shape[0]
        agent_num = x.shape[1]
        x = x.reshape(x.shape[0] * x.shape[1], -1)
        h = self.init_hidden(x)
        c = self.init_cell(x)
        return (h.reshape(batch_num, agent_num, -1), c.reshape(batch_num, agent_num, -1))

    def forward(self, x, hidden, cell, R_const, test = False):
        #print(x.shape, hidden[0].shape, hidden[1].shape)
        softmax = nn.Softmax(dim=-1)

        batch_num = x.shape[0]
        agent_num = x.shape[1]
        x = x.reshape(batch_num * agent_num, -1)
        hidden = hidden.view(batch_num * agent_num, -1)
        cell = cell.view(batch_num * agent_num, -1)

        hidden, cell = self.lstm(x, (hidden, cell))
        if test:
            b = hidden.view(batch_num, agent_num, -1)
            x = x.view(batch_num, agent_num, -1)

            k = self.key(b)
            q = self.query(b)
            v = self.value(b)
            c = None
            p_list = [v]
            
            mask_const = 10000
            mask = torch.zeros(x.shape[0], x.shape[1], x.shape[1]).to(x.device)
            mask = mask+torch.eye(x.shape[1], x.shape[1]).to(x.device)*mask_const

            if self.att_type == 'gat':
                for i in range(self.D_att_num):
                    ky = k[:, :, self.D_att * i: self.D_att * (i + 1)]
                    kz = [ky for _ in range(ky.shape[1])]
                    qy = q[:, :, self.D_att * i: self.D_att * (i + 1)]
                    qz = [qy for _ in range(qy.shape[1])]
                    z = torch.cat((torch.stack(kz, dim=-2), torch.stack(qz, dim=-3), R_const), -1)
                    w = F.dropout(softmax(torch.add(F.leaky_relu(self.att(z), 0.2).squeeze(-1), -mask)), p = self.dropout, training = self.training)
                    p_list.append(torch.bmm(w, v[:, :, self.D_att * i: self.D_att * (i + 1)]))
            elif self.att_type == 'kqv':
                for i in range(self.D_att_num):
                    ky = k[:, :, self.D_att * i: self.D_att * (i + 1)]
                    kz = torch.stack([ky for _ in range(ky.shape[1])], dim = -2)
                    qy = q[:, :, self.D_att * i: self.D_att * (i + 1)]
                    qz = torch.stack([qy for _ in range(qy.shape[1])], dim = -3)
                    w = F.dropout(softmax(torch.add(batchedDot(kz, qz)/np.sqrt(self.D_att), -mask)), p=self.dropout, training=self.training)
                    p_list.append(torch.bmm(w, v[:, :, self.D_att * i: self.D_att * (i + 1)]))
            else:
                print('NOT IMPLEMENTED : GAT attention type')
            
            c = torch.cat(p_list, dim=-1)
            d1 = self.dec1(c)
            d2 = self.dec2(c)

            mu1 = self.mu_dec1(d1)
            sig1 = self.sig_dec1(d1)   
            corr1 = self.corr_dec1(d1) 
            coef1 = torch.softmax(self.coef_dec1(d1), dim=-1)  # sum(coef) = 1

            mu1 = mu1.reshape(mu1.shape[0], mu1.shape[1], self.D_k, 2)
            sig1 = F.softplus(sig1.reshape(sig1.shape[0], sig1.shape[1], self.D_k, 2))
            corr1 = F.softsign(corr1.reshape(corr1.shape[0], corr1.shape[1], self.D_k, 1))

            mu2 = self.mu_dec2(d2)
            sig2 = self.sig_dec2(d2)   
            corr2 = self.corr_dec2(d2) 
            coef2 = torch.softmax(self.coef_dec2(d2), dim=-1)  # sum(coef) = 1
            
            mu2 = mu2.reshape(mu2.shape[0], mu2.shape[1], self.D_k, 2)
            sig2 = F.softplus(sig2.reshape(sig2.shape[0], sig2.shape[1], self.D_k, 2))
            corr2 = F.softsign(corr2.reshape(corr2.shape[0], corr2.shape[1], self.D_k, 1))

            return (mu1, sig1, corr1, coef1), (mu2, sig2, corr2, coef2), hidden.view(batch_num, agent_num, -1), cell.view(batch_num, agent_num, -1)
        else:
            return hidden.view(batch_num, agent_num, -1), cell.view(batch_num, agent_num, -1)

class Module_GAT_AOUP_ns(nn.Module):

    def __init__(self, cfg_init, cfg_lstm, cfg_enc, cfg_att, cfg_dec, cfg_mu, cfg_sig, cfg_corr, cfg_coef, D_att, D_att_num, D_agent, block_type, att_type, dropout):
        super(Module_GAT_AOUP_ns, self).__init__()

        self.D_att = D_att
        self.D_att_num = D_att_num
        self.D_agent = D_agent
        self.block_type = block_type
        self.att_type = att_type
        self.dropout = dropout

        self.init_hidden = cfg_Block(block_type, cfg_init, D_agent, 'RL', False, False)
        self.init_cell = cfg_Block(block_type, cfg_init, D_agent, 'RL', False, False)
        self.lstm = nn.LSTMCell(cfg_lstm[0], cfg_lstm[1])

        self.key = cfg_Block(block_type, cfg_enc, D_agent, 'RL', False, False)
        self.query = cfg_Block(block_type, cfg_enc, D_agent, 'RL', False, False)
        self.value = cfg_Block(block_type, cfg_enc, D_agent, 'RL', False, False)
        if self.att_type == 'gat':
            self.att = cfg_Block(block_type, cfg_att, D_agent, 'RL', False, False)
        elif self.att_type == 'add':
            self.att1 = cfg_Block(block_type, [D_att, D_att], D_agent, 'RL', False, False)
            self.att2 = cfg_Block(block_type, [D_att, D_att], D_agent, 'RL', False, False)
            self.att3 = cfg_Block(block_type, [D_att, 1], D_agent, 'RL', False, False)
        elif self.att_type == 'mul':
            self.att = None
        else:
            print('NOT IMPLEMENTED : gat att block creation')

        self.dec1 = cfg_Block(block_type, cfg_dec, D_agent, 'RL', False, False)
        self.mu_dec1 = cfg_Block(block_type, cfg_mu, D_agent, 'RL', False, False)
        self.sig_dec1 = cfg_Block(block_type, cfg_sig, D_agent, 'RL', False, False)
        self.corr_dec1 = cfg_Block(block_type, cfg_corr, D_agent, 'RL', False, False)
        self.coef_dec1 = cfg_Block(block_type, cfg_coef, D_agent, 'RL', False, False)

        self.dec2 = cfg_Block(block_type, cfg_dec, D_agent, 'RL', False, False)
        self.mu_dec2 = cfg_Block(block_type, cfg_mu, D_agent, 'RL', False, False)
        self.sig_dec2 = cfg_Block(block_type, cfg_sig, D_agent, 'RL', False, False)
        self.corr_dec2 = cfg_Block(block_type, cfg_corr, D_agent, 'RL', False, False)
        self.coef_dec2 = cfg_Block(block_type, cfg_coef, D_agent, 'RL', False, False)

        if block_type == 'mlp':
            self.D_k = self.coef_dec1.FC[-1].out_features
            self.D_s = int(self.mu_dec1.FC[-1].out_features / self.D_k)
        elif block_type == 'res':
            self.D_k = self.coef_dec1.FC2[-1].out_features
            self.D_s = int(self.mu_dec1.FC2[-1].out_features / self.D_k)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if type(m.bias) != type(None):
                    m.bias.data.zero_()

    def initialize(self, x):
        batch_num = x.shape[0]
        agent_num = x.shape[1]
        x = x.reshape(x.shape[0] * x.shape[1], -1)
        h = self.init_hidden(x)
        c = self.init_cell(x)
        return (h.reshape(batch_num, agent_num, -1), c.reshape(batch_num, agent_num, -1))

    def forward(self, x, hidden, cell, R_const, test = False):
        #print(x.shape, hidden[0].shape, hidden[1].shape)
        softmax = nn.Softmax(dim=-1)

        batch_num = x.shape[0]
        agent_num = x.shape[1]
        x = x.reshape(batch_num * agent_num, -1)
        hidden = hidden.view(batch_num * agent_num, -1)
        cell = cell.view(batch_num * agent_num, -1)

        hidden, cell = self.lstm(x, (hidden, cell))
        if test:
            b = hidden.view(batch_num, agent_num, -1)
            x = x.view(batch_num, agent_num, -1)

            k = self.key(b)
            q = self.query(b)
            v = self.value(b)
            c = None
            p_list = [v]
            
            mask_const = 10000
            mask = torch.zeros(x.shape[0], x.shape[1], x.shape[1]).to(x.device)
            mask = mask+torch.eye(x.shape[1], x.shape[1]).to(x.device)*mask_const

            if self.att_type == 'gat':
                for i in range(self.D_att_num):
                    ky = k[:, :, self.D_att * i: self.D_att * (i + 1)]
                    kz = [ky for _ in range(ky.shape[1])]
                    qy = q[:, :, self.D_att * i: self.D_att * (i + 1)]
                    qz = [qy for _ in range(qy.shape[1])]
                    z = torch.cat((torch.stack(kz, dim=-2), torch.stack(qz, dim=-3), R_const), -1)
                    w = torch.sigmoid(torch.add(F.leaky_relu(self.att(z), 0.2).squeeze(-1), -mask))/self.D_agent
                    p_list.append(torch.bmm(w, v[:, :, self.D_att * i: self.D_att * (i + 1)]))
            elif self.att_type == 'kqv':
                for i in range(self.D_att_num):
                    ky = k[:, :, self.D_att * i: self.D_att * (i + 1)]
                    kz = torch.stack([ky for _ in range(ky.shape[1])], dim = -2)
                    qy = q[:, :, self.D_att * i: self.D_att * (i + 1)]
                    qz = torch.stack([qy for _ in range(qy.shape[1])], dim = -3)
                    w = F.dropout(softmax(torch.add(batchedDot(kz, qz)/np.sqrt(self.D_att), -mask)), p=self.dropout, training=self.training)
                    p_list.append(torch.bmm(w, v[:, :, self.D_att * i: self.D_att * (i + 1)]))
            else:
                print('NOT IMPLEMENTED : GAT attention type')
            
            c = torch.cat(p_list, dim=-1)
            d1 = self.dec1(c)
            d2 = self.dec2(c)

            mu1 = self.mu_dec1(d1)
            sig1 = self.sig_dec1(d1)   
            corr1 = self.corr_dec1(d1) 
            coef1 = torch.softmax(self.coef_dec1(d1), dim=-1)  # sum(coef) = 1

            mu1 = mu1.reshape(mu1.shape[0], mu1.shape[1], self.D_k, 2)
            sig1 = F.softplus(sig1.reshape(sig1.shape[0], sig1.shape[1], self.D_k, 2))
            corr1 = F.softsign(corr1.reshape(corr1.shape[0], corr1.shape[1], self.D_k, 1))

            mu2 = self.mu_dec2(d2)
            sig2 = self.sig_dec2(d2)   
            corr2 = self.corr_dec2(d2) 
            coef2 = torch.softmax(self.coef_dec2(d2), dim=-1)  # sum(coef) = 1
            
            mu2 = mu2.reshape(mu2.shape[0], mu2.shape[1], self.D_k, 2)
            sig2 = F.softplus(sig2.reshape(sig2.shape[0], sig2.shape[1], self.D_k, 2))
            corr2 = F.softsign(corr2.reshape(corr2.shape[0], corr2.shape[1], self.D_k, 1))

            return (mu1, sig1, corr1, coef1), (mu2, sig2, corr2, coef2), hidden.view(batch_num, agent_num, -1), cell.view(batch_num, agent_num, -1)
        else:
            return hidden.view(batch_num, agent_num, -1), cell.view(batch_num, agent_num, -1)

class Module_GAT_AOUP_new(nn.Module):
    def __init__(self, cfg_init, cfg_lstm, cfg_enc, cfg_att, cfg_dec, cfg_mu, cfg_sig, cfg_corr, cfg_coef, D_att, D_att_num, D_agent, block_type, att_type, dropout):
        super(Module_GAT_AOUP_new, self).__init__()

        self.D_att = D_att
        self.D_att_num = D_att_num
        self.D_agent = D_agent
        self.block_type = block_type
        self.att_type = att_type
        self.dropout = dropout

        self.init_hidden = cfg_Block(block_type, cfg_init, D_agent, 'RL', False, False)
        self.init_cell = cfg_Block(block_type, cfg_init, D_agent, 'RL', False, False)
        self.lstm = nn.LSTMCell(cfg_lstm[0], cfg_lstm[1])

        self.key = cfg_Block(block_type, cfg_enc, D_agent, 'RL', False, False)
        self.query = cfg_Block(block_type, cfg_enc, D_agent, 'RL', False, False)
        self.value = cfg_Block(block_type, cfg_enc, D_agent, 'RL', False, False)
        if self.att_type == 'gat':
            self.att = cfg_Block(block_type, cfg_att, D_agent, 'RL', False, False)
        elif self.att_type == 'add':
            self.att1 = cfg_Block(block_type, [D_att, D_att], D_agent, 'RL', False, False)
            self.att2 = cfg_Block(block_type, [D_att, D_att], D_agent, 'RL', False, False)
            self.att3 = cfg_Block(block_type, [D_att, 1], D_agent, 'RL', False, False)
        elif self.att_type == 'mul':
            self.att = None
        else:
            print('NOT IMPLEMENTED : gat att block creation')

        self.att_agg1 = nn.Linear(D_agent, D_agent)
        self.att_agg2 = nn.Linear(D_agent, D_agent)

        self.dec1 = cfg_Block(block_type, cfg_dec, D_agent, 'RL', False, False)
        self.mu_dec1 = cfg_Block(block_type, cfg_mu, D_agent, 'RL', False, False)
        self.sig_dec1 = cfg_Block(block_type, cfg_sig, D_agent, 'RL', False, False)
        self.corr_dec1 = cfg_Block(block_type, cfg_corr, D_agent, 'RL', False, False)
        self.coef_dec1 = cfg_Block(block_type, cfg_coef, D_agent, 'RL', False, False)

        self.dec2 = cfg_Block(block_type, cfg_dec, D_agent, 'RL', False, False)
        self.mu_dec2 = cfg_Block(block_type, cfg_mu, D_agent, 'RL', False, False)
        self.sig_dec2 = cfg_Block(block_type, cfg_sig, D_agent, 'RL', False, False)
        self.corr_dec2 = cfg_Block(block_type, cfg_corr, D_agent, 'RL', False, False)
        self.coef_dec2 = cfg_Block(block_type, cfg_coef, D_agent, 'RL', False, False)

        if block_type == 'mlp':
            self.D_k = self.coef_dec1.FC[-1].out_features
            self.D_s = int(self.mu_dec1.FC[-1].out_features / self.D_k)
        elif block_type == 'res':
            self.D_k = self.coef_dec1.FC2[-1].out_features
            self.D_s = int(self.mu_dec1.FC2[-1].out_features / self.D_k)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if type(m.bias) != type(None):
                    m.bias.data.zero_()

    def initialize(self, x):
        batch_num = x.shape[0]
        agent_num = x.shape[1]
        x = x.reshape(x.shape[0] * x.shape[1], -1)
        h = self.init_hidden(x)
        c = self.init_cell(x)
        return (h.reshape(batch_num, agent_num, -1), c.reshape(batch_num, agent_num, -1))

    def forward(self, x, hidden, cell, R_const, test = False):
        #print(x.shape, hidden[0].shape, hidden[1].shape)
        softmax = nn.Softmax(dim=-1)

        batch_num = x.shape[0]
        agent_num = x.shape[1]
        x = x.reshape(batch_num * agent_num, -1)
        hidden = hidden.view(batch_num * agent_num, -1)
        cell = cell.view(batch_num * agent_num, -1)

        hidden, cell = self.lstm(x, (hidden, cell))
        if test:
            b = hidden.view(batch_num, agent_num, -1)
            x = x.view(batch_num, agent_num, -1)

            k = self.key(b)
            q = self.query(b)
            v = self.value(b)
            c = None
            p_list = [v]
            
            mask_const = 10000
            mask = torch.zeros(x.shape[0], x.shape[1], x.shape[1]).to(x.device)
            mask = mask+torch.eye(x.shape[1], x.shape[1]).to(x.device)*mask_const

            if self.att_type == 'gat':
                for i in range(self.D_att_num):
                    ky = k[:, :, self.D_att * i: self.D_att * (i + 1)]
                    kz = [ky for _ in range(ky.shape[1])]
                    qy = q[:, :, self.D_att * i: self.D_att * (i + 1)]
                    qz = [qy for _ in range(qy.shape[1])]
                    z = torch.cat((torch.stack(kz, dim=-2), torch.stack(qz, dim=-3), R_const), -1)
                    w = F.leaky_relu(self.att(z), 0.2).squeeze(-1)
                    w = w + self.att_agg2(F.leaky_relu(self.att_agg1(w), 0.2))
                    w = torch.sigmoid(torch.add(w, -mask))/self.D_agent
                    p_list.append(torch.bmm(w, v[:, :, self.D_att * i: self.D_att * (i + 1)]))
            elif self.att_type == 'kqv':
                for i in range(self.D_att_num):
                    ky = k[:, :, self.D_att * i: self.D_att * (i + 1)]
                    kz = torch.stack([ky for _ in range(ky.shape[1])], dim = -2)
                    qy = q[:, :, self.D_att * i: self.D_att * (i + 1)]
                    qz = torch.stack([qy for _ in range(qy.shape[1])], dim = -3)
                    w = F.dropout(softmax(torch.add(batchedDot(kz, qz)/np.sqrt(self.D_att), -mask)), p=self.dropout, training=self.training)
                    p_list.append(torch.bmm(w, v[:, :, self.D_att * i: self.D_att * (i + 1)]))
            else:
                print('NOT IMPLEMENTED : GAT attention type')
            
            c = torch.cat(p_list, dim=-1)
            d1 = self.dec1(c)
            d2 = self.dec2(c)

            mu1 = self.mu_dec1(d1)
            sig1 = self.sig_dec1(d1)   
            corr1 = self.corr_dec1(d1) 
            coef1 = torch.softmax(self.coef_dec1(d1), dim=-1)  # sum(coef) = 1

            mu1 = mu1.reshape(mu1.shape[0], mu1.shape[1], self.D_k, 2)
            sig1 = F.softplus(sig1.reshape(sig1.shape[0], sig1.shape[1], self.D_k, 2))
            corr1 = F.softsign(corr1.reshape(corr1.shape[0], corr1.shape[1], self.D_k, 1))

            mu2 = self.mu_dec2(d2)
            sig2 = self.sig_dec2(d2)   
            corr2 = self.corr_dec2(d2) 
            coef2 = torch.softmax(self.coef_dec2(d2), dim=-1)  # sum(coef) = 1
            
            mu2 = mu2.reshape(mu2.shape[0], mu2.shape[1], self.D_k, 2)
            sig2 = F.softplus(sig2.reshape(sig2.shape[0], sig2.shape[1], self.D_k, 2))
            corr2 = F.softsign(corr2.reshape(corr2.shape[0], corr2.shape[1], self.D_k, 1))

            return (mu1, sig1, corr1, coef1), (mu2, sig2, corr2, coef2), hidden.view(batch_num, agent_num, -1), cell.view(batch_num, agent_num, -1)
        else:
            return hidden.view(batch_num, agent_num, -1), cell.view(batch_num, agent_num, -1)

class Module_GAT_AOUP_old(nn.Module):
    def __init__(self, cfg_init, cfg_lstm, cfg_enc, cfg_att, cfg_dec, cfg_mu, cfg_sig, cfg_corr, cfg_coef, D_att, D_att_num, D_agent, block_type, att_type, dropout):
        super(Module_GAT_AOUP, self).__init__()

        self.D_att = D_att
        self.D_att_num = D_att_num
        self.D_agent = D_agent
        self.block_type = block_type
        self.att_type = att_type
        self.dropout = dropout

        self.init_hidden = cfg_Block(block_type, cfg_init, D_agent, 'RL', False, False)
        self.init_cell = cfg_Block(block_type, cfg_init, D_agent, 'RL', False, False)
        self.lstm = nn.LSTMCell(cfg_lstm[0], cfg_lstm[1])

        self.key = cfg_Block(block_type, cfg_enc, D_agent, 'RL', False, False)
        self.query = cfg_Block(block_type, cfg_enc, D_agent, 'RL', False, False)
        self.value = cfg_Block(block_type, cfg_enc, D_agent, 'RL', False, False)
        if self.att_type == 'gat':
            self.att = cfg_Block(block_type, cfg_att, D_agent, 'RL', False, False)
        elif self.att_type == 'add':
            self.att1 = cfg_Block(block_type, [D_att, D_att], D_agent, 'RL', False, False)
            self.att2 = cfg_Block(block_type, [D_att, D_att], D_agent, 'RL', False, False)
            self.att3 = cfg_Block(block_type, [D_att, 1], D_agent, 'RL', False, False)
        elif self.att_type == 'mul':
            self.att = None
        else:
            print('NOT IMPLEMENTED : gat att block creation')

        self.dec = cfg_Block(block_type, cfg_dec, D_agent, 'RL', False, False)
        self.mu_dec = cfg_Block(block_type, cfg_mu, D_agent, 'RL', False, False)
        self.sig_dec = cfg_Block(block_type, cfg_sig, D_agent, 'RL', False, False)
        self.corr_dec = cfg_Block(block_type, cfg_corr, D_agent, 'RL', False, False)
        self.coef_dec = cfg_Block(block_type, cfg_coef, D_agent, 'RL', False, False)

        if block_type == 'mlp':
            self.D_k = self.coef_dec.FC[-1].out_features
            self.D_s = int(self.mu_dec.FC[-1].out_features / self.D_k)
        elif block_type == 'res':
            self.D_k = self.coef_dec.FC2[-1].out_features
            self.D_s = int(self.mu_dec.FC2[-1].out_features / self.D_k)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if type(m.bias) != type(None):
                    m.bias.data.zero_()

    def initialize(self, x):
        x = x.reshape(x.shape[0] * x.shape[1], -1)
        h = self.init_hidden(x)
        c = self.init_cell(x)
        return (h, c)

    def forward(self, x, hidden, R_const, test = False):
        softmax = nn.Softmax(dim=-1)
        batch_num = x.shape[0]
        agent_num = x.shape[1]
        x = x.reshape(batch_num * agent_num, -1)

        hidden = self.lstm(x, hidden)
        if test:
            b = hidden[0].view(batch_num, agent_num, -1)
            x = x.view(batch_num, agent_num, -1)

            k = self.key(b)
            q = self.query(b)
            v = self.value(b)
            c = None
            p_list = [v]
            
            mask_const = 10000
            mask = torch.zeros(x.shape[0], x.shape[1], x.shape[1]).to(x.device)
            mask = mask+torch.eye(x.shape[1], x.shape[1]).to(x.device)*mask_const

            if self.att_type == 'gat':
                for i in range(self.D_att_num):
                    ky = k[:, :, self.D_att * i: self.D_att * (i + 1)]
                    kz = [ky for _ in range(ky.shape[1])]
                    qy = q[:, :, self.D_att * i: self.D_att * (i + 1)]
                    qz = [qy for _ in range(qy.shape[1])]
                    z = torch.cat((torch.stack(kz, dim=-2), torch.stack(qz, dim=-3), R_const), -1)
                    w = F.dropout(softmax(torch.add(F.leaky_relu(self.att(z), 0.2).squeeze(-1), -mask)), p = self.dropout, training = self.training)
                    p_list.append(torch.bmm(w, v[:, :, self.D_att * i: self.D_att * (i + 1)]))
            elif self.att_type == 'kqv':
                for i in range(self.D_att_num):
                    ky = k[:, :, self.D_att * i: self.D_att * (i + 1)]
                    kz = torch.stack([ky for _ in range(ky.shape[1])], dim = -2)
                    qy = q[:, :, self.D_att * i: self.D_att * (i + 1)]
                    qz = torch.stack([qy for _ in range(qy.shape[1])], dim = -3)
                    w = F.dropout(softmax(torch.add(batchedDot(kz, qz)/np.sqrt(self.D_att), -mask)), p=self.dropout, training=self.training)
                    p_list.append(torch.bmm(w, v[:, :, self.D_att * i: self.D_att * (i + 1)]))
            else:
                print('NOT IMPLEMENTED : GAT attention type')
            
            c = torch.cat(p_list, dim=-1)
            d = self.dec(c)

            mu = self.mu_dec(d)
            sig = self.sig_dec(d)   # divided by 100.0 due to numeircal stability
            corr = self.corr_dec(d) 
            coef = torch.softmax(self.coef_dec(d), dim=-1)  # sum(coef) = 1

            if self.D_s == 2:
                mu = mu.reshape(mu.shape[0], mu.shape[1], self.D_k, 2)
                sig = F.softplus(sig.reshape(sig.shape[0], sig.shape[1], self.D_k, 2))
                corr = F.softsign(corr.reshape(corr.shape[0], corr.shape[1], self.D_k, 1))

            elif self.D_s == 3:
                mu = mu.reshape(mu.shape[0], mu.shape[1], self.D_k, 3)
                sig = F.softplus(sig.reshape(sig.shape[0], sig.shape[1], self.D_k, 3))
                corr = F.softsign(corr.reshape(corr.shape[0], corr.shape[1], self.D_k, 3))

            elif self.D_s == 4:
                mu = mu.reshape(mu.shape[0], mu.shape[1], self.D_k, 4)
                sig = F.softplus(sig.reshape(sig.shape[0], sig.shape[1], self.D_k, 4))
                corr = F.softsign(corr.reshape(corr.shape[0], corr.shape[1], self.D_k, 6))

            elif self.D_s == 6:
                mu = mu.reshape(mu.shape[0], mu.shape[1], self.D_k, 6)
                sig = F.softplus(sig.reshape(sig.shape[0], sig.shape[1], self.D_k, 6))
                corr = F.softsign(corr.reshape(corr.shape[0], corr.shape[1], self.D_k, 15))
            else:
                print("NOT IMPLEMENTED : D_s reshaping")

            return mu, sig, corr, coef, hidden
        else:
            return hidden

class Module_GAT_LSTM(nn.Module):
    def __init__(self, cfg_init, cfg_lstm, cfg_enc, cfg_att, cfg_dec, cfg_mu, cfg_sig, cfg_corr, cfg_coef, D_att, D_att_num, D_agent, block_type, att_type, dropout, eval_type):
        super(Module_GAT_LSTM, self).__init__()

        self.D_att = D_att
        self.D_att_num = D_att_num
        self.D_agent = D_agent
        self.block_type = block_type
        self.att_type = att_type
        self.dropout = dropout
        self.eval_type = eval_type

        self.init_hidden = cfg_Block(block_type, cfg_init, D_agent, 'RL', False, False)
        self.init_cell = cfg_Block(block_type, cfg_init, D_agent, 'RL', False, False)
        self.lstm = nn.LSTMCell(cfg_lstm[0], cfg_lstm[1])

        self.key = cfg_Block(block_type, cfg_enc, D_agent, 'RL', False, False)
        self.query = cfg_Block(block_type, cfg_enc, D_agent, 'RL', False, False)
        self.value = cfg_Block(block_type, cfg_enc, D_agent, 'RL', False, False)
        if self.att_type == 'gat':
            self.att = cfg_Block(block_type, cfg_att, D_agent, 'TH', False, False)
        elif self.att_type == 'kqv':
            self.att = None
        else:
            print('NOT IMPLEMENTED : gat att block creation')

        self.dec1 = cfg_Block(block_type, cfg_dec, D_agent, 'RL', False, False)
        self.mu_dec1 = cfg_Block(block_type, cfg_mu, D_agent, 'RL', False, False)
        self.sig_dec1 = cfg_Block(block_type, cfg_sig, D_agent, 'RL', False, False)
        self.corr_dec1 = cfg_Block(block_type, cfg_corr, D_agent, 'RL', False, False)
        self.coef_dec1 = cfg_Block(block_type, cfg_coef, D_agent, 'RL', False, False)

        self.dec2 = cfg_Block(block_type, cfg_dec, D_agent, 'RL', False, False)
        self.mu_dec2 = cfg_Block(block_type, cfg_mu, D_agent, 'RL', False, False)
        self.sig_dec2 = cfg_Block(block_type, cfg_sig, D_agent, 'RL', False, False)
        self.corr_dec2 = cfg_Block(block_type, cfg_corr, D_agent, 'RL', False, False)
        self.coef_dec2 = cfg_Block(block_type, cfg_coef, D_agent, 'RL', False, False)
        
        if block_type == 'mlp':
            self.D_k = self.coef_dec1.FC[-1].out_features
            self.D_s = int(self.mu_dec1.FC[-1].out_features / self.D_k)
        elif block_type == 'res':
            self.D_k = self.coef_dec1.FC2[-1].out_features
            self.D_s = int(self.mu_dec1.FC2[-1].out_features / self.D_k)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if type(m.bias) != type(None):
                    m.bias.data.zero_()

    def initialize(self, x, new_hidden, hidden = None):
        x = x.view(x.shape[0] * x.shape[1], -1)
        init_mask = (new_hidden[:, :, 0] > 0).flatten().unsqueeze(-1).float()
        h = self.init_hidden(x)
        c = self.init_cell(x)
        if hidden is None:
            hidden = (torch.zeros_like(h), torch.zeros_like(c))
        return (hidden[0]*(1-init_mask)) + h*init_mask, (hidden[1]*(1-init_mask))+c*init_mask

    def forward(self, x, hidden):
        softmax = nn.Softmax(dim=-1)

        batch_num = x.shape[0]
        agent_num = x.shape[1]
        x = x.view(batch_num * agent_num, -1)

        hidden = self.lstm(x, hidden)
        b = hidden[0].view(batch_num, agent_num, -1)
        x = x.view(batch_num, agent_num, -1)

        k = self.key(b)
        q = self.query(b)
        v = self.value(b)
        c = None
        p_list = [v]
        
        mask_const = 10000
        mask = torch.zeros(x.shape[0], x.shape[1], x.shape[1]).to(x.device)
        mask = mask+torch.eye(x.shape[1], x.shape[1]).to(x.device)*mask_const

        for i in range(x.shape[0]):
            m = (x[i, :, -1] != 1).nonzero().squeeze()
            if m.dim() != 1:
                print(x[i, :, -1])
                print(m)
                print('wrong')  #
            for j in m:
                mask[i][j, :] = mask_const
                mask[i][:, j] = mask_const

        if self.att_type == 'gat':
            for i in range(self.D_att_num):
                ky = k[:, :, self.D_att * i: self.D_att * (i + 1)]
                kz = [ky for _ in range(ky.shape[1])]
                qy = q[:, :, self.D_att * i: self.D_att * (i + 1)]
                qz = [qy for _ in range(qy.shape[1])]
                z = torch.cat((torch.stack(kz, dim=-2), torch.stack(qz, dim=-3)), -1)
                w = F.dropout(softmax(torch.add(F.leaky_relu(self.att(z), 0.2).squeeze(-1), -mask)), p = self.dropout, training = self.training)
                p_list.append(torch.bmm(w, v[:, :, self.D_att * i: self.D_att * (i + 1)]))
        elif self.att_type == 'kqv':
            for i in range(self.D_att_num):
                ky = k[:, :, self.D_att * i: self.D_att * (i + 1)]
                kz = torch.stack([ky for _ in range(ky.shape[1])], dim = -2)
                qy = q[:, :, self.D_att * i: self.D_att * (i + 1)]
                qz = torch.stack([qy for _ in range(qy.shape[1])], dim = -3)
                w = F.dropout(softmax(torch.add(batchedDot(kz, qz)/np.sqrt(self.D_att), -mask)), p=self.dropout, training=self.training)
                p_list.append(torch.bmm(w, v[:, :, self.D_att * i: self.D_att * (i + 1)]))
        else:
            print('NOT IMPLEMENTED : GAT attention type')
          
        c = torch.cat(p_list, dim=-1)
        d1 = self.dec1(c)
        d2 = self.dec2(c)

        mu1 = self.mu_dec1(d1)
        sig1 = self.sig_dec1(d1)   
        corr1 = self.corr_dec1(d1) 
        coef1 = torch.softmax(self.coef_dec1(d1), dim=-1)  # sum(coef) = 1

        mu1 = mu1.reshape(mu1.shape[0], mu1.shape[1], self.D_k, 3)
        sig1 = F.softplus(sig1.reshape(sig1.shape[0], sig1.shape[1], self.D_k, 3))
        corr1 = F.softsign(corr1.reshape(corr1.shape[0], corr1.shape[1], self.D_k, 3))

        mu2 = self.mu_dec2(d2)
        sig2 = self.sig_dec2(d2)   
        corr2 = self.corr_dec2(d2) 
        coef2 = torch.softmax(self.coef_dec2(d2), dim=-1)  # sum(coef) = 1
        
        mu2 = mu2.reshape(mu2.shape[0], mu2.shape[1], self.D_k, 3)
        sig2 = F.softplus(sig2.reshape(sig2.shape[0], sig2.shape[1], self.D_k, 3))
        corr2 = F.softsign(corr2.reshape(corr2.shape[0], corr2.shape[1], self.D_k, 3))

        return (mu1, sig1, corr1, coef1), (mu2, sig2, corr2, coef2), hidden

class Module_GAT_LSTM_single(nn.Module):
    def __init__(self, cfg_init, cfg_lstm, cfg_enc, cfg_att, cfg_dec, cfg_mu, cfg_sig, cfg_corr, cfg_coef, D_att, D_att_num, D_agent, block_type, att_type, dropout, eval_type):
        super(Module_GAT_LSTM_single, self).__init__()

        self.D_att = D_att
        self.D_att_num = D_att_num
        self.D_agent = D_agent
        self.block_type = block_type
        self.att_type = att_type
        self.dropout = dropout
        self.eval_type = eval_type

        self.init_hidden = cfg_Block(block_type, cfg_init, D_agent, 'RL', False, False)
        self.init_cell = cfg_Block(block_type, cfg_init, D_agent, 'RL', False, False)
        self.lstm = nn.LSTMCell(cfg_lstm[0], cfg_lstm[1])

        self.key = cfg_Block(block_type, cfg_enc, D_agent, 'RL', False, False)
        self.query = cfg_Block(block_type, cfg_enc, D_agent, 'RL', False, False)
        self.value = cfg_Block(block_type, cfg_enc, D_agent, 'RL', False, False)
        if self.att_type == 'gat':
            self.att = cfg_Block(block_type, cfg_att, D_agent, 'TH', False, False)
        elif self.att_type == 'kqv':
            self.att = None
        else:
            print('NOT IMPLEMENTED : gat att block creation')

        self.dec1 = cfg_Block(block_type, cfg_dec, D_agent, 'RL', False, False)
        self.mu_dec1 = cfg_Block(block_type, cfg_mu, D_agent, 'RL', False, False)
        self.sig_dec1 = cfg_Block(block_type, cfg_sig, D_agent, 'RL', False, False)
        self.corr_dec1 = cfg_Block(block_type, cfg_corr, D_agent, 'RL', False, False)

        self.dec2 = cfg_Block(block_type, cfg_dec, D_agent, 'RL', False, False)
        self.mu_dec2 = cfg_Block(block_type, cfg_mu, D_agent, 'RL', False, False)
        self.sig_dec2 = cfg_Block(block_type, cfg_sig, D_agent, 'RL', False, False)
        self.corr_dec2 = cfg_Block(block_type, cfg_corr, D_agent, 'RL', False, False)
        
        if block_type == 'mlp':
            #self.D_k = self.coef_dec1.FC[-1].out_features
            self.D_k = 1
            self.D_s = int(self.mu_dec1.FC[-1].out_features / self.D_k)
        elif block_type == 'res':
            #self.D_k = self.coef_dec1.FC2[-1].out_features
            self.D_k = 1
            self.D_s = int(self.mu_dec1.FC2[-1].out_features / self.D_k)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if type(m.bias) != type(None):
                    m.bias.data.zero_()

    def initialize(self, x, new_hidden, hidden = None):
        x = x.view(x.shape[0] * x.shape[1], -1)
        init_mask = (new_hidden[:, :, 0] > 0).flatten().unsqueeze(-1).float()
        h = self.init_hidden(x)
        c = self.init_cell(x)
        if hidden is None:
            hidden = (torch.zeros_like(h), torch.zeros_like(c))
        return (hidden[0]*(1-init_mask)) + h*init_mask, (hidden[1]*(1-init_mask))+c*init_mask

    def forward(self, x, hidden):
        softmax = nn.Softmax(dim=-1)

        batch_num = x.shape[0]
        agent_num = x.shape[1]
        x = x.view(batch_num * agent_num, -1)

        hidden = self.lstm(x, hidden)
        b = hidden[0].view(batch_num, agent_num, -1)
        x = x.view(batch_num, agent_num, -1)

        k = self.key(b)
        q = self.query(b)
        v = self.value(b)
        c = None
        p_list = [v]
        
        mask_const = 10000
        mask = torch.zeros(x.shape[0], x.shape[1], x.shape[1]).to(x.device)
        mask = mask+torch.eye(x.shape[1], x.shape[1]).to(x.device)*mask_const

        for i in range(x.shape[0]):
            m = (x[i, :, -1] != 1).nonzero().squeeze()
            if m.dim() != 1:
                print(x[i, :, -1])
                print(m)
                print('wrong')  #
            for j in m:
                mask[i][j, :] = mask_const
                mask[i][:, j] = mask_const

        if self.att_type == 'gat':
            for i in range(self.D_att_num):
                ky = k[:, :, self.D_att * i: self.D_att * (i + 1)]
                kz = [ky for _ in range(ky.shape[1])]
                qy = q[:, :, self.D_att * i: self.D_att * (i + 1)]
                qz = [qy for _ in range(qy.shape[1])]
                z = torch.cat((torch.stack(kz, dim=-2), torch.stack(qz, dim=-3)), -1)
                w = F.dropout(softmax(torch.add(F.leaky_relu(self.att(z), 0.2).squeeze(-1), -mask)), p = self.dropout, training = self.training)
                p_list.append(torch.bmm(w, v[:, :, self.D_att * i: self.D_att * (i + 1)]))
        elif self.att_type == 'kqv':
            for i in range(self.D_att_num):
                ky = k[:, :, self.D_att * i: self.D_att * (i + 1)]
                kz = torch.stack([ky for _ in range(ky.shape[1])], dim = -2)
                qy = q[:, :, self.D_att * i: self.D_att * (i + 1)]
                qz = torch.stack([qy for _ in range(qy.shape[1])], dim = -3)
                w = F.dropout(softmax(torch.add(batchedDot(kz, qz)/np.sqrt(self.D_att), -mask)), p=self.dropout, training=self.training)
                p_list.append(torch.bmm(w, v[:, :, self.D_att * i: self.D_att * (i + 1)]))
        else:
            print('NOT IMPLEMENTED : GAT attention type')
          
        c = torch.cat(p_list, dim=-1)
        d1 = self.dec1(c)
        d2 = self.dec2(c)

        mu1 = self.mu_dec1(d1)
        sig1 = self.sig_dec1(d1)   
        corr1 = self.corr_dec1(d1) 

        mu1 = mu1.reshape(mu1.shape[0], mu1.shape[1], self.D_k, 3)
        sig1 = F.softplus(sig1.reshape(sig1.shape[0], sig1.shape[1], self.D_k, 3))
        corr1 = F.softsign(corr1.reshape(corr1.shape[0], corr1.shape[1], self.D_k, 3))

        mu2 = self.mu_dec2(d2)
        sig2 = self.sig_dec2(d2)   
        corr2 = self.corr_dec2(d2) 
        
        mu2 = mu2.reshape(mu2.shape[0], mu2.shape[1], self.D_k, 3)
        sig2 = F.softplus(sig2.reshape(sig2.shape[0], sig2.shape[1], self.D_k, 3))
        corr2 = F.softsign(corr2.reshape(corr2.shape[0], corr2.shape[1], self.D_k, 3))

        return (mu1, sig1, corr1), (mu2, sig2, corr2), hidden

class Module_GAT_LSTM_ns(nn.Module):
    def __init__(self, cfg_init, cfg_lstm, cfg_enc, cfg_att, cfg_dec, cfg_mu, cfg_sig, cfg_corr, cfg_coef, D_att, D_att_num, D_agent, block_type, att_type, dropout, eval_type):
        super(Module_GAT_LSTM_ns, self).__init__()

        self.D_att = D_att
        self.D_att_num = D_att_num
        self.D_agent = D_agent
        self.block_type = block_type
        self.att_type = att_type
        self.dropout = dropout
        self.eval_type = eval_type

        self.init_hidden = cfg_Block(block_type, cfg_init, D_agent, 'RL', False, False)
        self.init_cell = cfg_Block(block_type, cfg_init, D_agent, 'RL', False, False)
        self.lstm = nn.LSTMCell(cfg_lstm[0], cfg_lstm[1])

        self.key = cfg_Block(block_type, cfg_enc, D_agent, 'RL', False, False)
        self.query = cfg_Block(block_type, cfg_enc, D_agent, 'RL', False, False)
        self.value = cfg_Block(block_type, cfg_enc, D_agent, 'RL', False, False)
        if self.att_type == 'gat':
            self.att = cfg_Block(block_type, cfg_att, D_agent, 'TH', False, False)
        elif self.att_type == 'kqv':
            self.att = None
        else:
            print('NOT IMPLEMENTED : gat att block creation')

        self.dec1 = cfg_Block(block_type, cfg_dec, D_agent, 'RL', False, False)
        self.mu_dec1 = cfg_Block(block_type, cfg_mu, D_agent, 'RL', False, False)
        self.sig_dec1 = cfg_Block(block_type, cfg_sig, D_agent, 'RL', False, False)
        self.corr_dec1 = cfg_Block(block_type, cfg_corr, D_agent, 'RL', False, False)
        self.coef_dec1 = cfg_Block(block_type, cfg_coef, D_agent, 'RL', False, False)

        self.dec2 = cfg_Block(block_type, cfg_dec, D_agent, 'RL', False, False)
        self.mu_dec2 = cfg_Block(block_type, cfg_mu, D_agent, 'RL', False, False)
        self.sig_dec2 = cfg_Block(block_type, cfg_sig, D_agent, 'RL', False, False)
        self.corr_dec2 = cfg_Block(block_type, cfg_corr, D_agent, 'RL', False, False)
        self.coef_dec2 = cfg_Block(block_type, cfg_coef, D_agent, 'RL', False, False)
        
        if block_type == 'mlp':
            self.D_k = self.coef_dec1.FC[-1].out_features
            self.D_s = int(self.mu_dec1.FC[-1].out_features / self.D_k)
        elif block_type == 'res':
            self.D_k = self.coef_dec1.FC2[-1].out_features
            self.D_s = int(self.mu_dec1.FC2[-1].out_features / self.D_k)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if type(m.bias) != type(None):
                    m.bias.data.zero_()

    def initialize(self, x, new_hidden, hidden = None):
        x = x.view(x.shape[0] * x.shape[1], -1)
        init_mask = (new_hidden[:, :, 0] > 0).flatten().unsqueeze(-1).float()
        h = self.init_hidden(x)
        c = self.init_cell(x)
        if hidden is None:
            hidden = (torch.zeros_like(h), torch.zeros_like(c))
        return (hidden[0]*(1-init_mask)) + h*init_mask, (hidden[1]*(1-init_mask))+c*init_mask

    def forward(self, x, hidden):
        softmax = nn.Softmax(dim=-1)

        batch_num = x.shape[0]
        agent_num = x.shape[1]
        x = x.view(batch_num * agent_num, -1)

        hidden = self.lstm(x, hidden)
        b = hidden[0].view(batch_num, agent_num, -1)
        x = x.view(batch_num, agent_num, -1)

        k = self.key(b)
        q = self.query(b)
        v = self.value(b)
        c = None
        p_list = [v]
        
        mask_const = 100
        mask = torch.zeros(x.shape[0], x.shape[1], x.shape[1]).to(x.device)
        mask = mask+torch.eye(x.shape[1], x.shape[1]).to(x.device)*mask_const

        for i in range(x.shape[0]):
            m = (x[i, :, -1] != 1).nonzero().squeeze()
            if m.dim() != 1:
                print(x[i, :, -1])
                print(m)
                print('wrong')  #
            for j in m:
                mask[i][j, :] = mask_const
                mask[i][:, j] = mask_const

        if self.att_type == 'gat':
            for i in range(self.D_att_num):
                ky = k[:, :, self.D_att * i: self.D_att * (i + 1)]
                kz = [ky for _ in range(ky.shape[1])]
                qy = q[:, :, self.D_att * i: self.D_att * (i + 1)]
                qz = [qy for _ in range(qy.shape[1])]
                z = torch.cat((torch.stack(kz, dim=-2), torch.stack(qz, dim=-3)), -1)
                w = torch.sigmoid(torch.add(F.leaky_relu(self.att(z), 0.2).squeeze(-1), -mask))/(self.D_agent*2) # Around average number of agent (~900)
                p_list.append(torch.bmm(w, v[:, :, self.D_att * i: self.D_att * (i + 1)]))
        elif self.att_type == 'kqv':
            for i in range(self.D_att_num):
                ky = k[:, :, self.D_att * i: self.D_att * (i + 1)]
                kz = torch.stack([ky for _ in range(ky.shape[1])], dim = -2)
                qy = q[:, :, self.D_att * i: self.D_att * (i + 1)]
                qz = torch.stack([qy for _ in range(qy.shape[1])], dim = -3)
                w = F.dropout(softmax(torch.add(batchedDot(kz, qz)/np.sqrt(self.D_att), -mask)), p=self.dropout, training=self.training)
                p_list.append(torch.bmm(w, v[:, :, self.D_att * i: self.D_att * (i + 1)]))
        else:
            print('NOT IMPLEMENTED : GAT attention type')
          
        c = torch.cat(p_list, dim=-1)
        d1 = self.dec1(c)
        d2 = self.dec2(c)

        mu1 = self.mu_dec1(d1)
        sig1 = self.sig_dec1(d1)   
        corr1 = self.corr_dec1(d1) 
        coef1 = torch.softmax(self.coef_dec1(d1), dim=-1)  # sum(coef) = 1

        mu1 = mu1.reshape(mu1.shape[0], mu1.shape[1], self.D_k, 3)
        sig1 = F.softplus(sig1.reshape(sig1.shape[0], sig1.shape[1], self.D_k, 3))
        corr1 = F.softsign(corr1.reshape(corr1.shape[0], corr1.shape[1], self.D_k, 3))

        mu2 = self.mu_dec2(d2)
        sig2 = self.sig_dec2(d2)   
        corr2 = self.corr_dec2(d2) 
        coef2 = torch.softmax(self.coef_dec2(d2), dim=-1)  # sum(coef) = 1
        
        mu2 = mu2.reshape(mu2.shape[0], mu2.shape[1], self.D_k, 3)
        sig2 = F.softplus(sig2.reshape(sig2.shape[0], sig2.shape[1], self.D_k, 3))
        corr2 = F.softsign(corr2.reshape(corr2.shape[0], corr2.shape[1], self.D_k, 3))

        return (mu1, sig1, corr1, coef1), (mu2, sig2, corr2, coef2), hidden

class Module_IN(nn.Module):
    def __init__(self, cfg_enc, cfg_com, cfg_att, cfg_dec, cfg_mu, cfg_sig, cfg_corr, cfg_coef, D_att, D_att_num,
                 D_agent, block_type, att_type, dropout):
        super(Module_KQV, self).__init__()

        self.D_att = D_att
        self.D_att_num = D_att_num
        self.D_agent = D_agent
        self.block_type = block_type
        self.att_type = att_type
        self.dropout = dropout

        self.enc = cfg_Block(block_type, cfg_enc, D_agent, 'RL', False, False)
        self.com = cfg_Block(block_type, cfg_com, D_agent, 'RL', False, False)
        self.att = cfg_Block(block_type, cfg_att, D_agent, 'RL', False, False)

        self.dec = cfg_Block(block_type, cfg_dec, D_agent, 'RL', False, False)
        self.mu_dec = cfg_Block(block_type, cfg_mu, D_agent, 'RL', False, False)
        self.sig_dec = cfg_Block(block_type, cfg_sig, D_agent, 'RL', False, False)
        self.corr_dec = cfg_Block(block_type, cfg_corr, D_agent, 'RL', False, False)
        self.coef_dec = cfg_Block(block_type, cfg_coef, D_agent, 'RL', False, False)

        if block_type == 'mlp':
            self.D_k = self.coef_dec.FC[-1].out_features
            self.D_s = int(self.mu_dec.FC[-1].out_features / self.D_k)
        elif block_type == 'res':
            self.D_k = self.coef_dec.FC2[-1].out_features
            self.D_s = int(self.mu_dec.FC2[-1].out_features / self.D_k)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if type(m.bias) != type(None):
                    m.bias.data.zero_()

    def forward(self, x):
        softmax = nn.Softmax(dim=-1)

        e_s = self.enc(x)
        e_c = self.com(x)
        a = self.att(x)

        mask1 = torch.zeros(x.shape[0], x.shape[1], x.shape[1]).to(x.device)
        for i in range(x.shape[0]):
            m = (x[i, :, -1] != 1).nonzero().squeeze()
            if m.dim() != 1:
                print(x[i, :, -1])
                print(m)
                print('wrong')  #
            for j in m:
                mask1[i][j, :] = 1
                mask1[i][:, j] = 1

        mask1 = (mask1 + torch.eye(x.shape[1], x.shape[1]).to(x.device)) * 10000
        c = None
        if self.att_type == 'single':
            w = F.dropout(softmax(-torch.add(pdist(a), mask1)), p=self.dropout, training=self.training)
            p = torch.bmm(torch.mul(w, mask2), e_c)
            c = torch.cat((p, e_s), dim=-1)

        elif self.att_type == 'multi':
            p_list = [e_s]
            for i in range(self.D_att_num):
                p_list.append(
                    torch.bmm(torch.mul(
                        F.dropout(softmax(-torch.add(pdist(a[:, :, self.D_att * i: self.D_att * (i + 1)]), mask1)),
                                  p=self.dropout, training=self.training), mask2),
                        e_c))
            c = torch.cat(p_list, dim=-1)

        elif self.att_type == 'distort':
            q1 = pdist(a).unsqueeze(-1)
            q2 = (self.distort(q1) + q1).squeeze()  # Skip-connection
            w = F.dropout(softmax(-torch.add(q2, mask1)), p=self.dropout, training=self.training)
            p = torch.bmm(torch.mul(w, mask2), e_c)
            c = torch.cat((p, e_s), dim=-1)
        else:
            print('NOT IMPLEMENTED : ATTENTION FORWARD')

        d = self.dec(c)

        mu = self.mu_dec(d)
        sig = self.sig_dec(d) / 10.  # divided by 10.0 due to numeircal stability
        corr = self.corr_dec(d) / 10.
        coef = torch.softmax(self.coef_dec(d), dim=-1)  # sum(coef) = 1

        if self.D_s == 2:
            mu = mu.reshape(mu.shape[0], mu.shape[1], self.D_k, 2)
            sig = F.softplus(sig.reshape(sig.shape[0], sig.shape[1], self.D_k, 2))
            corr = F.softsign(corr.reshape(corr.shape[0], corr.shape[1], self.D_k, 1))

        elif self.D_s == 3:
            mu = mu.reshape(mu.shape[0], mu.shape[1], self.D_k, 3)
            sig = F.softplus(sig.reshape(sig.shape[0], sig.shape[1], self.D_k, 3))
            corr = F.softsign(corr.reshape(corr.shape[0], corr.shape[1], self.D_k, 3))

        elif self.D_s == 6:
            mu = mu.reshape(mu.shape[0], mu.shape[1], self.D_k, 6)
            sig = F.softplus(sig.reshape(sig.shape[0], sig.shape[1], self.D_k, 6))
            corr = F.softsign(corr.reshape(corr.shape[0], corr.shape[1], self.D_k, 15))
        else:
            print("NOT IMPLEMENTED : D_s reshaping")

        return mu, sig, corr, coef


## System
'''
1. Cellular Automata
2. Vicsek_markovian
3. Viscek_sigmoid
4. Vicsek_linear
5. Active Ornstein-Uhlenbeck Particle
6. Chimney Swift Flock
'''

##########################
## 1. Cellular Automata ##
##########################

class CA():
    '''
    Cellular Automata dataset
    
    '''
    
    def __init__(self):
        self.name = 'CA'
        self.rule_name = 'Agent_'+str(self.name)
        self.state_num = 3 # x, y, c (cell state)

    def assign_pp(self, plugin_parameters):
        self.pp = plugin_parameters
        assert 'side_length' in plugin_parameters
        self.side_length = self.pp['side_length']
        self.input_length = self.side_length**2
        assert 'rule' in plugin_parameters
        self.rule = self.pp['rule']
        self.rule_name += '_a'+str(self.rule['alive'])+'_d'+str(self.rule['dead'])
        
    # Data version
    
    def neighbor_fn_data(self, data):
        size = np.prod(data.shape) # size = L^2
        side = int(np.sqrt(size)) # perfect square lattice, L = sqrt(size)
        M, N = side, side
        cells = list(range(size))
        idx, idy = np.unravel_index(cells, dims=(M, N))
    
        neigh_idx = np.vstack((idx-1, idx+1, idx, idx, idx-1, idx-1, idx+1, idx+1))
        neigh_idy = np.vstack((idy, idy, idy-1, idy+1, idy-1, idy+1, idy-1, idy+1)) # ←→↑↓↖↙↗↘

        neighbor = np.ravel_multi_index((neigh_idx, neigh_idy), mode = 'warp', dims=(M,N)).T
        return np.concatenate((neighbor, np.expand_dims(np.arange(size), 1)), axis = 1)

    def __next__(self):
        L = self.side_length
        while(True):
            system = np.random.randint(0, 2, L**2)
            system_now = deepcopy(system)
            answer = []
            neighbor = self.neighbor_fn_data(system)
            system_next = deepcopy(system_now)
            
            for i in range(L**2):
                n_list = system_now[neighbor][i]
                if system_now[i] == 1:
                    if np.sum(n_list[:-1]) not in self.rule['alive']:
                        system_next[i] = 0
                elif system_now[i] == 0 :
                    if np.sum(n_list[:-1]) in self.rule['dead']:
                        system_next[i] = 1
                else:
                    print('ERROR')
            
            cells = list(range(self.input_length))
            idx, idy = np.unravel_index(cells, dims=(self.side_length, self.side_length))
            system_now = np.stack((idx, idy, system_now), axis = 1)
            
            return np.array(system_now), np.array(system_next)
        
    def test_figure(self, fig, model, device):
        x = ["".join(seq) for seq in itertools.product("01", repeat=8)]
        t1, t2 = [], []
        for i in x:
            t1.append(i+'0')
            t2.append(i+'1')

        t = t1 + t2
        expected = []
        predicted = []

        for i in t1:
            if np.sum(list(map(int, list(i)))[:-1]) in system.rule['dead']:
                expected.append(1)
            else:
                expected.append(0)

        for i in t2:
            if np.sum(list(map(int, list(i)))[:-1]) in system.rule['alive']:
                expected.append(1)
            else:
                expected.append(0)

        for i in t:
            predicted.append(float(model(torch.Tensor(list(map(int, list(i)))).to(device), 1)))

        ax = fig.add_subplot(2,2,(3,4))
        ax.plot(expected, color = 'r', linewidth = 2, label = 'expected', alpha = 0.5)
        ax.scatter(list(range(512)), predicted, color = 'b', linewidth = 1, label = 'predicted')
        ax.set_xlabel('Case number', fontsize = 15)
        ax.set_ylabel('Result', fontsize = 15)
        ax.set_ylim(-1,2)
        return ax

#########################
## 2. Vicsek_markovian ##
#########################


class Vicsek_markovian():
    def __init__(self):
        self.name = 'VM_mk'
        self.rule_name = 'Agent_' + str(self.name)
        self.speed_boost = 1
        self.pp = None
        self.agent_num = -1
        self.neighbor_dist = -1
        self.neighbor_angle = -1
        self.noise_type = None
        self.noise_strength = -1

    def assign_pp(self, plugin_parameters):
        self.pp = plugin_parameters
        assert 'agent_num' in plugin_parameters
        self.agent_num = self.pp['agent_num']
        assert 'neighbor_dist' in plugin_parameters
        self.neighbor_dist = self.pp['neighbor_dist']
        assert 'neighbor_angle' in plugin_parameters
        self.neighbor_angle = self.pp['neighbor_angle']
        assert 'noise_type' in plugin_parameters
        self.noise_type = self.pp['noise_type']
        assert 'noise_strength' in plugin_parameters
        self.noise_strength = self.pp['noise_strength']
        self.rule_name += '_a' + str(self.agent_num) + '_na' + str(self.neighbor_angle) + '_nt' + str(
            self.noise_type) + '_ns' + str(self.noise_strength)

    # Data version

    def neighbor_fn_data(self, data_now):

        data_pos = data_now[:, :2]  # position
        distance = sp.spatial.distance_matrix(data_pos, data_pos)
        angle = np.array([[angle_between_vec(x, y) for y in data_now[:, :4]] for x in data_now[:, :4]])
        neighbor = [np.insert(np.argwhere(d).reshape(-1), 0, i)
                    for i, d in enumerate(
                np.where((distance < self.neighbor_dist) & (np.abs(angle) < self.neighbor_angle), distance, 0))]

        return np.array([data_now[neighbor[i]] for i in range(data_now.shape[0])])

    def agg_fn_data(self, data_now, final_vel):
        vx = final_vel[:, 0]
        vy = final_vel[:, 1]
        data_next = np.stack((data_now[:, 0] + vx, data_now[:, 1] + vy, vx, vy, data_now[:, 4]), axis=1)

        return data_next, data_next[:, [4, 0, 1]]

    ## DataGen generator

    def __next__(self):
        N = self.agent_num
        while (True):
            # theta = np.random.uniform(-np.pi, np.pi, (N, 1))
            # data_pos = np.random.uniform(-1, 1, (N, 2))+np.random.uniform(-3, 3, (1,2)) # Total length of 5, 2 velocity / 2 pos / 1 group
            # data_grp = np.ones((N,1))
            # data = np.concatenate((np.cos(theta)*1/self.speed_boost, np.sin(theta)*1/self.speed_boost, data_pos, data_grp), axis = 1) # position range : 4 unit, velocity magnitude : 0.25

            length = np.sqrt(np.random.uniform(0, 2, (N, 1)))
            angle = np.pi * np.random.uniform(0, 2, (N, 1))
            data_pos = length * np.concatenate((np.cos(angle), np.sin(angle)), axis=1)
            theta = np.pi * np.random.uniform(0, 2, (N, 1))
            data_grp = np.ones((N, 1))
            data = np.concatenate(
                (data_pos, np.cos(theta) / self.speed_boost, np.sin(theta) / self.speed_boost, data_grp), axis=1)

            data_now = deepcopy(data)
            data_input = self.neighbor_fn_data(data_now)

            # viscek model calculation
            vx = np.zeros(len(data_input))
            vy = np.zeros(len(data_input))
            for i in range(len(data_input)):
                # print(data_input[i].shape)
                vx_sum = np.sum(data_input[i][:, 2])
                vy_sum = np.sum(data_input[i][:, 3])
                v2 = self.speed_boost * ((vx_sum ** 2 + vy_sum ** 2) ** 0.5)
                vx[i], vy[i] = vx_sum / v2, vy_sum / v2
            predicted_list = np.stack((vx, vy), axis=1)

            angle_change = np.radians(np.clip(
                np.array([angle_between(data_now[i, 2:4], predicted_list[i]) for i in range(predicted_list.shape[0])]),
                -120, 120))
            if self.noise_type == "angle":
                angle_change += np.random.normal(0, self.noise_strength, angle_change.shape)
            rot_mat = np.array([np.array([[np.cos(x), -np.sin(x)], [np.sin(x), np.cos(x)]]) for x in angle_change])
            final_vel = np.matmul(np.expand_dims(data_now[:, 2:4], -2), rot_mat).squeeze()
            final_vel += np.random.normal(0, 0.05, final_vel.shape)

            # update
            _, answer = self.agg_fn_data(data_now, final_vel)

            return np.array(data), answer

    # Test figure

    def vicsek_simulation(self, data_now, index, iter_num):
        data_input = self.neighbor_fn_data(data_now)[index]

        # viscek model calculation
        vx_sum = np.sum(data_input[:, 2])
        vy_sum = np.sum(data_input[:, 3])
        v2 = self.speed_boost * ((vx_sum ** 2 + vy_sum ** 2) ** 0.5)
        vx, vy = vx_sum / v2, vy_sum / v2
        predicted_list = np.array([vx, vy])

        angle_change = np.radians(
            np.clip(angle_between(data_now[index][2:4], predicted_list), -self.neighbor_angle, self.neighbor_angle))
        rot_mat = np.array(
            [[np.cos(angle_change), -np.sin(angle_change)], [np.sin(angle_change), np.cos(angle_change)]])
        final_vel = np.matmul(np.expand_dims(data_now[index][2:4], -2), rot_mat).squeeze()
        answer_mean = np.array([data_now[index][0] + final_vel[0], data_now[index][1] + final_vel[1]])
        answer_list = np.zeros((iter_num, 2))
        if self.noise_type == 'angle':
            for i in range(iter_num):
                sample_angle = angle_change + np.random.normal(0, self.noise_strength, angle_change.shape)
                rot_mat = np.array(
                    [[np.cos(sample_angle), -np.sin(sample_angle)], [np.sin(sample_angle), np.cos(sample_angle)]])
                final_vel = np.matmul(np.expand_dims(data_now[index][2:4], -2), rot_mat).squeeze()
                answer_list[i][0] = data_now[index][0] + final_vel[0]
                answer_list[i][1] = data_now[index][1] + final_vel[1]
            answer_list += np.random.normal(0, 0.05, answer_list.shape)
        elif self.noise_type == 'pos':
            for i in range(iter_num):
                answer_list[i][0] = data_now[index][0] + final_vel[0]
                answer_list[i][1] = data_now[index][1] + final_vel[1]
            answer_list += np.random.normal(0, self.noise_strength, answer_list.shape)

        return answer_mean, answer_list

    def test_figure(self, fig, model, device, test_loader, r, index):

        model.eval()
        ax_list = []
        # r = np.random.randint(len(test_loader.dataset.test_data))
        # data = test_loader.dataset.test_data[r].to(device)
        # label = test_loader.dataset.test_labels[r].to(device)

        data = test_loader.dataset.test_data[r].to(device)
        label = test_loader.dataset.test_labels[r].to(device).unsqueeze(0)

        mu, sig, corr, coef = model(data.unsqueeze(0), label[:, :, 0])
        mu, sig, corr, coef = DCN(mu.squeeze(0)), DCN(sig.squeeze(0)), DCN(corr.squeeze(0)), DCN(coef.squeeze(0))
        model.train()

        ax = fig.add_subplot(2, 2, 3)

        # drawing sampled 2d histogram from real answer (thanks to analytic form of vicsek, impossible for real bird)
        answer_mean, answer_list = self.vicsek_simulation(DCN(data), index=index, iter_num=1000)
        # drawing contour of VAINS result
        delta = 0.01
        x = np.arange(-5.0, 5.0, delta)
        y = np.arange(-5.0, 5.0, delta)
        X, Y = np.meshgrid(x, y)
        Z = corr_bivariate(np.stack((X, Y), axis=-1), mu[index], sig[index], corr[index], coef[index])

        ax.hist2d(answer_list[:, 0], answer_list[:, 1], bins=[20, 20],
                  range=[[answer_mean[0] - 1, answer_mean[0] + 1], [answer_mean[1] - 1, answer_mean[1] + 1]],
                  cmap=cm.gray.reversed())
        cs = ax.contour(X, Y, Z, color='white')
        # cs2 = ax.contourf(X, Y, Z, cmap = cm.autumn)
        ax.clabel(cs, inline=1, fontsize=8)

        data = DCN(data)
        label = DCN(label)

        ax.scatter(data[:, 0], data[:, 1], alpha=0.2)
        ax.scatter(data[index, 0], data[index, 1], s=30, c='b', label='origin')
        ax.scatter(answer_mean[0], answer_mean[1], s=30, c='g', label='expected')
        ax.scatter(label[0, index, 1], label[0, index, 2], s=30, c='r', label='label')
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('X', fontsize=15)
        ax.set_ylabel('Y', fontsize=15)
        ax.set_xlim(answer_mean[0] - 2, answer_mean[0] + 2)
        ax.set_ylim(answer_mean[1] - 2, answer_mean[1] + 2)

        # ax.set_xlim(-5, 5)
        # ax.set_ylim(-5, 5)

        ax.legend()
        ax_list.append(ax)

        ax2 = fig.add_subplot(2, 2, 4)

        softmax = nn.Softmax(dim=-1)
        x = test_loader.dataset.test_data[r].to(device).unsqueeze(0)
        b = test_loader.dataset.test_labels[r].to(device).unsqueeze(0)[:, :, 0]
        e_s = model.enc(x)
        e_c = model.com(x)
        a = e_c[:, :, :D_att]

        mask1 = torch.ones(x.shape[0], x.shape[1], x.shape[1]).to(x.device)
        for i in range(x.shape[0]):
            m = (b[i] == 1).nonzero().squeeze()
            if m.dim() != 1:
                print(b)
                print(m)
                print('wrong')  #
            for j in m:
                mask1[i][j, :] = 0
                mask1[i][:, j] = 0

        mask1 = (mask1 + torch.eye(x.shape[1], x.shape[1]).to(x.device)) * 10000
        mask2 = 1 - torch.eye(x.shape[1], x.shape[1]).to(x.device)

        w = softmax(-torch.add(pdist(a), mask1))
        # w = pdist(a)

        x = DCN(x)
        w = DCN(w)
        w[0][index][index] = 0.
        ax2.quiver(x[0][index, 0], x[0][index, 1], x[0][index, 2], x[0][index, 3], lw=0.5)
        ax2.scatter(x[0][:, 0], x[0][:, 1], c='k', s=50, alpha=0.5)
        ax2.scatter(x[0][:, 0], x[0][:, 1], cmap=cm.autumn, c=w[0][index], s=w[0][index] * 1000, alpha=0.5)
        ax2.scatter(x[0][index, 0], x[0][index, 1], color='royalblue', s=100)

        circle = plt.Circle((x[0][index][0], x[0][index][1]), 0.5, color='r', lw=2, ls='--', fill=False)
        ax2.add_artist(circle)
        ax2.set_aspect('equal', adjustable='box')  # 기본 VAIN
        ax_list.append(ax2)

        return ax_list  # New version, velocity alignment


#######################
## 3. Vicsek_sigmoid ##
#######################

def norm_sigmoid(a, b, x):
    s= 1/(1+np.exp(b*(x-a)))
    return 1*(s-np.min(s))/(np.max(s)-np.min(s)) # normalize function to 0-1

      
class Vicsek_sigmoid(): 
    
    def __init__(self, width):
        self.name = 'VM_sg' + str(width)
        self.rule_name = 'Agent_'+str(self.name)
        self.speed_boost = 1
        self.count = 0
        self.width = width
        
    def assign_pp(self, plugin_parameters):
        self.pp = plugin_parameters
        assert 'agent_num' in plugin_parameters
        self.agent_num = self.pp['agent_num']
        assert 'neighbor_dist' in plugin_parameters
        self.neighbor_dist = self.pp['neighbor_dist']
        assert 'neighbor_angle' in plugin_parameters
        self.neighbor_angle = self.pp['neighbor_angle']
        assert 'noise_type' in plugin_parameters
        self.noise_type = self.pp['noise_type']
        assert 'noise_strength' in plugin_parameters
        self.noise_strength = self.pp['noise_strength']
        self.rule_name += '_a' + str(self.agent_num) + '_na' + str(self.neighbor_angle) + '_nt' + str(self.noise_type) + '_ns' + str(self.noise_strength)

    # Data version
    
    def neighbor_fn_data(self, data_now): 

        data_pos = data_now[:,:2] # position
        distance = sp.spatial.distance_matrix(data_pos, data_pos)
        norm_distance = norm_sigmoid(1, self.width, distance)
        angle = np.array([[angle_between_vec(x,y) for y in data_now[:,:4]] for x in data_now[:,:4]])
        
        return np.array([(data_now.T*norm_distance[i]).T for i in range(norm_distance.shape[0])])

    def agg_fn_data(self, data_now, final_vel):
        vx = final_vel[:,0]
        vy = final_vel[:,1]
        data_next = np.stack((data_now[:,0]+vx, data_now[:,1]+vy, vx, vy,  data_now[:,4]), axis = 1)

        return data_next, data_next[:, [4,0,1]]
    
    ## DataGen generator
    
    def __next__(self):
        N = self.agent_num
        while(True):
            #theta = np.random.uniform(-np.pi, np.pi, (N, 1))
            #data_pos = np.random.uniform(-1, 1, (N, 2))+np.random.uniform(-3, 3, (1,2)) # Total length of 5, 2 velocity / 2 pos / 1 group
            #data_grp = np.ones((N,1))
            #data = np.concatenate((np.cos(theta)*1/self.speed_boost, np.sin(theta)*1/self.speed_boost, data_pos, data_grp), axis = 1) # position range : 4 unit, velocity magnitude : 0.25 
            
            self.count += 1
            if self.count % 100 == 0:
                print(self.count)
                
            length = np.sqrt(np.random.uniform(0, 9, (N,1)))
            angle = np.pi * np.random.uniform(0, 2, (N,1))
            data_pos =  length * np.concatenate((np.cos(angle),np.sin(angle)), axis = 1)
            theta = np.pi * np.random.uniform(0, 2, (N,1))
            data_grp = np.ones((N,1))
            data = np.concatenate((data_pos,np.cos(theta)/self.speed_boost, np.sin(theta)/self.speed_boost, data_grp), axis = 1)
            
            data_now = deepcopy(data)
            data_input = self.neighbor_fn_data(data_now)

            # viscek model calculation
            vx = np.zeros(len(data_input))
            vy = np.zeros(len(data_input))
            for i in range(len(data_input)):
                #print(data_input[i].shape)
                vx_sum = np.sum(data_input[i][:,2])
                vy_sum = np.sum(data_input[i][:,3])
                v2 = self.speed_boost*((vx_sum**2+vy_sum**2)**0.5)
                vx[i], vy[i] = vx_sum/v2, vy_sum/v2
            predicted_list = np.stack((vx, vy), axis = 1)    
            
            angle_change = np.radians(np.clip(np.array([angle_between(data_now[i,2:4], predicted_list[i]) for i in range(predicted_list.shape[0])]), -120, 120))
            if self.noise_type == "angle":
                angle_change += np.random.normal(0, self.noise_strength, angle_change.shape)
            rot_mat = np.array([np.array([[np.cos(x), -np.sin(x)], [np.sin(x), np.cos(x)]]) for x in angle_change])
            final_vel = np.matmul(np.expand_dims(data_now[:, 2:4], -2), rot_mat).squeeze()
            if self.noise_type == "pos" :
                final_vel += np.random.normal(0, self.noise_strength, final_vel.shape)
            if self.noise_type == 'corr':
                cv_default = np.array([[np.square(self.noise_strength), 0],[0, np.square(self.noise_strength)/3]])
                for i in range(final_vel.shape[0]):
                    velocity_angle = np.radians(angle_between([1, 0], final_vel[i]))
                    rotation = np.array([[np.cos(velocity_angle), -np.sin(velocity_angle)], [np.sin(velocity_angle), np.cos(velocity_angle)]])
                    cv = rotation @ cv_default @ rotation.transpose(1, 0)
                    final_vel[i] += np.random.multivariate_normal([0,0], cv)
                
            _, answer = self.agg_fn_data(data_now, final_vel)

            return np.array(data), answer
        
    # Test figure    
    
    def vicsek_simulation(self, data_now, index, iter_num):
        data_input = self.neighbor_fn_data(data_now)[index]

        # viscek model calculation
        vx_sum = np.sum(data_input[:,2])
        vy_sum = np.sum(data_input[:,3])
        v2 = self.speed_boost*((vx_sum**2+vy_sum**2)**0.5)
        vx, vy = vx_sum/v2, vy_sum/v2
        predicted_list = np.array([vx, vy])

        angle_change = np.radians(np.clip(angle_between(data_now[index][2:4], predicted_list), -self.neighbor_angle, self.neighbor_angle))
        rot_mat = np.array([[np.cos(angle_change), -np.sin(angle_change)], [np.sin(angle_change), np.cos(angle_change)]])
        final_vel = np.matmul(np.expand_dims(data_now[index][2:4], -2), rot_mat).squeeze()
        answer_mean = np.array([data_now[index][0] + final_vel[0], data_now[index][1] + final_vel[1]])
        answer_list = np.zeros((iter_num, 2))
        if self.noise_type == 'angle':
            for i in range(iter_num):
                sample_angle = angle_change + np.random.normal(0, self.noise_strength, angle_change.shape)
                rot_mat = np.array([[np.cos(sample_angle), -np.sin(sample_angle)], [np.sin(sample_angle), np.cos(sample_angle)]])
                final_vel = np.matmul(np.expand_dims(data_now[index][2:4], -2), rot_mat).squeeze()
                answer_list[i][0] = data_now[index][0] + final_vel[0]
                answer_list[i][1] = data_now[index][1] + final_vel[1]
        elif self.noise_type == 'pos':
            for i in range(iter_num):
                answer_list[i][0] = data_now[index][0] + final_vel[0]
                answer_list[i][1] = data_now[index][1] + final_vel[1]
            answer_list += np.random.normal(0, self.noise_strength, answer_list.shape)
        elif self.noise_type == 'corr':
            cv_default = np.array([[np.square(self.noise_strength), 0],[0, np.square(self.noise_strength)/3]])
            for i in range(iter_num):
                velocity_angle = np.radians(angle_between([1, 0], final_vel))
                rotation = np.array([[np.cos(velocity_angle), -np.sin(velocity_angle)], [np.sin(velocity_angle), np.cos(velocity_angle)]])
                cv = rotation @ cv_default @ rotation.transpose(1, 0)
                noise_x, noise_y = np.random.multivariate_normal([0,0], cv)
                answer_list[i][0] = data_now[index][0] + final_vel[0] + noise_x
                answer_list[i][1] = data_now[index][1] + final_vel[1] + noise_y

        return answer_mean, answer_list    
    
    def test_figure(self, fig, model, device, test_loader):
        index = 10
        model.eval()
        r = 0
        #r = np.random.randint(len(test_loader.dataset.test_data))
        #data = test_loader.dataset.test_data[r].to(device)
        #label = test_loader.dataset.test_labels[r].to(device)
        
        data = train_loader.dataset.train_data[r].to(device)
        label = train_loader.dataset.train_labels[r].to(device).unsqueeze(0)
        
        mu, sig, corr, coef = model(data.unsqueeze(0), label[:,:,0])
        mu, sig, corr, coef = DCN(mu.squeeze(0)), DCN(sig.squeeze(0)), DCN(corr.squeeze(0)), DCN(coef.squeeze(0))
        model.train()

        ax = fig.add_subplot(2,2,(3,4))
        
        # drawing sampled 2d histogram from real answer (thanks to analytic form of vicsek, impossible for real bird)
        answer_mean, answer_list = self.vicsek_simulation(DCN(data), index = index, iter_num = 1000)        
        # drawing contour of VAINS result
        delta = 0.01
        x = np.arange(-5.0, 5.0, delta)
        y = np.arange(-5.0, 5.0, delta)
        X, Y = np.meshgrid(x, y)
        Z = corr_bivariate(np.stack((X,Y), axis = -1), mu[index], sig[index], corr[index], coef[index])
        
        ax.hist2d(answer_list[:,0], answer_list[:,1], bins = [5, 5], range = [[answer_mean[0]-1, answer_mean[0]+1],[answer_mean[1]-1, answer_mean[1]+1]],cmap = cm.gray.reversed())
        cs = ax.contour(X, Y, Z, color = 'white')
        ax.clabel(cs, inline=1, fontsize=8)
        
        data = DCN(data)
        label = DCN(label)
        
        ax.scatter(data[:,0], data[:, 1], alpha = 0.2)
        ax.scatter(data[index, 0], data[index, 1], s = 30, c = 'b', label = 'origin')
        ax.scatter(answer_mean[0], answer_mean[1], s = 30, c = 'g', label = 'expected')
        ax.scatter(label[0,index, 1], label[0,index, 2], s = 30, c = 'r', label = 'label')
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('X', fontsize = 15)
        ax.set_ylabel('Y', fontsize = 15)
        #ax.set_xlim(answer_mean[0]-1, answer_mean[0]+1)
        #ax.set_ylim(answer_mean[1]-1, answer_mean[1]+1)
        
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        
        ax.legend()
        return [ax] # New version, velocity alignment


######################
## 4. Vicsek_linear ##
######################

class Vicsek_linear(): 
    
    def __init__(self):
        self.name = 'VM_ln'
        self.rule_name = 'Agent_'+str(self.name)
        self.speed_boost = 1
        self.count = 0
        
    def assign_pp(self, plugin_parameters):
        self.pp = plugin_parameters
        assert 'agent_num' in plugin_parameters
        self.agent_num = self.pp['agent_num']
        assert 'neighbor_dist' in plugin_parameters
        self.neighbor_dist = self.pp['neighbor_dist']
        assert 'neighbor_angle' in plugin_parameters
        self.neighbor_angle = self.pp['neighbor_angle']
        assert 'noise_type' in plugin_parameters
        self.noise_type = self.pp['noise_type']
        assert 'noise_strength' in plugin_parameters
        self.noise_strength = self.pp['noise_strength']
        self.rule_name += '_a' + str(self.agent_num) + '_na' + str(self.neighbor_angle) + '_nt' + str(self.noise_type) + '_ns' + str(self.noise_strength)

    # Data version
    
    def neighbor_fn_data(self, data_now): 

        data_pos = data_now[:,:2] # position
        distance = sp.spatial.distance_matrix(data_pos, data_pos)
        norm_distance = (-distance/6.)+1
        angle = np.array([[angle_between_vec(x,y) for y in data_now[:,:4]] for x in data_now[:,:4]])
        
        return np.array([(data_now.T*norm_distance[i]).T for i in range(norm_distance.shape[0])])

    def agg_fn_data(self, data_now, final_vel):
        vx = final_vel[:,0]
        vy = final_vel[:,1]
        data_next = np.stack((data_now[:,0]+vx, data_now[:,1]+vy, vx, vy,  data_now[:,4]), axis = 1)

        return data_next, data_next[:, [4,0,1]]
    
    ## DataGen generator
    
    def __next__(self):
        N = self.agent_num
        while(True):
            #theta = np.random.uniform(-np.pi, np.pi, (N, 1))
            #data_pos = np.random.uniform(-1, 1, (N, 2))+np.random.uniform(-3, 3, (1,2)) # Total length of 5, 2 velocity / 2 pos / 1 group
            #data_grp = np.ones((N,1))
            #data = np.concatenate((np.cos(theta)*1/self.speed_boost, np.sin(theta)*1/self.speed_boost, data_pos, data_grp), axis = 1) # position range : 4 unit, velocity magnitude : 0.25 
            
            self.count += 1
            if self.count % 100 == 0:
                print(self.count)
                
            length = np.sqrt(np.random.uniform(0, 9, (N,1)))
            angle = np.pi * np.random.uniform(0, 2, (N,1))
            data_pos =  length * np.concatenate((np.cos(angle),np.sin(angle)), axis = 1)
            theta = np.pi * np.random.uniform(0, 2, (N,1))
            data_grp = np.ones((N,1))
            data = np.concatenate((data_pos,np.cos(theta)/self.speed_boost, np.sin(theta)/self.speed_boost, data_grp), axis = 1)
            
            data_now = deepcopy(data)
            data_input = self.neighbor_fn_data(data_now)

            # viscek model calculation
            vx = np.zeros(len(data_input))
            vy = np.zeros(len(data_input))
            for i in range(len(data_input)):
                #print(data_input[i].shape)
                vx_sum = np.sum(data_input[i][:,2])
                vy_sum = np.sum(data_input[i][:,3])
                v2 = self.speed_boost*((vx_sum**2+vy_sum**2)**0.5)
                vx[i], vy[i] = vx_sum/v2, vy_sum/v2
            predicted_list = np.stack((vx, vy), axis = 1)    
            
            angle_change = np.radians(np.clip(np.array([angle_between(data_now[i,2:4], predicted_list[i]) for i in range(predicted_list.shape[0])]), -120, 120))
            if self.noise_type == "angle":
                angle_change += np.random.normal(0, self.noise_strength, angle_change.shape)
            rot_mat = np.array([np.array([[np.cos(x), -np.sin(x)], [np.sin(x), np.cos(x)]]) for x in angle_change])
            final_vel = np.matmul(np.expand_dims(data_now[:, 2:4], -2), rot_mat).squeeze()
            if self.noise_type == "pos" :
                final_vel += np.random.normal(0, self.noise_strength, final_vel.shape)
            if self.noise_type == 'corr':
                cv_default = np.array([[np.square(self.noise_strength), 0],[0, np.square(self.noise_strength)/3]])
                for i in range(final_vel.shape[0]):
                    velocity_angle = np.radians(angle_between([1, 0], final_vel[i]))
                    rotation = np.array([[np.cos(velocity_angle), -np.sin(velocity_angle)], [np.sin(velocity_angle), np.cos(velocity_angle)]])
                    cv = rotation @ cv_default @ rotation.transpose(1, 0)
                    final_vel[i] += np.random.multivariate_normal([0,0], cv)
                
            _, answer = self.agg_fn_data(data_now, final_vel)

            return np.array(data), answer
        
    # Test figure    
    
    def vicsek_simulation(self, data_now, index, iter_num):
        data_input = self.neighbor_fn_data(data_now)[index]

        # viscek model calculation
        vx_sum = np.sum(data_input[:,2])
        vy_sum = np.sum(data_input[:,3])
        v2 = self.speed_boost*((vx_sum**2+vy_sum**2)**0.5)
        vx, vy = vx_sum/v2, vy_sum/v2
        predicted_list = np.array([vx, vy])

        angle_change = np.radians(np.clip(angle_between(data_now[index][2:4], predicted_list), -self.neighbor_angle, self.neighbor_angle))
        rot_mat = np.array([[np.cos(angle_change), -np.sin(angle_change)], [np.sin(angle_change), np.cos(angle_change)]])
        final_vel = np.matmul(np.expand_dims(data_now[index][2:4], -2), rot_mat).squeeze()
        answer_mean = np.array([data_now[index][0] + final_vel[0], data_now[index][1] + final_vel[1]])
        answer_list = np.zeros((iter_num, 2))
        if self.noise_type == 'angle':
            for i in range(iter_num):
                sample_angle = angle_change + np.random.normal(0, self.noise_strength, angle_change.shape)
                rot_mat = np.array([[np.cos(sample_angle), -np.sin(sample_angle)], [np.sin(sample_angle), np.cos(sample_angle)]])
                final_vel = np.matmul(np.expand_dims(data_now[index][2:4], -2), rot_mat).squeeze()
                answer_list[i][0] = data_now[index][0] + final_vel[0]
                answer_list[i][1] = data_now[index][1] + final_vel[1]
        elif self.noise_type == 'pos':
            for i in range(iter_num):
                answer_list[i][0] = data_now[index][0] + final_vel[0]
                answer_list[i][1] = data_now[index][1] + final_vel[1]
            answer_list += np.random.normal(0, self.noise_strength, answer_list.shape)
        elif self.noise_type == 'corr':
            cv_default = np.array([[np.square(self.noise_strength), 0],[0, np.square(self.noise_strength)/3]])
            for i in range(iter_num):
                velocity_angle = np.radians(angle_between([1, 0], final_vel))
                rotation = np.array([[np.cos(velocity_angle), -np.sin(velocity_angle)], [np.sin(velocity_angle), np.cos(velocity_angle)]])
                cv = rotation @ cv_default @ rotation.transpose(1, 0)
                noise_x, noise_y = np.random.multivariate_normal([0,0], cv)
                answer_list[i][0] = data_now[index][0] + final_vel[0] + noise_x
                answer_list[i][1] = data_now[index][1] + final_vel[1] + noise_y

        return answer_mean, answer_list    
    
    def test_figure(self, fig, model, device, test_loader):
        index = 10
        model.eval()
        r = 0
        #r = np.random.randint(len(test_loader.dataset.test_data))
        #data = test_loader.dataset.test_data[r].to(device)
        #label = test_loader.dataset.test_labels[r].to(device)
        
        data = train_loader.dataset.train_data[r].to(device)
        label = train_loader.dataset.train_labels[r].to(device).unsqueeze(0)
        
        mu, sig, corr, coef = model(data.unsqueeze(0), label[:,:,0])
        mu, sig, corr, coef = DCN(mu.squeeze(0)), DCN(sig.squeeze(0)), DCN(corr.squeeze(0)), DCN(coef.squeeze(0))
        model.train()

        ax = fig.add_subplot(2,2,(3,4))
        
        # drawing sampled 2d histogram from real answer (thanks to analytic form of vicsek, impossible for real bird)
        answer_mean, answer_list = self.vicsek_simulation(DCN(data), index = index, iter_num = 1000)        
        # drawing contour of VAINS result
        delta = 0.01
        x = np.arange(-5.0, 5.0, delta)
        y = np.arange(-5.0, 5.0, delta)
        X, Y = np.meshgrid(x, y)
        Z = corr_bivariate(np.stack((X,Y), axis = -1), mu[index], sig[index], corr[index], coef[index])
        
        ax.hist2d(answer_list[:,0], answer_list[:,1], bins = [5, 5], range = [[answer_mean[0]-1, answer_mean[0]+1],[answer_mean[1]-1, answer_mean[1]+1]],cmap = cm.gray.reversed())
        cs = ax.contour(X, Y, Z, color = 'white')
        ax.clabel(cs, inline=1, fontsize=8)
        
        data = DCN(data)
        label = DCN(label)
        
        ax.scatter(data[:,0], data[:, 1], alpha = 0.2)
        ax.scatter(data[index, 0], data[index, 1], s = 30, c = 'b', label = 'origin')
        ax.scatter(answer_mean[0], answer_mean[1], s = 30, c = 'g', label = 'expected')
        ax.scatter(label[0,index, 1], label[0,index, 2], s = 30, c = 'r', label = 'label')
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('X', fontsize = 15)
        ax.set_ylabel('Y', fontsize = 15)
        #ax.set_xlim(answer_mean[0]-1, answer_mean[0]+1)
        #ax.set_ylim(answer_mean[1]-1, answer_mean[1]+1)
        
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        
        ax.legend()
        return [ax] # New version, velocity alignment


###########################################
## 5. Active Ornstein-Uhlenbeck Particle ##
###########################################

def force(delta, R):
    r = np.expand_dims(np.power((delta[:,0]**2 + delta[:,1]**2), 0.5), axis = -1)
    return -3*(delta)*r*np.exp(-(r**3)/R**3)/R**3

def potential(data, R):
    pot = np.zeros_like(data)
    for i in range(data.shape[0]):
        delta = data-data[i]
        f = force(delta, R)
        #print(delta.shape, f.shape)
        pot[i] = np.sum(f, axis = 0)
    return pot


class AOUP():
    def __init__(self):
        self.name = 'AOUP'
        self.rule_name = 'Agent_' + str(self.name)
        
    def assign_pp(self, plugin_parameters):
        self.pp = plugin_parameters
        assert 'agent_num' in plugin_parameters
        self.agent_num = self.pp['agent_num']
        assert 'dt' in plugin_parameters
        self.dt = self.pp['dt']
        assert 'data_step' in plugin_parameters
        self.data_step = self.pp['data_step']
        assert 'label_step' in plugin_parameters
        self.label_step = self.pp['label_step']
        assert 'state_num' in plugin_parameters
        self.state_num = self.pp['state_num']
        assert 'answer_num' in plugin_parameters
        self.answer_num = self.pp['answer_num']
        assert 'const_num' in plugin_parameters
        self.const_num = self.pp['const_num']
    
        self.heating_step = 5
        self.count = 0
        self.R_const = None
    
    def assign_const(self, train_size, test_size, R_const_list):
        #Tconst_list = [[train_T_start, train_T_end, train_T_step], [test_T_start,test_T_end, test_T_step]]
        train_R_start, train_R_end, train_R_step = R_const_list[0]
        test_R_start, test_R_end, test_R_step = R_const_list[1]
        R_const = []
        if train_R_step > 0:
        
            train_num = int((train_R_end-train_R_start)/train_R_step)
            train_batch = int(train_size/train_num)+1
            for i in range(train_num):
                R_const += [train_R_start + i*train_R_step] * train_batch

            R_const = R_const[:train_size]

            test_num =int((test_R_end-test_R_start)/test_R_step)
            test_batch = int(test_size/test_num)+1
            for i in range(test_num):
                R_const += [test_R_start + i*test_R_step] * test_batch

            self.R_const = R_const
        else:
            self.R_const = [train_R_start]*(train_size + test_size + 100)
    
    def __next__(self):
        while True:
            self.count += 1
            if self.count % 100 == 0:
                print(self.count)
            
            data = np.zeros((self.agent_num, self.state_num))
            N = 100
            dt = self.dt
            jump_step = 10
            
            R = self.R_const[self.count]
            k=0.1
            gamma = 0.5
            tau = 0.2
            U0 = 0.05
            Da = U0**2/2
            T = (0.2)*(U0**2)*tau/gamma
            
            length = np.sqrt(np.random.uniform(0, 64, (N, 1)))
            angle = np.pi * np.random.uniform(0, 4, (N, 1))
            data = length * np.concatenate((np.cos(angle), np.sin(angle)), axis=1)
            vel = np.zeros_like(data)
            noise_length = np.sqrt(np.random.uniform(0, 0.05, (N, 1)))
            noise_angle = np.pi * np.random.uniform(0, 4, (N, 1))
            noise = noise_length * np.concatenate((np.cos(noise_angle), np.sin(noise_angle)), axis=1)
            data_list = []
            vel_list = []
            noise_list = []

            for i in range(self.heating_step*jump_step):
                noise_vel = (-noise*dt+np.sqrt(2*Da)*np.random.normal(loc = 0, scale = np.sqrt(dt), size = noise.shape))/tau
                noise = noise + noise_vel
                vel = ((potential(data, R)-k*data)*dt + np.sqrt(2*gamma*T)*np.random.normal(loc = 0, scale = np.sqrt(dt), size = noise.shape))/gamma + noise
                data = data + vel

            for i in range(self.data_step*jump_step):
                noise_vel = (-noise*dt+np.sqrt(2*Da)*np.random.normal(loc = 0, scale = np.sqrt(dt), size = noise.shape))/tau
                noise = noise + noise_vel
                vel = ((potential(data, R)-k*data)*dt + np.sqrt(2*gamma*T)*np.random.normal(loc = 0, scale = np.sqrt(dt), size = noise.shape))/gamma + noise * dt
                data = data + vel
                noise_list.append(noise*dt)
                vel_list.append(vel*dt)
                data_list.append(data)
            
            data_list = data_list[::10]
            vel_list = vel_list[::10]
            noise_list = noise_list[::10]
            R_const_list = np.ones((self.data_step, 100, 1))*self.R_const[self.count]
            data_data = np.concatenate((data_list, vel_list, noise_list, R_const_list), axis = -1)
            data_list = []
            vel_list = []
            noise_list = []
            for i in range(self.label_step*jump_step):
                noise_vel = (-noise*dt+np.sqrt(2*Da)*np.random.normal(loc = 0, scale = np.sqrt(dt), size = noise.shape))/tau
                noise += noise_vel
                #print(data.shape)
                vel = ((potential(data, R)-k*data)*dt + np.sqrt(2*gamma*T)*np.random.normal(loc = 0, scale = np.sqrt(dt), size = noise.shape))/gamma + noise * dt
                data = data + vel
                noise_list.append(noise/dt)
                vel_list.append(vel/dt)
                data_list.append(data)
                
            data_list = data_list[::10]
            vel_list = vel_list[::10]
            noise_list = noise_list[::10]
            
            answer_data = np.concatenate((data_list, vel_list, noise_list), axis = -1)
            
            return data_data, answer_data

#############################
## 6. Chimney Swift Flock  ##
##############squeue###############

class Flock():
    '''
    data : 3D Labratory swarm trajectory data from "Three-dimensional time-resolved trajectories from laboratory insect swarms"
    https://www.nature.com/articles/sdata201936.pdf

    '''

    def __init__(self):
        self.name = 'FL_pos'
        self.rule_name = 'Agent_' + str(self.name)

    # assign time_interval and other constants

    def assign_pp(self, plugin_parameters):
        self.pp = plugin_parameters
        assert 'ob_num' in plugin_parameters
        self.ob_num = self.pp['ob_num']
        assert 'time_interval' in plugin_parameters

        self.time_interval = self.pp['time_interval']
        self.state_num = 9  # x, y, z, vx, vy, vz, hdg, hdg_rate, V
        self.answer_num = 9  # same as state
        self.count = 0
        self.max_agent = 0

        self.time_list = []  # Time snapshot
        self.time_len_list = []  # Number of agents in each time step
        self.id_set_list = []  # IDs of agents in each time step
        self.id_list = []  # Trajectory of each agent
        self.id_len_list = []  # Length of trajectory of each agent
        self.len_change_list = []  # Change of number of agents between successive steps
        self.id_change_list = []  # Number of disappeared ID between successive steps

        #print('pp assign finished')

    ## Construct time and id list from raw data

    def calc(self):
        hf = h5py.File(self.ob_num + '_nodes.hdf5', 'r')
        df_list = []
        for i in range(1000, 17000):
            data = hf[str(i)]
            x = pd.DataFrame([data['tid'].value, data['x'].value, data['y'].value, data['z'].value,
                              np.ones_like(data['tid'].value) * i, data['vx'].value, data['vy'].value, data['vz'].value,
                              data['hdg'].value, data['hdg_rate'].value, data['V'].value]).T
            x.columns = ['ID', 'x', 'y', 'z', 't', 'vx', 'vy', 'vz', 'hdg', 'hdg_rate', 'V']
            df_list.append(x)

        ob = pd.concat(df_list, ignore_index=True)
        ob[['hdg', 'hdg_rate']] *= np.pi / 180.0

        self.ob = ob

        times = sorted(set(ob['t'].values))
        ids = sorted(set(ob['ID'].values))

        self.time_list = []  # Time snapshot
        self.time_len_list = []  # N}umber of agents in each time step
        for i in times:
            self.time_list.append(ob[ob['t'] == i])
            self.time_len_list.append(len(self.time_list[-1]))
        self.max_agent = max(self.time_len_list)

        self.id_set_list = []  # IDs of agents in each time step
        for t in self.time_list:
            self.id_set_list.append(set(t['ID'].values))

        self.id_list = []  # Trajectory of each agent
        self.id_len_list = []  # Length of trajectory of each agent
        for i in ids:
            self.id_list.append(ob[ob['ID'] == i])
            self.id_len_list.append(len(self.id_list[-1]))

        self.len_change_list = []  # Change of number of agents between successive steps
        self.id_change_list = []  # Number of disappeared ID between successive steps

        for i in range(len(self.time_list) - 1):
            self.len_change_list.append(self.time_len_list[i] - self.time_len_list[i + 1])
            self.id_change_list.append(
                self.time_len_list[i] - len(set.intersection(self.id_set_list[i], self.id_set_list[i + 1])))

    ## Assign generating order (neccessary for dataset generation)

    def assign_order(self, start_cut, end_cut, block_size, shuffle=False):
        self.min_time = start_cut
        self.max_time = (len(self.time_list) - end_cut) - self.time_interval
        self.order = np.arange(self.max_time)
        if shuffle:
            z = [list(np.arange(block_size * i, block_size * (i + 1))) for i in
                 range(int(self.min_time / block_size), int(self.max_time / block_size))]
            np.random.shuffle(z)
            from functools import reduce
            self.order = reduce(lambda x, y: x + y, z)

    ## DataGen generator

    def __next__(self):
        while True:
            i = self.order[self.count]
            self.count += 1
            if self.count % 100 == 0:
                print(self.count)
            data_input = self.time_list[i][['x', 'y', 'z', 'vx', 'vy', 'vz', 'hdg', 'hdg_rate', 'V']].values
            data_input = np.concatenate((data_input, np.ones((data_input.shape[0], 1))), axis=1)
            data = np.zeros((self.max_agent, self.state_num))
            data = np.concatenate((data, -np.ones((data.shape[0], 1))), axis=1)
            data[:data_input.shape[0], :data_input.shape[1]] = data_input

            mask_input = np.where(self.time_list[i]['ID'].isin(self.id_set_list[i + self.time_interval]) == True, 1.,
                                  0.)
            mask = np.zeros(self.max_agent)
            mask[:mask_input.shape[0]] = mask_input

            answer = np.zeros((self.max_agent, 1 + self.answer_num))
            answer[:mask.shape[0], 0] = mask

            for t, x in enumerate(self.time_list[i]['ID']):
                if answer[t][0] == 1:
                    answer[t][1:] = (
                    self.time_list[i + self.time_interval][self.time_list[i + self.time_interval]['ID'] == x][
                        ['x', 'y', 'z', 'vx', 'vy', 'vz', 'hdg', 'hdg_rate', 'V']]).values

            return data, answer

    # Test figure

    def test_figure(self, fig, model, device, test_loader):

        model.eval()
        r = np.random.randint(len(test_loader.dataset.test_data))
        data = test_loader.dataset.test_data[r].to(device)
        label = DCN(test_loader.dataset.test_labels[r].to(device))
        predicted_list = DCN(model(data.unsqueeze(0)))
        model.train()

        ax1 = fig.add_subplot(2, 2, 3)
        ax1.plot(np.linspace(-5, 5), np.linspace(-5, 5), color='r', lw=2)

        mask = [True if label[i, 0] == 1 else False for i in range(label.shape[0])]
        label = label[mask]
        predicted_list = predicted_list[:, mask]

        ax1.scatter(label[:, 1], predicted_list[:, :, 0], alpha=0.5, label=r'$x$')
        ax1.scatter(label[:, 2], predicted_list[:, :, 1], alpha=0.5, label=r'$y$')
        ax1.scatter(label[:, 3], predicted_list[:, :, 2], alpha=0.5, label=r'$z$')

        ax1.set_aspect('equal', adjustable='box')
        ax1.set_xlabel('Expected', fontsize=15)
        ax1.set_ylabel('Predicted', fontsize=15)
        ax1.set_xlim(-1, 1)
        ax1.set_ylim(-1, 1)
        ax1.legend()

        ax2 = fig.add_subplot(2, 2, 4)
        ax2.plot(np.linspace(-5, 5), np.linspace(-5, 5), color='r', lw=2)

        mask = [True if label[i, 0] == 1 else False for i in range(label.shape[0])]
        label = label[mask]
        predicted_list = predicted_list[:, mask]

        ax2.scatter(label[:, 4], predicted_list[:, :, 3], alpha=0.5, label=r'$v_x$')
        ax2.scatter(label[:, 5], predicted_list[:, :, 4], alpha=0.5, label=r'$v_y$')
        ax2.scatter(label[:, 6], predicted_list[:, :, 5], alpha=0.5, label=r'$v_z$')

        ax2.set_aspect('equal', adjustable='box')
        ax2.set_xlabel('Expected', fontsize=15)
        ax2.set_ylabel('Predicted', fontsize=15)
        ax2.set_xlim(-1, 1)
        ax2.set_ylim(-1, 1)
        ax2.legend()

        return [ax1, ax2]


class Flock_LSTM():
    '''
    data : 3D Labratory swarm trajectory data from "Three-dimensional time-resolved trajectories from laboratory insect swarms"
    https://www.nature.com/articles/sdata201936.pdf

    '''

    def __init__(self):
        self.name = 'FL_pos'
        self.rule_name = 'Agent_' + str(self.name)

    # assign time_interval and other constants

    def assign_pp(self, plugin_parameters):
        self.pp = plugin_parameters
        assert 'ob_num' in plugin_parameters
        self.ob_num = self.pp['ob_num']
        assert 'time_interval' in plugin_parameters

        self.time_interval = self.pp['time_interval']
        self.state_num = 9  # x, y, z, vx, vy, vz, hdg, hdg_rate, V
        self.answer_num = 9  # same as state
        self.count = 0
        self.max_agent = 0
        self.step_num = self.pp['time_interval']
        self.time_interval = 15
        self.jump_interval = 75
       
        self.time_list = []  # Time snapshot
        self.time_len_list = []  # Number of agents in each time step
        self.id_set_list = []  # IDs of agents in each time step
        self.id_list = []  # Trajectory of each agent
        self.id_len_list = []  # Length of trajectory of each agent
        self.len_change_list = []  # Change of number of agents between successive steps
        self.id_change_list = []  # Number of disappeared ID between successive steps

        print('pp assign finished')

    ## Construct time and id list from raw data

    def calc(self, start_time, end_time):
        hf = h5py.File(self.ob_num + '_nodes.hdf5', 'r')
        df_list = []
        for i in range(start_time, end_time):
            data = hf[str(i)]
            x = pd.DataFrame([data['tid'].value, data['x'].value, data['y'].value, data['z'].value,
                              np.ones_like(data['tid'].value) * i, data['vx'].value, data['vy'].value, data['vz'].value,
                              data['hdg'].value, data['hdg_rate'].value, data['V'].value]).T
            x.columns = ['ID', 'x', 'y', 'z', 't', 'vx', 'vy', 'vz', 'hdg', 'hdg_rate', 'V']
            df_list.append(x)

        ob = pd.concat(df_list, ignore_index=True)
        ob[['hdg', 'hdg_rate']] *= np.pi / 180.0

        self.ob = ob

        times = sorted(set(ob['t'].values))
        ids = sorted(set(ob['ID'].values))

        self.time_list = []  # Time snapshot
        self.time_len_list = []  # N}umber of agents in each time step
        for i in times:
            self.time_list.append(ob[ob['t'] == i])
            self.time_len_list.append(len(self.time_list[-1]))
        self.max_agent = max(self.time_len_list)

        self.id_set_list = []  # IDs of agents in each time step
        for t in self.time_list:
            self.id_set_list.append(set(t['ID'].values))

        self.id_list = []  # Trajectory of each agent
        self.id_len_list = []  # Length of trajectory of each agent
        for i in ids:
            self.id_list.append(ob[ob['ID'] == i])
            self.id_len_list.append(len(self.id_list[-1]))

        self.len_change_list = []  # Change of number of agents between successive steps
        self.id_change_list = []  # Number of disappeared ID between successive steps

        for i in range(len(self.time_list) - 1):
            self.len_change_list.append(self.time_len_list[i] - self.time_len_list[i + 1])
            self.id_change_list.append(
                self.time_len_list[i] - len(set.intersection(self.id_set_list[i], self.id_set_list[i + 1])))

    ## Assign generating order (neccessary for dataset generation)

    def assign_order(self, start_cut, end_cut, block_size, shuffle=False):
        self.min_time = start_cut
        self.max_time = (len(self.time_list) - end_cut) - self.time_interval
        self.order = np.arange(0, self.max_time, self.jump_interval)
        if shuffle:
            z = [list(np.arange(block_size * i, block_size * (i + 1))) for i in
                 range(int(self.min_time / block_size), int(self.max_time / block_size))]
            np.random.shuffle(z)
            from functools import reduce
            self.order = reduce(lambda x, y: x + y, z)

    def assign_order(self, start_cut, end_cut, block_size, shuffle=False):
        self.min_time = start_cut
        self.max_time = (len(self.time_list) - end_cut) - self.time_interval
        self.order = np.arange(self.max_time)
        if shuffle:
            z = [list(np.arange(block_size * i, block_size * (i + 1))) for i in
                 range(int(self.min_time / block_size), int(self.max_time / block_size))]
            np.random.shuffle(z)
            from functools import reduce
            self.order = reduce(lambda x, y: x + y, z)

    ## DataGen generator

    def __next__(self):
        while True:
            i = self.order[self.count]
            self.count += 1
            if self.count % 100 == 0:
                print(self.count)
                
            full_set = set([])
            for s in range(system.step_num+1):
                full_set = set.union(full_set, self.time_list[int(i+(s*self.time_interval))]['ID'])
            full_set = np.sort(np.array(list(full_set)))
            
            data_input = self.time_list[i][['x', 'y', 'z', 'vx', 'vy', 'vz', 'hdg', 'hdg_rate', 'V']].values
            data_input = np.concatenate((data_input, np.ones((data_input.shape[0], 1))), axis=1)
            data = np.zeros((self.max_agent, self.state_num))
            data = np.concatenate((data, (-1)*np.ones((data.shape[0], 1))), axis=1)
            data[idx2idx(full_set, np.sort(np.array(list(set(self.time_list[int(i)]['ID'])))))] = data_input    
            
            answer_list = []

            previous_set = set([])
            present_set = set(self.time_list[int(i)]['ID'])
            for s in range(self.step_num):
                mask_input_new = np.zeros(self.max_agent)
                mask_input_next = np.zeros(self.max_agent)
                next_set = set(self.time_list[int(i + (s+1) * self.time_interval)]['ID'])
                
                # A : agents which exists in next step AND new entry (ready for hidden state init)
                new_set = present_set - previous_set
                mask_input_new[idx2idx(full_set, np.sort(np.array(list(new_set))))] = 1 
                
                # B1 : agents in next step
                mask_input_next[idx2idx(full_set, np.sort(np.array(list(next_set))))] = 1
                
                # B2 : agents which exists currently and next step (covers A)
                present_next_set = set.intersection(present_set, next_set)
                mask_input_next[idx2idx(full_set, np.sort(np.array(list(present_next_set))))] = 2 
                
                answer = np.zeros((self.max_agent, 2 + self.answer_num))
                answer[:, 0] = mask_input_new
                answer[:, 1] = mask_input_next

                for x in self.time_list[int(i + (s+1) * self.time_interval)]['ID']:
                    idx_x = np.argwhere(full_set==x)[0][0]
                    answer[idx_x][2:] = (
                    self.time_list[int(i + (s+1) * self.time_interval)][self.time_list[int(i + (s+1) * self.time_interval)]['ID'] == x][
                        ['x', 'y', 'z', 'vx', 'vy', 'vz', 'hdg', 'hdg_rate', 'V']]).values

                answer_list.append(answer)
                previous_set = present_set
                present_set = next_set 

            return data, answer_list

    # Test figure

    def test_figure(self, fig, model, device, test_loader):

        model.eval()
        r = np.random.randint(len(test_loader.dataset.test_data))
        data = test_loader.dataset.test_data[r].to(device)
        label = DCN(test_loader.dataset.test_labels[r].to(device))
        predicted_list = DCN(model(data.unsqueeze(0)))
        model.train()

        ax1 = fig.add_subplot(2, 2, 3)
        ax1.plot(np.linspace(-5, 5), np.linspace(-5, 5), color='r', lw=2)

        mask = [True if label[i, 0] == 1 else False for i in range(label.shape[0])]
        label = label[mask]
        predicted_list = predicted_list[:, mask]

        ax1.scatter(label[:, 1], predicted_list[:, :, 0], alpha=0.5, label=r'$x$')
        ax1.scatter(label[:, 2], predicted_list[:, :, 1], alpha=0.5, label=r'$y$')
        ax1.scatter(label[:, 3], predicted_list[:, :, 2], alpha=0.5, label=r'$z$')

        ax1.set_aspect('equal', adjustable='box')
        ax1.set_xlabel('Expected', fontsize=15)
        ax1.set_ylabel('Predicted', fontsize=15)
        ax1.set_xlim(-1, 1)
        ax1.set_ylim(-1, 1)
        ax1.legend()

        ax2 = fig.add_subplot(2, 2, 4)
        ax2.plot(np.linspace(-5, 5), np.linspace(-5, 5), color='r', lw=2)

        mask = [True if label[i, 0] == 1 else False for i in range(label.shape[0])]
        label = label[mask]
        predicted_list = predicted_list[:, mask]

        ax2.scatter(label[:, 4], predicted_list[:, :, 3], alpha=0.5, label=r'$v_x$')
        ax2.scatter(label[:, 5], predicted_list[:, :, 4], alpha=0.5, label=r'$v_y$')
        ax2.scatter(label[:, 6], predicted_list[:, :, 5], alpha=0.5, label=r'$v_z$')

        ax2.set_aspect('equal', adjustable='box')
        ax2.set_xlabel('Expected', fontsize=15)
        ax2.set_ylabel('Predicted', fontsize=15)
        ax2.set_xlim(-1, 1)
        ax2.set_ylim(-1, 1)
        ax2.legend()

        return [ax1, ax2]