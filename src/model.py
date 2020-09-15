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
from src.utils import *
from src.system import *

## Architecture


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
                w = torch.sigmoid(torch.add(self.att(z).squeeze(-1), -mask))
                #w = F.dropout(softmax(torch.add(F.leaky_relu(self.att(z), 0.2).squeeze(-1), -mask)), p = self.dropout, training = self.training)
                p_list.append(torch.bmm(w, v))

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

class Module_GAT_VC_split(nn.Module):
    def __init__(self, cfg_enc, cfg_att, cfg_dec, cfg_mu, cfg_sig, D_att, D_att_num, D_agent, block_type, att_type, dropout):
        super(Module_GAT_VC_split, self).__init__()

        self.D_att = D_att
        self.D_att_num = D_att_num
        self.D_agent = D_agent
        self.block_type = block_type
        self.att_type = att_type
        self.dropout = dropout

        self.key = cfg_Block(block_type, cfg_enc, D_agent, 'MS', False, False)
        self.query = cfg_Block(block_type, cfg_enc, D_agent, 'MS', False, False)
        self.value = cfg_Block(block_type, cfg_enc, D_agent, 'MS', False, False)

        if self.att_type == 'gat':
            self.att = cfg_Block(block_type, cfg_att, D_agent, 'MS', False, False)
        elif self.att_type == 'add':
            self.att1 = cfg_Block(block_type, [D_att, D_att], D_agent, 'MS', False, False)
            self.att2 = cfg_Block(block_type, [D_att, D_att], D_agent, 'MS', False, False)
            self.att3 = cfg_Block(block_type, [D_att, 1], D_agent, 'MS', False, False)
        elif self.att_type == 'mul':
            self.att = None
        else:
            print('NOT IMPLEMENTED : gat att block creation')

        self.dec_x = cfg_Block(block_type, cfg_dec, D_agent, 'MS', False, False)
        self.mu_dec_x = cfg_Block(block_type, cfg_mu, D_agent, 'MS', False, False)
        self.sig_dec_x = cfg_Block(block_type, cfg_sig, D_agent, 'MS', False, False)

        self.dec_y = cfg_Block(block_type, cfg_dec, D_agent, 'MS', False, False)
        self.mu_dec_y = cfg_Block(block_type, cfg_mu, D_agent, 'MS', False, False)
        self.sig_dec_y = cfg_Block(block_type, cfg_sig, D_agent, 'MS', False, False)

        if block_type == 'mlp':
            self.D_k = 1
            self.D_s = 1
        elif block_type == 'res':
            self.D_k = 1
            self.D_s = 1

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if type(m.bias) != type(None):
                    m.bias.data.zero_()

    def forward(self, x):
        #print(x.shape, hidden[0].shape, hidden[1].shape)
        softmax = nn.Softmax(dim=-1)
        k = self.key(x) 
        q = self.query(x)
        v = self.value(x)
            
        p_list = [[v] for _ in range(self.D_att_num)]

        mask_const = 10000.
        mask = torch.zeros(x.shape[0], x.shape[1], x.shape[1]).to(x.device)
        mask = mask+torch.eye(x.shape[1], x.shape[1]).to(x.device)*mask_const
        mask = mask.unsqueeze(-1).repeat(1, 1, 1, self.D_att_num)

        if self.att_type == 'gat':
            kz = [k for _ in range(k.shape[1])]
            qz = [q for _ in range(q.shape[1])]
            z = torch.cat((torch.stack(kz, dim=-2), torch.stack(qz, dim=-3)), -1)
            w = torch.sigmoid(torch.add(self.att(z).squeeze(-1), -mask))
            #w = torch.tanh(torch.mul(self.att(z).squeeze(-1), 1-mask))/(self.D_agent)
            for i in range(self.D_att_num):
                #p_list[i].append(torch.matmul(w[:,:,:,i], v[:, :, head_dim * i: head_dim * (i + 1)]))
                p_list[i].append(torch.matmul(w[:,:,:,i], v))
            
        c_list = [[] for _ in range(self.D_att_num)] 
        for i in range(self.D_att_num):
            c_list[i] = torch.cat(p_list[i], dim=-1)

        d_x = self.dec_x(c_list[0])
        mu_x = self.mu_dec_x(d_x).squeeze()
        sig_x = F.softplus(self.sig_dec_x(d_x)).squeeze()

        d_y = self.dec_y(c_list[1])
        mu_y = self.mu_dec_y(d_y).squeeze()
        sig_y = F.softplus(self.sig_dec_y(d_y)).squeeze()

        return (mu_x, sig_x), (mu_y, sig_y)

class Module_GAT_AOUP_split(nn.Module):
    def __init__(self, cfg_init, cfg_lstm, cfg_enc, cfg_self, cfg_att, cfg_dec, cfg_mu, cfg_sig, D_att, D_att_num, D_agent, block_type, att_type, dropout):
        super(Module_GAT_AOUP_split, self).__init__()

        self.D_att = D_att
        self.D_att_num = D_att_num
        self.D_agent = D_agent
        self.block_type = block_type
        self.att_type = att_type
        self.dropout = dropout

        self.init_hidden = cfg_Block(block_type, cfg_init, D_agent, 'MS', False, False)
        self.init_cell = cfg_Block(block_type, cfg_init, D_agent, 'MS', False, False)
        self.lstm = nn.LSTMCell(cfg_lstm[0], cfg_lstm[1])

        self.key = cfg_Block(block_type, cfg_enc, D_agent, 'MS', False, False)
        self.query = cfg_Block(block_type, cfg_enc, D_agent, 'MS', False, False)
        self.value = cfg_Block(block_type, cfg_enc, D_agent, 'MS', False, False)
        #self.self = cfg_Block(block_type, cfg_self, D_agent, 'MS', False, False)

        if self.att_type == 'gat':
            self.att = cfg_Block(block_type, cfg_att, D_agent, 'MS', False, False)
        elif self.att_type == 'add':
            self.att1 = cfg_Block(block_type, [D_att, D_att], D_agent, 'MS', False, False)
            self.att2 = cfg_Block(block_type, [D_att, D_att], D_agent, 'MS', False, False)
            self.att3 = cfg_Block(block_type, [D_att, 1], D_agent, 'MS', False, False)
        elif self.att_type == 'mul':
            self.att = None
        else:
            print('NOT IMPLEMENTED : gat att block creation')

        self.dec_x = cfg_Block(block_type, cfg_dec, D_agent, 'MS', False, False)
        self.mu_dec_x = cfg_Block(block_type, cfg_mu, D_agent, 'MS', False, False)
        self.sig_dec_x = cfg_Block(block_type, cfg_sig, D_agent, 'MS', False, False)

        self.dec_y = cfg_Block(block_type, cfg_dec, D_agent, 'MS', False, False)
        self.mu_dec_y = cfg_Block(block_type, cfg_mu, D_agent, 'MS', False, False)
        self.sig_dec_y = cfg_Block(block_type, cfg_sig, D_agent, 'MS', False, False)

        self.dec_vx = cfg_Block(block_type, cfg_dec, D_agent, 'MS', False, False)
        self.mu_dec_vx = cfg_Block(block_type, cfg_mu, D_agent, 'MS', False, False)
        self.sig_dec_vx = cfg_Block(block_type, cfg_sig, D_agent, 'MS', False, False)

        self.dec_vy = cfg_Block(block_type, cfg_dec, D_agent, 'MS', False, False)
        self.mu_dec_vy = cfg_Block(block_type, cfg_mu, D_agent, 'MS', False, False)
        self.sig_dec_vy = cfg_Block(block_type, cfg_sig, D_agent, 'MS', False, False)

        if block_type == 'mlp':
            self.D_k = 1
            self.D_s = 1
        elif block_type == 'res':
            self.D_k = 1
            self.D_s = 1

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
            #s = self.self(b)

            p_list = [[v] for _ in range(self.D_att_num)]
            #p_list = [[s] for _ in range(self.D_att_num)]

            mask_const = 10000.
            mask = torch.zeros(x.shape[0], x.shape[1], x.shape[1]).to(x.device)
            mask = mask+torch.eye(x.shape[1], x.shape[1]).to(x.device)*mask_const
            mask = mask.unsqueeze(-1).repeat(1, 1, 1, self.D_att_num)
            #assert self.D_att_num == 4
            #head_dim = int(self.D_att / self.D_att_num)
            #assert head_dim * self.D_att_num == self.D_att

            if self.att_type == 'gat':
                kz = [k for _ in range(k.shape[1])]
                qz = [q for _ in range(q.shape[1])]
                z = torch.cat((torch.stack(kz, dim=-2), torch.stack(qz, dim=-3), R_const), -1)
                w = torch.sigmoid(torch.add(self.att(z).squeeze(-1), -mask))/(self.D_agent)
                #w = torch.tanh(torch.mul(self.att(z).squeeze(-1), 1-mask))/(self.D_agent)
                for i in range(self.D_att_num):
                    #p_list[i].append(torch.matmul(w[:,:,:,i], v[:, :, head_dim * i: head_dim * (i + 1)]))
                    p_list[i].append(torch.matmul(w[:,:,:,i], v))
            
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
            
            c_list = [[] for _ in range(self.D_att_num)] 
            for i in range(4):
                c_list[i] = torch.cat(p_list[i], dim=-1)

            d_x = self.dec_x(c_list[0])
            mu_x = self.mu_dec_x(d_x).squeeze()
            sig_x = F.softplus(self.sig_dec_x(d_x)).squeeze()

            d_y = self.dec_y(c_list[1])
            mu_y = self.mu_dec_y(d_y).squeeze()
            sig_y = F.softplus(self.sig_dec_y(d_y)).squeeze()

            d_vx = self.dec_vx(c_list[2])
            mu_vx = self.mu_dec_vx(d_vx).squeeze()
            sig_vx = F.softplus(self.sig_dec_vx(d_vx)).squeeze()

            d_vy = self.dec_vy(c_list[3])
            mu_vy = self.mu_dec_vy(d_vy).squeeze()
            sig_vy = F.softplus(self.sig_dec_vy(d_vy)).squeeze()

            return (mu_x, sig_x), (mu_y, sig_y), (mu_vx, sig_vx), (mu_vy, sig_vy), hidden.view(batch_num, agent_num, -1), cell.view(batch_num, agent_num, -1)
        else:
            return hidden.view(batch_num, agent_num, -1), cell.view(batch_num, agent_num, -1)

class Module_GAT_CS_split(nn.Module):
    def __init__(self, cfg_enc, cfg_att, cfg_dec, cfg_mu, cfg_sig, D_att, D_att_num, D_agent, block_type, att_type, dropout):
        super(Module_GAT_CS_split, self).__init__()

        self.D_att = D_att
        self.D_att_num = D_att_num
        self.D_agent = D_agent
        self.block_type = block_type
        self.att_type = att_type
        self.dropout = dropout
        self.agent_norm = self.D_agent
        self.reg_norm = 10.

        self.key = cfg_Block(block_type, cfg_enc, D_agent, 'MS', False, False)
        self.query = cfg_Block(block_type, cfg_enc, D_agent, 'MS', False, False)
        self.value = cfg_Block(block_type, cfg_enc, D_agent, 'MS', False, False)

        if self.att_type == 'gat':
            self.att = cfg_Block(block_type, cfg_att, D_agent, 'MS', False, False)
        elif self.att_type == 'add':
            self.att1 = cfg_Block(block_type, [D_att, D_att], D_agent, 'MS', False, False)
            self.att2 = cfg_Block(block_type, [D_att, D_att], D_agent, 'MS', False, False)
            self.att3 = cfg_Block(block_type, [D_att, 1], D_agent, 'MS', False, False)
        elif self.att_type == 'mul':
            self.att = None
        else:
            print('NOT IMPLEMENTED : gat att block creation')

        self.dec_x = cfg_Block(block_type, cfg_dec, D_agent, 'MS', False, False)
        self.mu_dec_x = cfg_Block(block_type, cfg_mu, D_agent, 'MS', False, False)
        self.sig_dec_x = cfg_Block(block_type, cfg_sig, D_agent, 'MS', False, False)

        self.dec_y = cfg_Block(block_type, cfg_dec, D_agent, 'MS', False, False)
        self.mu_dec_y = cfg_Block(block_type, cfg_mu, D_agent, 'MS', False, False)
        self.sig_dec_y = cfg_Block(block_type, cfg_sig, D_agent, 'MS', False, False)

        self.dec_z = cfg_Block(block_type, cfg_dec, D_agent, 'MS', False, False)
        self.mu_dec_z = cfg_Block(block_type, cfg_mu, D_agent, 'MS', False, False)
        self.sig_dec_z = cfg_Block(block_type, cfg_sig, D_agent, 'MS', False, False)

        self.dec_vx = cfg_Block(block_type, cfg_dec, D_agent, 'MS', False, False)
        self.mu_dec_vx = cfg_Block(block_type, cfg_mu, D_agent, 'MS', False, False)
        self.sig_dec_vx = cfg_Block(block_type, cfg_sig, D_agent, 'MS', False, False)

        self.dec_vy = cfg_Block(block_type, cfg_dec, D_agent, 'MS', False, False)
        self.mu_dec_vy = cfg_Block(block_type, cfg_mu, D_agent, 'MS', False, False)
        self.sig_dec_vy = cfg_Block(block_type, cfg_sig, D_agent, 'MS', False, False)

        self.dec_vz = cfg_Block(block_type, cfg_dec, D_agent, 'MS', False, False)
        self.mu_dec_vz = cfg_Block(block_type, cfg_mu, D_agent, 'MS', False, False)
        self.sig_dec_vz = cfg_Block(block_type, cfg_sig, D_agent, 'MS', False, False)

        if block_type == 'mlp':
            self.D_k = 1
            self.D_s = 1
        elif block_type == 'res':
            self.D_k = 1
            self.D_s = 1

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if type(m.bias) != type(None):
                    m.bias.data.zero_()

    def forward(self, x):
        #print(x.shape, hidden[0].shape, hidden[1].shape)
        softmax = nn.Softmax(dim=-1)
        k = self.key(x) 
        q = self.query(x)
        v = self.value(x)
            
        p_list = [[v] for _ in range(self.D_att_num)]

        mask_const = 10000.
        mask = torch.zeros(x.shape[0], x.shape[1], x.shape[1]).to(x.device)
        mask = mask+torch.eye(x.shape[1], x.shape[1]).to(x.device)*mask_const
        mask = mask.unsqueeze(-1).repeat(1, 1, 1, self.D_att_num)

        if self.att_type == 'gat':
            kz = [k for _ in range(k.shape[1])]
            qz = [q for _ in range(q.shape[1])]
            z = torch.cat((torch.stack(kz, dim=-2), torch.stack(qz, dim=-3)), -1)
            w = torch.sigmoid(torch.add(self.att(z).squeeze(-1), -mask)) / (self.D_agent / self.agent_norm)
            #w = torch.tanh(torch.mul(self.att(z).squeeze(-1), 1-mask))/(self.D_agent)
            for i in range(self.D_att_num):
                #p_list[i].append(torch.matmul(w[:,:,:,i], v[:, :, head_dim * i: head_dim * (i + 1)]))
                p_list[i].append(torch.matmul(w[:,:,:,i], v))
        
        epsilon = 1e-6
        
        d_x = self.dec_x(torch.cat(p_list[0], dim=-1))
        mu_x = self.mu_dec_x(d_x).squeeze()
        sig_x = torch.sigmoid(self.sig_dec_x(d_x)).squeeze() + epsilon / self.reg_norm

        d_y = self.dec_y(torch.cat(p_list[1], dim=-1))
        mu_y = self.mu_dec_y(d_y).squeeze()
        sig_y = torch.sigmoid(self.sig_dec_y(d_y)).squeeze() + epsilon / self.reg_norm

        d_z = self.dec_z(torch.cat(p_list[2], dim=-1))
        mu_z = self.mu_dec_z(d_z).squeeze()
        sig_z = torch.sigmoid(self.sig_dec_z(d_z)).squeeze() + epsilon / self.reg_norm

        d_vx = self.dec_vx(torch.cat(p_list[3], dim=-1))
        mu_vx = self.mu_dec_vx(d_vx).squeeze()
        sig_vx = torch.sigmoid(self.sig_dec_vx(d_vx)).squeeze() + epsilon / self.reg_norm

        d_vy = self.dec_vy(torch.cat(p_list[4], dim=-1))
        mu_vy = self.mu_dec_vy(d_vy).squeeze()
        sig_vy = torch.sigmoid(self.sig_dec_vy(d_vy)).squeeze() + epsilon / self.reg_norm

        d_vz = self.dec_vz(torch.cat(p_list[5], dim=-1))
        mu_vz = self.mu_dec_vz(d_vz).squeeze()
        sig_vz = torch.sigmoid(self.sig_dec_vz(d_vz)).squeeze() + epsilon / self.reg_norm

        return (mu_x, sig_x), (mu_y, sig_y), (mu_z, sig_z), (mu_vx, sig_vx), (mu_vy, sig_vy), (mu_vz, sig_vz)

    def __init__(self, cfg_init, cfg_lstm, cfg_enc, cfg_att, cfg_dec, cfg_mu, cfg_sig, cfg_corr, cfg_coef, D_att, D_att_num, D_agent, block_type, att_type, dropout, eval_type):
        super(Module_GAT_LSTM_ns, self).__init__()

        self.D_att = D_att
        self.D_att_num = D_att_num
        self.D_agent = D_agent
        self.block_type = block_type
        self.att_type = att_type
        self.dropout = dropout
        self.eval_type = eval_type
        self.agent_norm = 1.

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

    def forward(self, x, hidden, verbose=False):
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
                w = torch.sigmoid(torch.add(self.att(z).squeeze(-1), -mask))
                wz = torch.bmm(w, v[:, :, self.D_att * i: self.D_att * (i + 1)])
                p_list.append(wz/ (self.D_agent / self.agent_norm))  # Normalizing the scale
                #if verbose:
                #    print(self.D_agent/10)
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

class Module_GAT_LSTM_split(nn.Module):
    def __init__(self, cfg_init, cfg_lstm, cfg_enc, cfg_self, cfg_att, cfg_dec, cfg_mu, cfg_sig, D_att, D_att_num, D_agent, block_type, att_type, dropout, eval_type, sig=True, use_sample=True):
        super(Module_GAT_LSTM_split, self).__init__()

        self.D_att = D_att
        self.D_att_num = D_att_num
        self.D_agent = D_agent
        self.block_type = block_type
        self.att_type = att_type
        self.dropout = dropout
        self.eval_type = eval_type
        self.agent_norm = self.D_agent
        self.mu_norm = 1.
        self.sig_norm = 1.
        self.sig = sig
        self.use_sample = use_sample

        self.init_hidden = cfg_Block(block_type, cfg_init, D_agent, 'MS', False, False)
        self.init_cell = cfg_Block(block_type, cfg_init, D_agent, 'RL', False, False)
        self.lstm = nn.LSTMCell(cfg_lstm[0], cfg_lstm[1])

        self.key = cfg_Block(block_type, cfg_enc, D_agent, 'MS', False, False)
        self.query = cfg_Block(block_type, cfg_enc, D_agent, 'MS', False, False)
        self.value = cfg_Block(block_type, cfg_enc, D_agent, 'MS', False, False)
        #self.self = cfg_Block(block_type, cfg_self, D_agent, 'MS', False, False)

        if self.att_type == 'gat':
            self.att = cfg_Block(block_type, cfg_att, D_agent, 'TH', False, False)
        elif self.att_type == 'kqv':
            self.att = None
        else:
            print('NOT IMPLEMENTED : gat att block creation')

        self.dec_x = cfg_Block(block_type, cfg_dec, D_agent, 'MS', False, False)
        self.mu_dec_x = cfg_Block(block_type, cfg_mu, D_agent, 'MS', False, False)
        if self.sig:
            self.sig_dec_x = cfg_Block(block_type, cfg_sig, D_agent, 'MS', False, False)

        self.dec_y = cfg_Block(block_type, cfg_dec, D_agent, 'MS', False, False)
        self.mu_dec_y = cfg_Block(block_type, cfg_mu, D_agent, 'MS', False, False)
        if self.sig:
            self.sig_dec_y = cfg_Block(block_type, cfg_sig, D_agent, 'MS', False, False)

        self.dec_z = cfg_Block(block_type, cfg_dec, D_agent, 'MS', False, False)
        self.mu_dec_z = cfg_Block(block_type, cfg_mu, D_agent, 'MS', False, False)
        if self.sig:
            self.sig_dec_z = cfg_Block(block_type, cfg_sig, D_agent, 'MS', False, False)

        self.dec_vx = cfg_Block(block_type, cfg_dec, D_agent, 'MS', False, False)
        self.mu_dec_vx = cfg_Block(block_type, cfg_mu, D_agent, 'MS', False, False)
        if self.sig:
            self.sig_dec_vx = cfg_Block(block_type, cfg_sig, D_agent, 'MS', False, False)

        self.dec_vy = cfg_Block(block_type, cfg_dec, D_agent, 'MS', False, False)
        self.mu_dec_vy = cfg_Block(block_type, cfg_mu, D_agent, 'MS', False, False)
        if self.sig: 
            self.sig_dec_vy = cfg_Block(block_type, cfg_sig, D_agent, 'MS', False, False)

        self.dec_vz = cfg_Block(block_type, cfg_dec, D_agent, 'MS', False, False)
        self.mu_dec_vz = cfg_Block(block_type, cfg_mu, D_agent, 'MS', False, False)
        if self.sig:
            self.sig_dec_vz = cfg_Block(block_type, cfg_sig, D_agent, 'MS', False, False)

        if block_type == 'mlp':
            self.D_k = 1
            self.D_s = 1
        elif block_type == 'res':
            self.D_k = 1
            self.D_s = 1

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if type(m.bias) != type(None):
                    m.bias.data.zero_()

    def initialize(self, x, hidden = None, cell = None, init_mask = None):
        batch_num = x.shape[0]
        agent_num = x.shape[1]
        x = x.reshape(batch_num * agent_num, -1)
        init_mask = init_mask.flatten().unsqueeze(-1).float()
        h = self.init_hidden(x)
        c = self.init_cell(x)
        
        if hidden is None:
            hidden, cell = (torch.zeros_like(h), torch.zeros_like(c))
        else:
            hidden, cell = hidden.reshape(batch_num * agent_num, -1), cell.reshape(batch_num * agent_num, -1)
        #print(x.shape, init_mask.shape, hidden.shape)
        return (hidden * (1 - init_mask) + h * init_mask), (cell * (1 - init_mask) + c * init_mask)

    def forward(self, x, hidden, cell, verbose=False):
        softmax = nn.Softmax(dim=-1)

        batch_num = x.shape[0]
        agent_num = x.shape[1]
        x = x.view(batch_num * agent_num, -1)

        hidden, cell = self.lstm(x, (hidden, cell))
        b = hidden.view(batch_num, agent_num, -1)
        x = x.view(batch_num, agent_num, -1)

        k = self.key(b) 
        q = self.query(b)
        v = self.value(b)
        #s = self.self(b)

        #p_list = [[s] for _ in range(self.D_att_num)]
        p_list = [[v] for _ in range(self.D_att_num)]

        mask_const = 10000
        mask = torch.zeros(x.shape[0], x.shape[1], x.shape[1]).to(x.device)
        mask = mask + torch.eye(x.shape[1], x.shape[1]).to(x.device) * mask_const
        #assert self.D_att_num == 6
        #head_dim = int(self.D_att / self.D_att_num)
        #assert head_dim * self.D_att_num == self.D_att

        for i in range(x.shape[0]):
            m = (x[i, :, -1] != 1).nonzero().squeeze()
            if m.dim() != 1:
                print(x[i, :, -1])
                print(m)
                print('wrong')  #
            for j in m:
                mask[i][j, :] = mask_const
                mask[i][:, j] = mask_const

        #mask.masked_fill_(mask == 1, -np.inf)
        mask = mask.unsqueeze(-1).repeat(1, 1, 1, self.D_att_num)

        if self.att_type == 'gat':
            kz = [k for _ in range(k.shape[1])]
            qz = [q for _ in range(q.shape[1])]
            z = torch.cat((torch.stack(kz, dim=-2), torch.stack(qz, dim=-3)), -1)
            w = torch.sigmoid(torch.add(self.att(z).squeeze(-1), -mask))
            for i in range(self.D_att_num):
                p_list[i].append(torch.matmul(w[:,:,:,i], v))

        elif self.att_type == 'kqv':
            print('NOT IMPLEMENTED')
            for i in range(self.D_att_num):
                ky = k[:, :, self.D_att * i: self.D_att * (i + 1)]
                kz = torch.stack([ky for _ in range(ky.shape[1])], dim = -2)
                qy = q[:, :, self.D_att * i: self.D_att * (i + 1)]
                qz = torch.stack([qy for _ in range(qy.shape[1])], dim = -3)
                w = F.dropout(softmax(torch.add(batchedDot(kz, qz)/np.sqrt(self.D_att), -mask)), p=self.dropout, training=self.training)
                p_list.append(torch.bmm(w, v[:, :, self.D_att * i: self.D_att * (i + 1)]))
        else:
            print('NOT IMPLEMENTED : GAT attention type')
          
        '''
        c_list = [[] for _ in range(self.D_att_num)] 
        for i in range(self.D_att_num):
            c_list[i] = torch.cat(p_list[i], dim=-1)
        '''

        epsilon = 1e-6
        fixed_var = 5e-4
        d_x = self.dec_x(torch.cat(p_list[0], dim=-1))
        mu_x = (self.mu_dec_x(d_x).squeeze()) / self.mu_norm
        if self.sig:
            sig_x = (torch.sigmoid(self.sig_dec_x(d_x)).squeeze() + epsilon) / self.sig_norm 

        d_y = self.dec_y(torch.cat(p_list[1], dim=-1))
        mu_y = (self.mu_dec_y(d_y).squeeze()) / self.mu_norm
        if self.sig:
            sig_y = (torch.sigmoid(self.sig_dec_y(d_y)).squeeze() + epsilon) / self.sig_norm

        d_z = self.dec_z(torch.cat(p_list[2], dim=-1))
        mu_z = (self.mu_dec_z(d_z).squeeze()) / self.mu_norm
        if self.sig:
            sig_z = (torch.sigmoid(self.sig_dec_z(d_z)).squeeze() + epsilon) / self.sig_norm

        d_vx = self.dec_vx(torch.cat(p_list[3], dim=-1))
        mu_vx =(self.mu_dec_vx(d_vx).squeeze()) / self.mu_norm
        if self.sig:
            sig_vx = (torch.sigmoid(self.sig_dec_vx(d_vx)).squeeze() + epsilon) / self.sig_norm

        d_vy = self.dec_vy(torch.cat(p_list[4], dim=-1))
        mu_vy = (self.mu_dec_vy(d_vy).squeeze()) / self.mu_norm
        if self.sig:
            sig_vy = (torch.sigmoid(self.sig_dec_vy(d_vy)).squeeze() + epsilon) / self.sig_norm

        d_vz = self.dec_vz(torch.cat(p_list[5], dim=-1))
        mu_vz = (self.mu_dec_vz(d_vz).squeeze()) / self.mu_norm
        if self.sig:
            sig_vz = (torch.sigmoid(self.sig_dec_vz(d_vz)).squeeze() + epsilon) / self.sig_norm

        sig = torch.ones_like(mu_x) * fixed_var
        if self.sig:
            return (mu_x, sig_x), (mu_y, sig_y), (mu_z, sig_z), (mu_vx, sig_vx), (mu_vy, sig_vy), (mu_vz, sig_vz), hidden.view(batch_num, agent_num, -1), cell.view(batch_num, agent_num, -1)
        else:
            return (mu_x, sig), (mu_y, sig), (mu_z, sig), (mu_vx, sig), (mu_vy, sig), (mu_vz, sig), hidden.view(batch_num, agent_num, -1), cell.view(batch_num, agent_num, -1)

class Module_MLP_LSTM_split(nn.Module):
    def __init__(self, cfg_init, cfg_lstm, cfg_dec, cfg_mu, cfg_sig, D_agent, block_type, eval_type, sig=True, use_sample=True):
        super(Module_MLP_LSTM_split, self).__init__()

        self.D_agent = D_agent
        self.block_type = block_type
        self.dropout = 0.0
        self.eval_type = eval_type
        self.agent_norm = self.D_agent
        self.mu_norm = 10.
        self.sig_norm = 10.
        self.sig = sig
        self.use_sample = use_sample

        self.init_hidden = cfg_Block(block_type, cfg_init, D_agent, 'MS', False, False)
        self.init_cell = cfg_Block(block_type, cfg_init, D_agent, 'RL', False, False)
        self.lstm = nn.LSTMCell(cfg_lstm[0], cfg_lstm[1])

        self.dec_x = cfg_Block(block_type, cfg_dec, D_agent, 'MS', False, False)
        self.mu_dec_x = cfg_Block(block_type, cfg_mu, D_agent, 'MS', False, False)
        if self.sig:
            self.sig_dec_x = cfg_Block(block_type, cfg_sig, D_agent, 'MS', False, False)

        self.dec_y = cfg_Block(block_type, cfg_dec, D_agent, 'MS', False, False)
        self.mu_dec_y = cfg_Block(block_type, cfg_mu, D_agent, 'MS', False, False)
        if self.sig:
            self.sig_dec_y = cfg_Block(block_type, cfg_sig, D_agent, 'MS', False, False)

        self.dec_z = cfg_Block(block_type, cfg_dec, D_agent, 'MS', False, False)
        self.mu_dec_z = cfg_Block(block_type, cfg_mu, D_agent, 'MS', False, False)
        if self.sig:
            self.sig_dec_z = cfg_Block(block_type, cfg_sig, D_agent, 'MS', False, False)

        self.dec_vx = cfg_Block(block_type, cfg_dec, D_agent, 'MS', False, False)
        self.mu_dec_vx = cfg_Block(block_type, cfg_mu, D_agent, 'MS', False, False)
        if self.sig:
            self.sig_dec_vx = cfg_Block(block_type, cfg_sig, D_agent, 'MS', False, False)

        self.dec_vy = cfg_Block(block_type, cfg_dec, D_agent, 'MS', False, False)
        self.mu_dec_vy = cfg_Block(block_type, cfg_mu, D_agent, 'MS', False, False)
        if self.sig: 
            self.sig_dec_vy = cfg_Block(block_type, cfg_sig, D_agent, 'MS', False, False)

        self.dec_vz = cfg_Block(block_type, cfg_dec, D_agent, 'MS', False, False)
        self.mu_dec_vz = cfg_Block(block_type, cfg_mu, D_agent, 'MS', False, False)
        if self.sig:
            self.sig_dec_vz = cfg_Block(block_type, cfg_sig, D_agent, 'MS', False, False)

        if block_type == 'mlp':
            self.D_k = 1
            self.D_s = 1
        elif block_type == 'res':
            self.D_k = 1
            self.D_s = 1

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if type(m.bias) != type(None):
                    m.bias.data.zero_()

    def initialize(self, x, hidden = None, cell = None, init_mask = None):
        batch_num = x.shape[0]
        agent_num = x.shape[1]
        x = x.reshape(batch_num * agent_num, -1)
        init_mask = init_mask.flatten().unsqueeze(-1).float()
        h = self.init_hidden(x)
        c = self.init_cell(x)
        
        if hidden is None:
            hidden, cell = (torch.zeros_like(h), torch.zeros_like(c))
        else:
            hidden, cell = hidden.reshape(batch_num * agent_num, -1), cell.reshape(batch_num * agent_num, -1)
        #print(x.shape, init_mask.shape, hidden.shape)
        return (hidden * (1 - init_mask) + h * init_mask), (cell * (1 - init_mask) + c * init_mask)

    def forward(self, x, hidden, cell, verbose=False):
        softmax = nn.Softmax(dim=-1)

        batch_num = x.shape[0]
        agent_num = x.shape[1]
        x = x.view(batch_num * agent_num, -1)

        hidden, cell = self.lstm(x, (hidden, cell))
        b = hidden.view(batch_num, agent_num, -1)

        epsilon = 1e-6
        fixed_var = 5e-3
        d_x = self.dec_x(b)
        mu_x = (self.mu_dec_x(d_x).squeeze()) / self.mu_norm
        if self.sig:
            sig_x = (torch.sigmoid(self.sig_dec_x(d_x)).squeeze() + epsilon) / self.sig_norm 

        d_y = self.dec_y(b)
        mu_y = (self.mu_dec_y(d_y).squeeze()) / self.mu_norm
        if self.sig:
            sig_y = (torch.sigmoid(self.sig_dec_y(d_y)).squeeze() + epsilon) / self.sig_norm

        d_z = self.dec_z(b)
        mu_z = (self.mu_dec_z(d_z).squeeze()) / self.mu_norm
        if self.sig:
            sig_z = (torch.sigmoid(self.sig_dec_z(d_z)).squeeze() + epsilon) / self.sig_norm

        d_vx = self.dec_vx(b)
        mu_vx = (self.mu_dec_vx(d_vx).squeeze()) / self.mu_norm
        if self.sig:
            sig_vx = (torch.sigmoid(self.sig_dec_vx(d_vx)).squeeze() + epsilon) / self.sig_norm

        d_vy = self.dec_vy(b)
        mu_vy = (self.mu_dec_vy(d_vy).squeeze()) / self.mu_norm
        if self.sig:
            sig_vy = (torch.sigmoid(self.sig_dec_vy(d_vy)).squeeze() + epsilon) / self.sig_norm

        d_vz = self.dec_vz(b)
        mu_vz = (self.mu_dec_vz(d_vz).squeeze()) / self.mu_norm
        if self.sig:
            sig_vz = (torch.sigmoid(self.sig_dec_vz(d_vz)).squeeze() + epsilon) / self.sig_norm

        sig = torch.ones_like(mu_x) * fixed_var
        if self.sig:
            return (mu_x, sig_x), (mu_y, sig_y), (mu_z, sig_z), (mu_vx, sig_vx), (mu_vy, sig_vy), (mu_vz, sig_vz), hidden.view(batch_num, agent_num, -1), cell.view(batch_num, agent_num, -1)
        else:
            return (mu_x, sig), (mu_y, sig), (mu_z, sig), (mu_vx, sig), (mu_vy, sig), (mu_vz, sig), hidden.view(batch_num, agent_num, -1), cell.view(batch_num, agent_num, -1)
