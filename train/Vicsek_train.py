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
from torch.utils.data import DataLoader # (testset, batch_size=4,shuffle=False, num_workers=4)
from torch.optim.lr_scheduler import ReduceLROnPlateau as RLRP
from torch.nn.init import xavier_normal
from torch.nn.parameter import Parameter

import pickle
import importlib
import itertools
import random
from collections import OrderedDict
from copy import deepcopy

import src.DataStructure as DS
from src.utils import *
from src.system import *
from src.model import *

from utils.slurm import init_distributed_mode
## Methods

parser = argparse.ArgumentParser(description='Pytorch VAINS Training')

parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=2, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=5e-4, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=0.0005, type=float,
                    metavar='W', help='weight decay (default: 0)',
                    dest='weight_decay')

parser.add_argument("--debug_slurm", action='store_false', default=True,
                    help="Debug multi-GPU / multi-node within a SLURM job")

parser.add_argument('--distributed', action='store_true', default=True,
                    help='slurm distributed system enable'
                         'multi node data parallel training')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--local_rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument("--master_port", type=int, default=-1,
                    help="Master port (for multi-node SLURM jobs)")
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')

parser.add_argument('--model-type', default = 'vains', type = str, help='model type : vains, mlp')
parser.add_argument('--block-type', default = 'mlp', type = str, help='mlp : simple multi-layer perceptron, res : skip-connection')
parser.add_argument('--att-type', default = 'single', type = str, help = 'gat : attention layer, kqv : dot product ')
parser.add_argument('--att-dim', default = '50', type = int, help = 'Dimension of attention vector')
parser.add_argument('--att-num', default = 1, type = int, help = 'For "multi", works as number of heads.')
parser.add_argument('--mode-num', type = int, help = 'Number of gaussian mixture mode.')
parser.add_argument('--dropout', default = 0.0, type = float, help = 'Rate of dropout on attention.')
parser.add_argument('--indicator', default = '', type = str, help = 'Additional specification for file name.')

parser.add_argument('--agent_num', type = int)
parser.add_argument('--neighbor_dist', type = float)
parser.add_argument('--neighbor_angle', type = int)
parser.add_argument('--noise_type', type = str)
parser.add_argument('--noise_strength', type = float)

parser.add_argument("--seed", default=42, type=int, help="Random seed for torch and numpy")


def main():
    best_test_loss = 100
    args = parser.parse_args()
    print(torch.cuda.device_count(), args.local_rank)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.set_device(args.local_rank)
    if args.distributed:
        init_distributed_mode(args)
    else:
        args.local_rank = 0

    pp = {}
    system = Vicsek_markovian()
    #sig_num = int(args.indicator[3:])
    #system = Vicsek_sigmoid(sig_num)
    # system = Vicsek_linear()

    pp['agent_num'] = args.agent_num
    pp['neighbor_dist'] = args.neighbor_dist
    pp['neighbor_angle'] = args.neighbor_angle
    pp['noise_type'] = args.noise_type
    pp['noise_strength'] = args.noise_strength
    pp['grp_num'] = 1
    system.assign_pp(pp)
    system.state_num = 4
    system.answer_num = 1

    # Data loading code
    file_name = system.rule_name   
    indicator = '_MT' + str(args.model_type) + '_Dk' + str(args.mode_num) + '_BT' + args.block_type + '_DO' + str(args.dropout)+  '_' + args.indicator

    if args.local_rank == 0:
        print(file_name + indicator)

    train_set = DS.Customset('./data/Vicsek/'+file_name, train=True)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers, sampler = train_sampler)
    test_set = DS.Customset('./data/Vicsek/'+file_name, train=False)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_set)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers, sampler = test_sampler)

    train_set.train_data = train_set.train_data[:, :, :-1]
    test_set.test_data = test_set.test_data[:, :, :-1]

    if args.model_type == 'vains':

        D_in_enc = system.state_num  # x, y, z / vx. vy, vz / hdg , hdg_rate, V/ group
        D_hidden_enc = 256
        D_out_enc = 256
        D_att = args.att_dim
        D_att_num = args.att_num
        D_in_dec = D_out_enc * (D_att_num + 1)
        D_hidden_dec = 256
        D_out_dec = 128
        D_hidden_stat = 128

        D_agent = system.agent_num
        D_s = system.answer_num
        D_k = args.mode_num

        if args.block_type == 'mlp':
            cfg_enc = [D_in_enc, D_hidden_enc, D_hidden_enc, D_hidden_enc, D_out_enc]
            cfg_com = [D_in_enc, D_hidden_enc, D_hidden_enc, D_hidden_enc, D_out_enc]
            cfg_att = [D_in_enc, D_hidden_enc, D_hidden_enc, D_hidden_enc, D_att*D_att_num]
            cfg_dec = [D_in_dec, D_hidden_dec, D_hidden_dec, D_hidden_dec, D_out_dec]

        elif args.block_type == 'res':

            cfg_enc = [D_in_enc, D_hidden_enc,  D_hidden_enc, D_out_enc]
            cfg_com = [D_in_enc, D_hidden_enc,D_hidden_enc, D_out_enc]
            cfg_att = [D_in_enc, D_hidden_enc, D_hidden_enc, D_att*D_att_num]
            cfg_dec = [D_in_dec, D_hidden_dec,  D_hidden_dec, D_out_dec]

        else:
            print('NOT IMPLEMENTED : cfg block type')

        if D_s == 2:
            cfg_mu = [D_out_dec, D_hidden_stat, 2 * D_k]
            cfg_sig = [D_out_dec, D_hidden_stat, 2 * D_k]
            cfg_corr = [D_out_dec, D_hidden_stat, D_k]
            cfg_coef = [D_out_dec, D_hidden_stat, D_k]
        elif D_s == 3:
            cfg_mu = [D_out_dec, D_hidden_stat, 3 * D_k]
            cfg_sig = [D_out_dec, D_hidden_stat, 3 * D_k]
            cfg_corr = [D_out_dec, D_hidden_stat, 3 * D_k]
            cfg_coef = [D_out_dec, D_hidden_stat, D_k]
        elif D_s == 6:
            cfg_mu = [D_out_dec, D_hidden_stat, 6 * D_k]
            cfg_sig = [D_out_dec, D_hidden_stat, 6 * D_k]
            cfg_corr = [D_out_dec, D_hidden_stat, 15 * D_k]
            cfg_coef = [D_out_dec, D_hidden_stat, D_k]
        else:
            print("NOT  : D_s assigning")

        model = Module_VAINS(cfg_enc, cfg_com, cfg_att, cfg_dec, cfg_mu, cfg_sig, cfg_corr, cfg_coef, D_att, D_att_num,
                             D_agent, args.block_type, args.att_type, args.dropout).cuda()
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)

    elif args.model_type == 'mlp':

        D_in_dec = system.state_num  # x, y, z / vx. vy, vz / hdg , hdg_rate, V/ group
        D_hidden_dec = 1024
        D_out_dec = 128
        D_hidden_stat = 128

        D_agent = system.agent_num
        D_s = system.answer_num
        D_k = args.mode_num

        if args.block_type == 'mlp':
            cfg_dec = [D_in_dec, D_hidden_dec, D_hidden_dec, D_hidden_dec, D_hidden_dec, D_hidden_dec, D_out_dec]

        elif args.block_type == 'res':
            cfg_dec = [D_in_dec, D_hidden_dec, D_hidden_dec, D_hidden_dec, D_out_dec]

        else:
            print('NOT IMPLEMENTED : cfg block type')

        if D_s == 2:
            cfg_mu = [D_out_dec, D_hidden_stat, 2 * D_k]
            cfg_sig = [D_out_dec, D_hidden_stat, 2 * D_k]
            cfg_corr = [D_out_dec, D_hidden_stat, D_k]
            cfg_coef = [D_out_dec, D_hidden_stat, D_k]
        elif D_s == 3:
            cfg_mu = [D_out_dec, D_hidden_stat, 3 * D_k]
            cfg_sig = [D_out_dec, D_hidden_stat, 3 * D_k]
            cfg_corr = [D_out_dec, D_hidden_stat, 3 * D_k]
            cfg_coef = [D_out_dec, D_hidden_stat, D_k]
        elif D_s == 6:
            cfg_mu = [D_out_dec, D_hidden_stat, 6 * D_k]
            cfg_sig = [D_out_dec, D_hidden_stat, 6 * D_k]
            cfg_corr = [D_out_dec, D_hidden_stat, 15 * D_k]
            cfg_coef = [D_out_dec, D_hidden_stat, D_k]

        model = Module_MLP(cfg_dec, cfg_mu, cfg_sig, cfg_corr, cfg_coef, D_agent, args.block_type).cuda()
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    elif args.model_type == 'gat':

        D_s = system.answer_num

        D_in_enc = system.state_num
        D_hidden_enc = 128
        D_out_enc = args.att_dim
  
        D_att = args.att_dim
        D_att_num = args.att_num

        D_in_dec = D_att * 2
        D_hidden_dec = 128
        D_out_dec = 128
        D_hidden_stat = 64

        D_agent = system.agent_num
        D_k = args.mode_num

        # cfg construction

        cfg_enc = [D_in_enc, D_hidden_enc, D_out_enc]
        cfg_att = [D_att * 2, 64, 32, D_att_num]
        cfg_dec = [D_in_dec, D_hidden_dec, D_out_dec]

        cfg_mu = [D_out_dec, D_hidden_stat, D_k]
        cfg_sig = [D_out_dec, D_hidden_stat, D_k]

        model = Module_GAT_VC_split(
            cfg_enc,
            cfg_att,
            cfg_dec,
            cfg_mu,
            cfg_sig,
            D_att,
            D_att_num,
            D_agent,
            args.block_type,
            args.att_type,
            args.dropout,
        ).cuda()
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank, find_unused_parameters=True)

    else:
        print('NOT IMPLEMENTED : model type')

    # define loss function (criterion) and optimizer
    criterion = gmm_criterion(D_s)
    optimizer = torch.optim.AdamW(
        model.parameters(), args.lr, weight_decay=args.weight_decay
    )
    # optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum = 0.9, weight_decay = args.weight_decay)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 100, 2, eta_min=0)
    #optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay = args.weight_decay)
    #scheduler = RLRP(optimizer, 'min', factor=0.5, patience=30, min_lr=5e-6, verbose=1)
    cudnn.benchmark = True

    with torch.autograd.detect_anomaly():
        for epoch in range(args.start_epoch, args.epochs):
            train_sampler.set_epoch(epoch)

            # train for one epoch
            train_loss = train(train_loader, model, criterion, optimizer, epoch, scheduler, args)
            scheduler.step()
            # evaluate on validation set
            test_loss = test(test_loader, model, criterion, args)
            if args.local_rank == 0:
                print("Epoch {} : Train Loss {} , Test Loss {}".format(str(epoch), str(train_loss), str(test_loss)))

            # remember best acc@1 and save checkpoint
            is_best = test_loss < best_test_loss
            best_test_loss = min(test_loss, best_test_loss)

            if args.local_rank == 0 and is_best:
                torch.save(model.module.state_dict(), file_name + '_' + indicator +'.pth')

def train(train_loader, model, criterion, optimizer, epoch, scheduler, args):

    train_losses = AverageMeter('Loss', ':.4e')
    model.train()

    for data, labels in train_loader:

        data = data.cuda(args.local_rank)
        labels = labels.cuda(args.local_rank).squeeze()
        label_diff = (labels[:,:,1:] - data[:,:,:2])
        (mu_x, sig_x), (mu_y, sig_y) = model(data)
        #print(label_diff.shape, mu_x.shape, sig_x.shape)
        #print(labels.shape, mu.shape)
        nll1 = criterion(label_diff[:, :, 0], mu_x, sig_x, None)
        nll2 = criterion(label_diff[:, :, 1], mu_y, sig_y, None)
        nll = nll1+nll2

        train_loss = torch.sum(nll * labels[:, :, 0]) / int(torch.sum(labels[:,:,0]))
        train_losses.update(train_loss.item(), int(torch.sum(labels[:,:,0])))
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step() 

    #scheduler.step(train_losses.avg, epoch)

    return train_losses.avg


def test(test_loader, model, criterion, args):

    test_losses = AverageMeter('Loss', ':.4e')
    model.eval()

    with torch.no_grad():
        for data, labels in test_loader:

            data = data.cuda(args.local_rank)
            labels = labels.cuda(args.local_rank).squeeze()
            label_diff = (labels[:,:,1:] - data[:,:,:2])
            (mu_x, sig_x), (mu_y, sig_y) = model(data)
            
            nll1 = criterion(label_diff[:, :, 0], mu_x, sig_x, None)
            nll2 = criterion(label_diff[:, :, 1], mu_y, sig_y, None)
            nll = nll1+nll2

            test_loss = torch.sum(nll * labels[:, :, 0]) / int(torch.sum(labels[:,:,0]))
            test_losses.update(test_loss.item(), int(torch.sum(labels[:,:,0])))

    return test_losses.avg

if __name__ == '__main__':
    print('started!') # For test
    main()

