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
import math

from time import sleep

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed as dist
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
from datetime import datetime
from collections import OrderedDict
from copy import deepcopy

import src.DataStructure as DS
from src.utils import *
from src.system import *
from src.model import *

parser = argparse.ArgumentParser(description='Pytorch VAINS Training')

parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
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
parser.add_argument('--wd', '--weight-decay', default=0.005, type=float,
                    metavar='W', help='weight decay (default: 0)',
                    dest='weight_decay')

parser.add_argument('--world-size', default=4, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='env://', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')

parser.add_argument('--local_rank', type=int)
parser.add_argument('--model-type', default = 'vains', type = str, help='model type : vains, mlp')
parser.add_argument('--block-type', default = 'mlp', type = str, help='mlp : simple multi-layer perceptron, res : skip-connection')
parser.add_argument('--att-type', default = 'single', type = str, help = 'single : single attention, multi : multi-head attention, distort : Tanh distortion layer')
parser.add_argument('--att-dim', default = '10', type = int, help = 'Dimension of attention vector')
parser.add_argument('--att-num', default = 1, type = int, help = 'For "multi", works as number of heads.')
parser.add_argument('--mode-num', type = int, help = 'Number of gaussian mixture mode.')
parser.add_argument('--dropout', default = 0.0, type = float, help = 'Rate of dropout on attention.')
parser.add_argument('--indicator', default = '', type = str, help = 'Additional specification for file name.')
parser.add_argument("--seed", default=42, type=int, help="Random seed for torch and numpy")


def main():
    best_test_loss = 100
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print(torch.cuda.device_count(), args.local_rank)
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl',
                                         init_method='env://')

    # Plugin parameters
    pp = {}
    system = CA()

    pp['side_length'] = 14 
    pp['rule']={'alive' : [2, 3], 'dead' : [3]}
    system.assign_pp(pp)

    # Data loading code
    file_name = system.rule_name + '_L' + str(system.side_length-2)
    train_set = DS.Customset('./data/CA/' + file_name, train=True)
    test_set = DS.Customset('./data/CA/' + file_name, train=False)
    #train_set = DS.Customset('./data/' + file_name, train=True)
    #test_set = DS.Customset('./data/' + file_name, train=False)
    
    if args.local_rank == 0:
        print(file_name + args.indicator)

    train_sampler = torch.utils.data.DistributedSampler(train_set)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, pin_memory=True,
                            num_workers=args.workers, sampler=train_sampler)
    test_sampler = torch.utils.data.DistributedSampler(test_set)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, pin_memory=True,
                            num_workers=args.workers, sampler=test_sampler)

    D_in_enc = system.state_num  # x, y, c
    D_hidden_enc = 128
    D_out_enc = args.att_dim * args.att_num
    D_att = args.att_dim
    D_att_num = args.att_num
    D_in_dec = D_out_enc * (args.att_num+1)
    D_in_dec = D_out_enc * 2
    
    D_hidden_dec = 64
    D_out_dec = 1 # possibility of 1
    D_agent = system.input_length

    if args.block_type == 'mlp':

        cfg_enc = [D_in_enc, D_hidden_enc, D_hidden_enc, D_out_enc]
        cfg_att = [D_att*2, 8, 8, 1]
        cfg_dec = [D_in_dec, D_hidden_dec, D_hidden_dec, D_out_dec]

    elif args.block_type == 'res':

        cfg_enc = [D_in_enc, D_hidden_enc, D_out_enc]
        cfg_att = [D_att * 2, 8, 1]
        cfg_dec = [D_in_dec, D_hidden_dec, D_out_dec]

    else:
        print('NOT IMPLEMENTED : cfg block type')

    model = Module_GAT_DET(cfg_enc, cfg_att, cfg_dec, D_att, D_att_num, D_agent, args.block_type,  args.att_type, args.dropout).cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)

    # define loss function (criterion) and optimizer

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), args.lr, weight_decay=args.weight_decay
    )
    scheduler = RLRP(optimizer, 'min', factor=0.7, patience=50, min_lr=5e-6, verbose=1)
    cudnn.benchmark = True

    with torch.autograd.detect_anomaly():
        for epoch in range(args.start_epoch, args.epochs):
            #train_sampler.set_epoch(epoch)

            # train for one epoch
            if args.local_rank==0:
                print("============== Epoch {} =============".format(epoch))
            train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, scheduler, args)

            # evaluate on test set
            test_loss, test_acc = test(test_loader, model, criterion, args)
            if args.local_rank == 0:
                print("[Epoch {}] Train Loss : {} / Train Acc : {} /// Test Loss : {} / Test Acc : {}".format(str(epoch), str(train_loss), str(train_acc*100), str(test_loss), str(test_acc*100)))
            # remember best acc@1 and save checkpoint
            is_best = test_loss < best_test_loss
            best_test_loss = min(test_loss, best_test_loss)

            if args.local_rank == 0 and epoch%10 == 0:
            #if args.local_rank == 0 and is_best:
                torch.save(model.module.state_dict(), './video/'+file_name + '_' + args.indicator + '_'+str(epoch)+'.pth')

def train(train_loader, model, criterion, optimizer, epoch, scheduler, args):

    train_losses = AverageMeter('Loss', ':.4e')
    train_acc = AverageMeter('Accuracy', ':.4e')
    model.train()

    for data, labels in train_loader:
        data = data.cuda(args.local_rank)
        labels = labels.cuda(args.local_rank).squeeze()
        output = model(data).squeeze(-1)
        output = output.reshape(-1, 14, 14)[:, 1:-1, 1:-1].reshape(-1, 144)
        labels = labels.reshape(-1, 14, 14)[:, 1:-1, 1:-1].reshape(-1, 144) # cutting out the corners
        train_loss = criterion(output, labels) # reduction = sum
        train_losses.update(train_loss.item(), np.prod(output.shape))
        train_loss = train_loss/np.prod(output.shape) # reverting to mean

        x = DCN(torch.sigmoid(output))
        x = np.where(x>=0.5, 1, 0)
        answer_ratio = np.mean(np.where(x==DCN(labels), 1, 0))
        train_acc.update(answer_ratio, np.prod(output.shape))

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

    scheduler.step(train_losses.avg, epoch)
    return train_losses.avg, train_acc.avg

def test(test_loader, model, criterion, args):

    test_losses = AverageMeter('Loss', ':.4e')
    test_acc = AverageMeter('Accuracy', ':.4e')
    model.eval()

    with torch.no_grad():
        for data, labels in test_loader:
            data = data.cuda(args.local_rank)
            labels = labels.cuda(args.local_rank).squeeze()
            output = model(data).squeeze(-1)
            output = output.reshape(-1, 14, 14)[:, 1:-1, 1:-1].reshape(-1, 144)
            labels = labels.reshape(-1, 14, 14)[:, 1:-1, 1:-1].reshape(-1, 144) # cutting out the corners
            test_loss = criterion(output, labels) # reduction = sum
            test_losses.update(test_loss.item(), np.prod(output.shape))

            x = DCN(torch.sigmoid(output))
            x = np.where(x>=0.5, 1, 0)
            answer_ratio = np.mean(np.where(x==DCN(labels), 1, 0))
            test_acc.update(answer_ratio, np.prod(output.shape))

    return test_losses.avg, test_acc.avg

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
    print('started!') # For test
    main()
