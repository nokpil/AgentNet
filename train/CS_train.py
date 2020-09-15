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
import traceback
import warnings
import sys
import distutils
import distutils.util

import src.DataStructure as DS
from src.utils import *
from src.system import *
from src.model import *

def str2bool(v):
    return bool(distutils.util.strtobool(v))

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

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
parser.add_argument('--ob-num', default = 'A', type = str)
parser.add_argument('--att-dim', default = '10', type = int, help = 'Dimension of attention vector')
parser.add_argument('--att-num', default = 1, type = int, help = 'For "multi", works as number of heads.')
parser.add_argument('--mode-num', type = int, help = 'Number of gaussian mixture mode.')
parser.add_argument('--dropout', default = 0.0, type = float, help = 'Rate of dropout on attention.')
parser.add_argument('--time_interval', default=30, type = int)
parser.add_argument('--jump_interval', default=20, type = int)
parser.add_argument('--eval-type', default = 'pv2pv', type = str, help = 'pv2pv, pv2v, pv2a, p2p')
parser.add_argument('--checkpoint', default = 'no', type = str, help = 'no, cp')
parser.add_argument('--indicator', default = '', type = str, help = 'Additional specification for file name.')
parser.add_argument("--seed", default=42, type=int, help="Random seed for torch and numpy")
parser.add_argument("--forcing-period", default=30, type=int, help="Teacher forcing period")
parser.add_argument("--sig", default=True, type=str2bool, help="Whether using generated variance or not")
parser.add_argument("--use-sample", default=True, type=str2bool, help="Whether use generated mu or not")

def evalType(eval_type, inout):
    x = eval_type.split('2')
    if inout == 'in':
        return x[0]
    elif inout == 'out':
        return x[1]
    else:
        return NotImplementedError

def main():
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    best_test_loss = np.inf
    print(torch.cuda.device_count(), args.local_rank)
    torch.distributed.init_process_group(backend='nccl',
                                         init_method='env://')
    torch.cuda.set_device(args.local_rank)

    def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
        log = file if hasattr(file,'write') else sys.stderr
        traceback.print_stack(file=log)
        log.write(warnings.formatwarning(message, category, filename, lineno, line))

    warnings.showwarning = warn_with_traceback

    # Plugin parameters
    pp = {}
    system = Flock_LSTM()
    
    if system.name[:2] == 'FL':
        pp['ob_num'] = 'A'
        pp['time_interval'] = args.time_interval
        system.assign_pp(pp)

    system.time_interval = args.time_interval
    system.step_num = 10
    system.jump_interval = 20
    system.count = 0
    system.max_agent = 1463  # For entire sequence (length 10), 1886

    # Data loading code
    #file_name = system.rule_name + '_OB' + str(system.ob_num) + '_TI' + str(system.time_interval) + '_JI' + str(system.jump_interval) + '_LSTM'
    file_name = system.rule_name + '_OB' + str(system.ob_num) + '_TI' + str(system.time_interval) + '_LSTM'
    indicator = '_MT' + str(args.model_type) + '_BT' + str(args.block_type) + '_AN' + str(args.att_num) + '_Dk' + str(args.mode_num) + '_DO' + str(args.dropout)+  '_ET' + str(args.eval_type) + '_' + args.indicator
    train_set = DS.Customset('./data/Flock/dataset/' + file_name, train=True)
    test_set = DS.Customset('./data/Flock/dataset/' + file_name, train=False)
    if args.local_rank == 0:
        print(file_name + indicator)

    # Data masking

    data_mask = [0, 1, 2, 3, 4, 5, 9]  # x, y, z, vx, vy, vz, G
    answer_mask = [0, 1, 2, 3, 4, 5, 6, 7]  # G, N, x, y, z, vx, vy, vz
    if evalType(args.eval_type, 'in') == 'p':
        data_mask = [0, 1, 2, 9]  # x, y, z, vx, vy, vz, G
        answer_mask = [0, 1, 2, 3, 4]  # G, N, x, y, z, vx, vy, vz
    
    train_set.train_data = train_set.train_data[:, :, data_mask]
    test_set.test_data = test_set.test_data[:, :, data_mask]

    train_set.train_labels = train_set.train_labels[:, :, :, answer_mask]
    test_set.test_labels = test_set.test_labels[:, :, :, answer_mask]

    system.state_num = len(data_mask)
    system.answer_num = 1
    
    # Data normalizing
    norm_constant = torch.max(torch.max(torch.abs(train_set.train_data[:, :, :-1]), dim=1)[0], dim=0)[0]  # numerical stability
    
    #norm_constant = torch.tensor([100., 100., 100., 20., 20., 20.])
    norm_constant = torch.tensor([1., 1., 1., 1., 1., 1.])
    train_set.train_data[:, :, :-1] /= norm_constant
    test_set.test_data[:, :, :-1] /= norm_constant
    train_set.train_labels[:, :, :, 2:] /= norm_constant
    test_set.test_labels[:, :, :, 2:] /= norm_constant
    norm_constant = norm_constant.cuda(args.local_rank)

    # Exclude overlapping area
    train_set.train_data = train_set.train_data[:-int(system.step_num * system.time_interval / system.jump_interval)]
    train_set.train_labels = train_set.train_labels[:-int(system.step_num * system.time_interval / system.jump_interval)]

    # sort data by max agent (545 = 700 * 0.8 - 15)
    with open('./data/Flock/system/'+'flock_lstm_OBA_agent_list.pkl', 'rb') as f:
        agent_list = pickle.load(f)    
    #
    print('norm_constant : {}'.format(str(norm_constant)))
    print('agent_list length : {}'.format(str(len(agent_list))))
    train_sortby = np.argsort(agent_list[:545])
    test_sortby = np.argsort(agent_list[560:700-1])
    #train_sortby = np.argsort(agent_list[:155])
    #test_sortby = np.argsort(agent_list[159:-1])
    #train_sortby = np.argsort(agent_list[:79])
    #test_sortby = np.argsort(agent_list[79:-1])

    train_set.train_data = train_set.train_data[train_sortby]
    train_set.train_labels = train_set.train_labels[train_sortby]
    test_set.test_data = test_set.test_data[test_sortby]
    test_set.test_labels = test_set.test_labels[test_sortby]

    #Half the size
    train_set.train_data = train_set.train_data
    train_set.train_labels = train_set.train_labels
    #test_set.test_data = test_set.test_data[:]
    #test_set.test_labels = test_set.test_labels[:]

    def collate_fn(batch):
        data, labels = zip(*batch)
        data, labels = torch.stack(data), torch.stack(labels)

        # data max

        data_checksum = torch.max(torch.argmax((data[:,:,-1]>0).double(), dim = 1)+1)
        seq_sum = torch.sum((labels[:,:,:,1]>1).float(), dim = 1)
        label_checksum = torch.max(torch.argmax((torch.sum(labels[:,:,:,1], dim = 1)>0).double(), dim = 1)+1)
        cm = torch.max(data_checksum, label_checksum)
        seq_len = seq_sum[:, :cm] 

        data, labels = data[:,:cm,:], labels[:,:,:cm, :]

        mask_input_new = labels[:,:,:,0].unsqueeze(-1)
        mask_input_next = ((labels[:,:,:,1].unsqueeze(-1))>0).float()
        mask_input_check = ((labels[:,:,:,1].unsqueeze(-1))>1).float()
        labels = labels[:,:,:,2:]

        present = torch.zeros_like(mask_input_check[:,0,:,0])
        zero_present = torch.zeros_like(mask_input_check[:,0,:,0])
        target_mask_list = [[] for _ in range(labels.shape[1])]

        for i in range(labels.shape[1]):
            present += (mask_input_check[:,i,:,0]>0).float()
            for j in range(labels.shape[1]):
                target_mask_list[i].append((present==(j+1)).float())
            present = torch.where(present == seq_len, torch.zeros_like(zero_present), present)

        return data, labels, target_mask_list, mask_input_new, mask_input_next, mask_input_check

    train_sampler = DistributedSampler_LSTM(train_set)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, pin_memory=True,
                              num_workers=args.workers,
                              sampler=train_sampler, collate_fn = collate_fn)
    test_sampler = DistributedSampler_LSTM(test_set)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, pin_memory=True,
                             num_workers=args.workers, sampler=test_sampler, collate_fn = collate_fn)

    if args.model_type == 'mlp':

        D_in_lstm = system.state_num
        D_hidden_lstm = 256

        D_in_dec = D_hidden_lstm  # x, y, z / vx. vy, vz / hdg , hdg_rate, V/ group
        D_hidden_dec = 256
        D_out_dec = 128
        D_hidden_stat = 128

        D_agent = system.max_agent
        D_s = 3
        D_k = args.mode_num

        cfg_init = [D_in_lstm, D_hidden_lstm]
        cfg_lstm = [D_in_lstm, D_hidden_lstm]
        cfg_dec = [D_in_dec, D_hidden_dec, D_hidden_dec, D_out_dec]

        cfg_mu = [D_out_dec, D_hidden_stat, D_k]
        cfg_sig = [D_out_dec, D_hidden_stat, D_k]

        model = Module_MLP_LSTM_split(cfg_init, cfg_lstm, cfg_dec, cfg_mu, cfg_sig, D_agent, args.block_type, args.eval_type, args.sig, args.use_sample).cuda()

    elif args.model_type == 'gat':

        D_head = int(args.att_dim / args.att_num)

        D_in_lstm = system.state_num
        D_hidden_lstm = 128

        D_in_enc = D_hidden_lstm
        D_hidden_enc = 128
        D_out_enc = args.att_dim
        D_out_self = args.att_dim  

        D_att = args.att_dim
        D_att_num = args.att_num

        #D_in_dec = D_head + D_att
        D_in_dec = D_att * 2
        D_hidden_dec = 64
        D_out_dec = 64
        D_hidden_stat = 64

        D_agent = system.max_agent
        D_k = args.mode_num

        cfg_init = [D_in_lstm, D_hidden_lstm]
        cfg_lstm = [D_in_lstm, D_hidden_lstm]

        cfg_enc = [D_in_enc, D_hidden_enc, D_out_enc]
        cfg_self = [D_in_enc, D_hidden_enc, D_out_self]
        cfg_att = [D_att * 2, 8, 8, D_att_num]
        cfg_dec = [D_in_dec, D_hidden_dec, D_out_dec]

        cfg_mu = [D_out_dec, D_hidden_stat, D_k]
        cfg_sig = [D_out_dec, D_hidden_stat, D_k]


        model = Module_GAT_LSTM_split(cfg_init, cfg_lstm, cfg_enc, cfg_self, cfg_att, cfg_dec, cfg_mu, cfg_sig, D_att, D_att_num,
                             D_agent, args.block_type,  args.att_type, args.dropout, args.eval_type, args.sig, args.use_sample).cuda()
    else:
        print('NOT IMPLEMENTED : model type')

    # define loss function (criterion) and optimizer

    criterion = gmm_criterion(system.answer_num)
    #criterion = 
    optimizer = torch.optim.AdamW(
        model.parameters(), args.lr, weight_decay=args.weight_decay
    )
    # optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum = 0.9, weight_decay = args.weight_decay)
    
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 50, 1, eta_min=0)
    scheduler = RLRP(optimizer, "min", factor=0.7, patience=20, min_lr=0, verbose=1)
    sampler = gmm_sample(system.answer_num)
    #cudnn.benchmark = False

    if args.checkpoint[:2] == "cp":
        print('cp entered')
        checkpoint = torch.load(file_name + '_' + indicator +'_checkpoint.pth',  map_location='cuda:{}'.format(args.local_rank))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"]
        best_test_loss = checkpoint["best_test_loss"]
        epochs = args.epochs + start_epoch
        if len(args.checkpoint) > 2:
            change_rate = float(args.checkpoint[2:])
            for g in optimizer.param_groups:
                g['lr'] = args.lr * change_rate
            #scheduler.base_lrs[0] = args.lr * change_rate
            if args.local_rank == 0:
                print(optimizer.param_groups[0]["lr"])
            #if args.local_rank == 0:
            #    print(f'change rate : {}, change_ratechanged lr : {optimizer.param_groups[0]#["lr"]}, scheduler_lr : {scheduler.get_lr()[0]}')
    else:
        start_epoch = args.start_epoch
        epochs = args.epochs

    
    # 임시
    
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank, find_unused_parameters = True)
    
    with torch.autograd.detect_anomaly():
        for epoch in range(start_epoch, epochs):
            #train_sampler.set_epoch(epoch)

            # train for one epoch
            if args.local_rank==0:
                print("============== Epoch {} =============".format(epoch))
            train_loss, train_count = train(train_loader, model, criterion, optimizer, epoch, scheduler, sampler, norm_constant, args)
            
            if args.local_rank == 0:
                train_string = "Epoch {} / Train Loss : [Total : {}] {}, ".format(str(epoch), str(train_count[-1]), str(train_loss[-1]))
                for i in range(system.step_num):
                    train_string += " [{} : {}] {}, ".format(str(i+1), str(train_count[i]), str(train_loss[i]))
                train_string += f' / Learning Rate : {optimizer.param_groups[0]["lr"]}'
                print(train_string)

            # evaluate on test set
            #if epoch > args.forcing_period:
            if epoch > 0:
                test_loss, test_count = test(test_loader, model, criterion, sampler, norm_constant, args)
                #scheduler.step()
                scheduler.step(test_loss[-1], epoch)
                if args.local_rank == 0:
                    test_string = "Epoch {} / Test Loss : [Total : {}] {}, ".format(str(epoch), str(test_count[-1]), str(test_loss[-1]))
                    for i in range(system.step_num):
                        test_string += " [{} : {}] {}, ".format(str(i+1), str(test_count[i]), str(test_loss[i]))
                    print(test_string)

                    # remember best acc@1 and save checkpoint
                    #is_best = test_loss[-1] < best_test_loss
                    #best_test_loss = min(test_loss[-1], best_test_loss)
                    #print(is_best, best_test_loss, test_loss[-1])
                    is_best = test_loss[-1] < best_test_loss
                    best_test_loss = min(test_loss[-1], best_test_loss)
                    print(is_best, best_test_loss, test_loss[-1])
                    if is_best:
                        #torch.save(model.module.state_dict(), file_name + '_' + indicator +'.pth')
                        torch.save({
                            'epoch' : epoch,
                            'model_state_dict' : model.module.state_dict(),
                            'optimizer_state_dict' : optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                            "loss": test_loss,
                            "best_test_loss" : best_test_loss,
                            "train_loss_list" : (train_loss, train_count),
                            "test_loss_list" : (test_loss, test_count)
                        }, file_name + '_' + indicator +'_checkpoint.pth')
                    

def train(train_loader, model, criterion, optimizer, epoch, scheduler, sampler, norm_constant, args):
   
    train_losses_list = [] 
    step_num = train_loader.dataset.train_labels.shape[1] 
    for i in range(step_num): # labels.shape[1]
        train_losses_list.append(AverageMeter('Loss_' + str(i), ':.4e'))
    train_losses_list.append(AverageMeter('Total_Loss', ':.4e'))	
    model.train()
    forcing_period = args.forcing_period
    
    for i, (data, labels, target_mask_list, mask_input_new, mask_input_next, mask_input_check) in enumerate(train_loader):
        data = data.cuda(args.local_rank)
        hidden, cell = None, None
        optimizer.zero_grad()

        if args.local_rank==0 and epoch == 0:
            print(i, data.shape)

        total_count = [0. for i in range(step_num + 1)]
        for n in range(step_num):
            for j in range(step_num):
                total_count[j] += int(torch.sum(target_mask_list[n][j]))
                total_count[-1] += int(torch.sum(target_mask_list[n][j]))

        for n in range(step_num): #labels.shape[1]
            label = labels.cuda(args.local_rank)[:, n]
            hidden_mask = mask_input_new.cuda(args.local_rank)[:, n]
            data_mask = mask_input_next.cuda(args.local_rank)[:, n]
            label_mask = mask_input_check.cuda(args.local_rank)[:, n]

            label_diff = (label - data[:,:,:-1]) # remove Group part

            hidden, cell = model.module.initialize(data, hidden, cell, hidden_mask)
            (mu_x, sig_x), (mu_y, sig_y), (mu_z, sig_z), (mu_vx, sig_vx), (mu_vy, sig_vy), (mu_vz, sig_vz), hidden, cell = model(data, hidden, cell)
            
            nll_x = criterion(label_diff[:, :, 0], mu_x, sig_x)  
            nll_y = criterion(label_diff[:, :, 1], mu_y, sig_y)
            nll_z = criterion(label_diff[:, :, 2], mu_z, sig_z)
            nll_vx = criterion(label_diff[:, :, 3], mu_vx, sig_vx)
            nll_vy = criterion(label_diff[:, :, 4], mu_vy, sig_vy)
            nll_vz = criterion(label_diff[:, :, 5], mu_vz, sig_vz)
            nll = nll_x + nll_y + nll_z + nll_vx + nll_vy + nll_vz

            if (args.local_rank == 0) and (i == 0) and (n==0) :
                #print(i, n)
                print(torch.mean(nll_x * label_mask.squeeze()).item(), torch.mean(nll_y * label_mask.squeeze()).item(), torch.mean(nll_z * label_mask.squeeze()).item())
                print(torch.mean(nll_vx * label_mask.squeeze()).item(), torch.mean(nll_vy * label_mask.squeeze()).item(), torch.mean(nll_vz * label_mask.squeeze()).item())
                print(label_diff.shape, mu_x.shape, sig_x.shape)
                print(label_diff[0][:10, 0], mu_x[:10], sig_x[:10])
                print(label_diff[0][:10, 3], mu_vx[:10], sig_vx[:10])

            #train_loss_real = torch.sum(nll * label_mask.squeeze()) / torch.sum(label_mask.squeeze())
            #train_losses_list[-1].update(train_loss_real.item(), int(torch.sum(label_mask.squeeze())))
            
            
            train_loss_real = 0.
            for j in range(step_num):
                if torch.sum(target_mask_list[n][j] > 0):
                    train_loss_tmp = torch.sum(nll * target_mask_list[n][j].cuda(args.local_rank)) / int(torch.sum(target_mask_list[n][j]))
                    train_losses_list[j].update(train_loss_tmp.item(), int(torch.sum(target_mask_list[n][j])))
                    train_losses_list[-1].update(train_loss_tmp.item(), int(torch.sum(target_mask_list[n][j])))
                    train_loss_real += torch.sum(nll * target_mask_list[n][j].cuda(args.local_rank)) * total_count[-1] / total_count[j]
            
            train_loss_real.backward(retain_graph = True)
            optimizer.step()
            
            # Teacher forcing (depends on epoch)
            if args.indicator[-2:]=='tf'and epoch <= forcing_period:
                if args.use_sample:
                    sample_x = sampler(mu_x, sig_x).unsqueeze(-1)
                    sample_y = sampler(mu_y, sig_y).unsqueeze(-1)
                    sample_z = sampler(mu_z, sig_z).unsqueeze(-1)
                    sample_vx = sampler(mu_vx, sig_vx).unsqueeze(-1)
                    sample_vy = sampler(mu_vy, sig_vy).unsqueeze(-1)
                    sample_vz = sampler(mu_vz, sig_vz).unsqueeze(-1)
                else:
                    sample_x = mu_x.unsqueeze(-1)
                    sample_y = mu_y.unsqueeze(-1)
                    sample_z = mu_z.unsqueeze(-1)
                    sample_vx = mu_vx.unsqueeze(-1)
                    sample_vy = mu_vy.unsqueeze(-1)
                    sample_vz = mu_vz.unsqueeze(-1)

                sample = torch.cat((sample_x, sample_y, sample_z, sample_vx, sample_vy, sample_vz), dim=-1)
                if type(sample) == type(None):
                    if args.local_rank == 0:
                            print('tuzim')
                    sample = label_diff
                else:
                    sample = sample.cuda(args.local_rank)

                if sample.shape[0] > 100:  # This means that batch size is 1, thus automatically squeezed
                    sample = sample.unsqueeze(0)

                next_data_mask = (torch.bernoulli(torch.ones((sample.shape[0], sample.shape[1], 1))*F.relu(torch.tensor(1-epoch/forcing_period))))
                next_data_mask = next_data_mask.to(data.device)
                #if verbose: print(next_data_mask.shape, label_diff.shape, sample.shape)
                next_data = next_data_mask * label_diff + (1-next_data_mask) * sample
                next_data = next_data + data[:,:,:-1]

                for j in range(labels.shape[0]):
                    next_first_mask = (data_mask[j]-label_mask[j]).squeeze(-1).bool()
                    next_data[j][next_first_mask] = label[j][next_first_mask]
            else:
                sample_x = sampler(mu_x, sig_x).unsqueeze(-1)
                sample_y = sampler(mu_y, sig_y).unsqueeze(-1)
                sample_z = sampler(mu_z, sig_z).unsqueeze(-1)
                sample_vx = sampler(mu_vx, sig_vx).unsqueeze(-1)
                sample_vy = sampler(mu_vy, sig_vy).unsqueeze(-1)
                sample_vz = sampler(mu_vz, sig_vz).unsqueeze(-1)
                sample = torch.cat((sample_x, sample_y, sample_z, sample_vx, sample_vy, sample_vz), dim=-1)
                if type(sample) == type(None):
                    if args.local_rank == 0:
                            print('tuzim')
                    sample = label_diff
                else:
                    sample = sample.cuda(args.local_rank)

                if sample.shape[0] > 100:  # This means that batch size is 1, thus automatically squeezed
                    sample = sample.unsqueeze(0)
                next_data = sample + data[:,:,:-1]

                for j in range(labels.shape[0]):
                    next_first_mask = (data_mask[j]-label_mask[j]).squeeze(-1).bool()
                    next_data[j][next_first_mask] = label[j][next_first_mask]
                
            data = torch.cat((next_data, 2*data_mask-1), dim = -1)

        del mu_x, sig_x, mu_y, sig_y, mu_z, sig_z, mu_vx, sig_vx, mu_vy, sig_vy, mu_vz, sig_vz
        del nll_x, nll_y, nll_z, nll_vx, nll_vy, nll_vz, nll, train_loss_real 
        del sample_x, sample_y, sample_z, sample_vx, sample_vy, sample_vz, sample, next_data
        del data, hidden, cell, hidden_mask, data_mask, label_mask, label_diff
        torch.cuda.empty_cache()

    return [train_losses_list[i].avg for i in range(len(train_losses_list))], [train_losses_list[i].count for i in range(len(train_losses_list))]

def test(test_loader, model, criterion, sampler, norm_constant, args):

    test_losses_list = []  
    step_num = test_loader.dataset.test_labels.shape[1]
    for i in range(step_num):
        test_losses_list.append(AverageMeter('Loss_' + str(i), ':.4e'))
    test_losses_list.append(AverageMeter('Total_Loss', ':.4e'))		
    model.eval()

    with torch.no_grad():
        for i, (data, labels, target_mask_list, mask_input_new, mask_input_next, mask_input_check) in enumerate(test_loader):
            data = data.cuda(args.local_rank)
            hidden, cell = None, None

            total_count = [0. for i in range(step_num + 1)]
            for n in range(step_num):
                for j in range(step_num):
                    total_count[j] += int(torch.sum(target_mask_list[n][j]))
                    total_count[-1] += int(torch.sum(target_mask_list[n][j]))

            for n in range(step_num): #labels.shape[1]
                label = labels.cuda(args.local_rank)[:, n]
                hidden_mask = mask_input_new.cuda(args.local_rank)[:, n]
                data_mask = mask_input_next.cuda(args.local_rank)[:, n]
                label_mask = mask_input_check.cuda(args.local_rank)[:, n]
                label_diff = (label - data[:,:,:-1]) # remove Group part

                hidden, cell = model.module.initialize(data, hidden, cell, hidden_mask)
                (mu_x, sig_x), (mu_y, sig_y), (mu_z, sig_z), (mu_vx, sig_vx), (mu_vy, sig_vy), (mu_vz, sig_vz), hidden, cell = model(data, hidden, cell)
                
                nll_x = criterion(label_diff[:, :, 0], mu_x, sig_x)  
                nll_y = criterion(label_diff[:, :, 1], mu_y, sig_y)
                nll_z = criterion(label_diff[:, :, 2], mu_z, sig_z)
                nll_vx = criterion(label_diff[:, :, 3], mu_vx, sig_vx)
                nll_vy = criterion(label_diff[:, :, 4], mu_vy, sig_vy)
                nll_vz = criterion(label_diff[:, :, 5], mu_vz, sig_vz)
                nll = nll_x + nll_y + nll_z + nll_vx + nll_vy + nll_vz

                #test_loss = torch.sum(nll * label_mask.squeeze())
                #test_losses_list[-1].update(test_loss.item() / int(torch.sum(label_mask)), int(torch.sum(label_mask)))
                
                for j in range(step_num):
                    if torch.sum(target_mask_list[n][j] > 0):
                        test_loss_tmp = torch.sum(nll * target_mask_list[n][j].cuda(args.local_rank)) / int(torch.sum(target_mask_list[n][j]))
                        test_losses_list[j].update(test_loss_tmp.item(), int(torch.sum(target_mask_list[n][j])))
                        test_losses_list[-1].update(test_loss_tmp.item(), int(torch.sum(target_mask_list[n][j])))
                
                if args.use_sample:
                    sample_x = sampler(mu_x, sig_x).unsqueeze(-1)
                    sample_y = sampler(mu_y, sig_y).unsqueeze(-1)
                    sample_z = sampler(mu_z, sig_z).unsqueeze(-1)
                    sample_vx = sampler(mu_vx, sig_vx).unsqueeze(-1)
                    sample_vy = sampler(mu_vy, sig_vy).unsqueeze(-1)
                    sample_vz = sampler(mu_vz, sig_vz).unsqueeze(-1)
                else:
                    sample_x = mu_x.unsqueeze(-1)
                    sample_y = mu_y.unsqueeze(-1)
                    sample_z = mu_z.unsqueeze(-1)
                    sample_vx = mu_vx.unsqueeze(-1)
                    sample_vy = mu_vy.unsqueeze(-1)
                    sample_vz = mu_vz.unsqueeze(-1)
                sample = torch.cat((sample_x, sample_y, sample_z, sample_vx, sample_vy, sample_vz), dim=-1)
                if type(sample) == type(None) :
                    if args.local_rank == 0:
                            print('tuzim')
                    sample = label_diff
                else:
                    sample = sample.cuda(args.local_rank)
            
                next_data = sample + data[:,:,:-1]
                for j in range(labels.shape[0]):
                    next_first_mask = (data_mask[j]-label_mask[j]).squeeze(-1).bool()
                    next_data[j][next_first_mask] = label[j][next_first_mask]
                
                data = torch.cat((next_data, 2*data_mask-1), dim = -1)
                # data = torch.cat((mu.squeeze(), data[:,:,-1].unsqueeze(-1)), dim = -1)

            del nll_x, nll_y, nll_z, nll_vx, nll_vy, nll_vz, nll
            del sample_x, sample_y, sample_z, sample_vx, sample_vy, sample_vz, sample, next_data
            del data, hidden, cell, hidden_mask, data_mask, label_mask, label_diff
            del mu_x, sig_x, mu_y, sig_y, mu_z, sig_z, mu_vx, sig_vx, mu_vy, sig_vy, mu_vz, sig_vz
            torch.cuda.empty_cache()

    return [test_losses_list[i].avg for i in range(len(test_losses_list))], [test_losses_list[i].count for i in range(len(test_losses_list))]

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
