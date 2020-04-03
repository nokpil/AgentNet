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

import src.DataStructure as DS
from src.mypytorch import *

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
parser.add_argument('--wd', '--weight-decay', default=0, type=float,
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
parser.add_argument('--time_interval', type = int)
parser.add_argument('--jump_interval', type = int)
parser.add_argument('--seed', default = 42, help = 'Random seed for pytorch and numpy.')
parser.add_argument('--eval-type', default = 'pv2pv', type = str, help = 'pv2pv, pv2v, pv2a, p2p')
parser.add_argument('--checkpoint', default = 'no', type = str, help = 'no, cp')
parser.add_argument('--indicator', default = '', type = str, help = 'Additional specification for file name.')

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
    best_test_loss = 100
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
    system.answer_num = len(answer_mask)
    
    # Data normalizing
    norm_constant = torch.max(torch.max(torch.abs(train_set.train_data[:, :, :-1]), dim=1)[0], dim=0)[0]  # numerical stability
    
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
    #train_set.train_data = train_set.train_data[::2]
    #train_set.train_labels = train_set.train_labels[::2]
    #test_set.test_data = test_set.test_data[::2]
    #test_set.test_labels = test_set.test_labels[::2]

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
        D_hidden_lstm = 128

        D_in_dec = D_hidden_lstm  # x, y, z / vx. vy, vz / hdg , hdg_rate, V/ group
        D_hidden_dec = 512
        D_out_dec = 128
        D_hidden_stat = 128

        D_agent = system.max_agent
        D_s = 3
        D_k = args.mode_num

        cfg_init = [D_in_lstm, D_hidden_lstm]
        cfg_lstm = [D_in_lstm, D_hidden_lstm]
        cfg_dec = [D_in_dec, D_hidden_dec, D_hidden_dec, D_hidden_dec, D_out_dec]

        cfg_mu = [D_out_dec, D_hidden_stat, 3 * D_k]
        cfg_sig = [D_out_dec, D_hidden_stat, 3 * D_k]
        cfg_corr = [D_out_dec, D_hidden_stat, 3 * D_k]
        cfg_coef = [D_out_dec, D_hidden_stat, D_k]

        model = Module_MLP_LSTM_single(cfg_init, cfg_lstm, cfg_dec, cfg_mu, cfg_sig, cfg_corr, cfg_coef, D_agent, args.block_type, args.eval_type).cuda()

    elif args.model_type == 'gat':

        D_in_lstm = system.state_num
        D_hidden_lstm = 128

        D_in_enc = D_hidden_lstm
        D_hidden_enc = 128
        D_out_enc = args.att_dim * args.att_num

        D_att = args.att_dim
        D_att_num = args.att_num

        #D_in_dec = D_out_enc * (D_att_num + 1)
        D_in_dec = D_out_enc * 2 
        D_hidden_dec = 256
        D_out_dec = 128
        D_hidden_stat = 128

        D_agent = system.max_agent
        D_s = 3
        D_k = args.mode_num

        cfg_init = [D_in_lstm, D_hidden_lstm]
        cfg_lstm = [D_in_lstm, D_hidden_lstm]

        cfg_enc = [D_in_enc, D_hidden_enc, D_out_enc]
        cfg_att = [D_att*2, 16, 8, 1]
        cfg_dec = [D_in_dec, D_hidden_dec, D_out_dec]

        cfg_mu = [D_out_dec, D_hidden_stat, 3 * D_k]
        cfg_sig = [D_out_dec, D_hidden_stat, 3 * D_k]
        cfg_corr = [D_out_dec, D_hidden_stat, 3 * D_k]
        cfg_coef = [D_out_dec, D_hidden_stat, D_k]

        model = Module_GAT_LSTM_single(cfg_init, cfg_lstm, cfg_enc, cfg_att, cfg_dec, cfg_mu, cfg_sig, cfg_corr, cfg_coef, D_att, D_att_num,
                             D_agent, args.block_type,  args.att_type, args.dropout, args.eval_type)
    else:
        print('NOT IMPLEMENTED : model type')

    # define loss function (criterion) and optimizer

    criterion = gmm_criterion(D_s)
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay = args.weight_decay)
    scheduler = RLRP(optimizer, 'min', factor=0.5, patience=10, min_lr=5e-8, verbose=1)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=20, gamma=0.7)
    sampler = gmm_sample(D_s)
    #cudnn.benchmark = False

    if args.checkpoint == 'cp':
        print('cp entered')
        
        checkpoint = torch.load(file_name + '_' + indicator +'_checkpoint.pth',  map_location='cuda:{}'.format(args.local_rank))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()
        start_epoch = checkpoint['epoch']
        epochs = args.epochs + start_epoch
    else:
        start_epoch = args.start_epoch
        epochs = args.epochs
    
    # 임시
    
    model = torch.nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[args.local_rank],
                                                          output_device=args.local_rank, find_unused_parameters = True)
    
    with torch.autograd.detect_anomaly():
        for epoch in range(start_epoch, epochs):
            #train_sampler.set_epoch(epoch)

            # train for one epoch
            if args.local_rank==0:
                print("============== Epoch {} =============".format(epoch))
            train_loss, train_count = train(train_loader, model, criterion, optimizer, epoch, scheduler, sampler, norm_constant, args)
            # evaluate on test set
            test_loss, test_count = test(test_loader, model, criterion, sampler, norm_constant, args)
            scheduler.step(test_loss[-1], epoch)
            if args.local_rank == 0:
                train_string = "Epoch {} / Train Loss : [Total : {}] {}, ".format(str(epoch), str(train_count[-1]), str(train_loss[-1]))
                test_string = "Epoch {} / Test Loss : [Total : {}] {}, ".format(str(epoch), str(test_count[-1]), str(test_loss[-1]))
                for i in range(system.step_num):
                    train_string += " [{} : {}] {}, ".format(str(i+1), str(train_count[i]), str(train_loss[i]))
                    test_string += " [{} : {}] {}, ".format(str(i+1), str(test_count[i]), str(test_loss[i]))
                print(train_string)
                print(test_string)

            # remember best acc@1 and save checkpoint
            is_best = test_loss[-1] < best_test_loss
            best_test_loss = min(test_loss[-1], best_test_loss)
            if args.local_rank == 0 :
                print(is_best, best_test_loss, test_loss[-1])
                if is_best:
                    #torch.save(model.module.state_dict(), file_name + '_' + indicator +'.pth')
                    torch.save({
                        'epoch' : epoch,
                        'model_state_dict' : model.module.state_dict(),
                        'optimizer_state_dict' : optimizer.state_dict(),
                        'loss' : test_loss
                    }, file_name + '_' + indicator +'_checkpoint.pth')
                    

def train(train_loader, model, criterion, optimizer, epoch, scheduler, sampler, norm_constant, args):
   
    train_losses_list = [] 
    step_num = train_loader.dataset.train_labels.shape[1] 
    for i in range(step_num+1): # labels.shape[1]
        train_losses_list.append(AverageMeter('Loss_' + str(i), ':.4e'))
    train_losses_list.append(AverageMeter('Total_Loss' + str(i), ':.4e'))	
    model.train()

    for i, (data, labels, target_mask_list, mask_input_new, mask_input_next, mask_input_check) in enumerate(train_loader):
        data = data.cuda(args.local_rank)
        hidden = None
        optimizer.zero_grad()
        total_count = [0. for i in range(step_num)]
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

            hidden = model.module.initialize(data, hidden_mask, hidden)
            set1, set2, hidden = model(data, hidden)
            #mu1, sig1, corr1, coef1 = set1
            #mu2, sig2, corr2, coef2 = set2
            mu1, sig1, corr1 = set1
            mu2, sig2, corr2 = set2
            coef1 = None
            coef2 = None
            nll1, cv1 = criterion(label_diff[:,:,:3], mu1, sig1, corr1, coef1, loss_out = True, cv_out = True) # position
            nll2, cv2 = criterion(label_diff[:,:,3:], mu2, sig2, corr2, coef2, loss_out = True, cv_out = True) # velocity
            nll = nll1+nll2
            train_loss = torch.sum(nll* label_mask.squeeze()) / int(torch.sum(label_mask))
            train_losses_list[-1].update(train_loss.item(), int(torch.sum(label_mask)))

            train_loss_real = 0.
            for j in range(step_num):
                if torch.sum(target_mask_list[n][j] > 0):
                    train_loss_tmp = torch.sum(nll * target_mask_list[n][j].cuda(args.local_rank)) / int(torch.sum(target_mask_list[n][j]))
                    train_losses_list[j].update(train_loss_tmp.item(), int(torch.sum(target_mask_list[n][j])))
                    train_loss_real += torch.sum(nll * target_mask_list[n][j].cuda(args.local_rank)) * total_count[-1] / total_count[j]
           
            train_loss_real.backward()
            
            if args.local_rank == 0:
                train_string = "Batch {}, Step {} / Train Loss : {}, ".format(str(i), str(n), str(train_loss_real.item()))

            # Teacher forcing (depends on epoch)
            if args.indicator[-2:]=='tf':
                forcing_period = 30
                sample_pos = sampler(mu1, cv1)
                sample_vel = sampler(mu2, cv2)
                if type(sample_pos) == type(None) or type(sample_vel) == type(None) :
                    if args.local_rank == 0:
                            print('tuzim')
                    sample = label_diff
                else:
                    if args.mode_num == 1: #sample shape = batch_num * agent_num * mode_num * state_nu
                        sample = torch.cat((sample_pos.cuda(args.local_rank), sample_vel.cuda(args.local_rank)), dim = -1).squeeze(-2)
                    else:
                        selected_sample1 = torch.zeros(sample_pos.shape[0], sample_pos.shape[1], sample_pos.shape[3]).to(data.device)
                        selected_sample2 = torch.zeros(sample_vel.shape[0], sample_vel.shape[1], sample_vel.shape[3]).to(data.device)
                        shape = coef1.shape
                        coef1 = coef1.view(-1, 3)
                        coef2 = coef2.view(-1, 3)
                        x1 = torch.multinomial(coef1, 1, replacement = True).view(shape[0], shape[1])
                        x2 = torch.multinomial(coef2, 1, replacement = True).view(shape[0], shape[1])
                        for i in range(shape[0]):
                            for j in range(shape[1]):
                                selected_sample1[i][j] = sample_pos[i][j][x1[i][j]]
                                selected_sample2[i][j] = sample_vel[i][j][x2[i][j]]
                        
                        sample = torch.cat((selected_sample1, selected_sample2), dim = -1).squeeze(-2)

                next_data_mask = (torch.bernoulli(torch.ones((sample.shape[0], sample.shape[1], 1))*F.relu(torch.tensor(1-epoch/forcing_period))).repeat(1, 1, sample.shape[2]))
                next_data_mask = next_data_mask.to(data.device)
                next_data = next_data_mask* label_diff + (1-next_data_mask) * sample
                nex_data = next_data + data[:,:,:-1]

                for j in range(labels.shape[0]):
                    next_first_mask = (data_mask[j]-label_mask[j]).squeeze(-1).bool()
                    next_data[j][next_first_mask] = label[j][next_first_mask]
            else:
                next_data = label
                
            data = torch.cat((next_data, 2*data_mask-1), dim = -1)
        #if args.local_rank==0:
        #    print(i)
        #    torch.save(model.module.state_dict(), 'test'+str(i)+'.pth')
        optimizer.step()

    return [train_losses_list[i].avg for i in range(len(train_losses_list))], [train_losses_list[i].count for i in range(len(train_losses_list))]


def test(test_loader, model, criterion, sampler, norm_constant, args):

    test_losses_list = []  
    step_num = test_loader.dataset.test_labels.shape[1]
    for i in range(step_num+1):
        test_losses_list.append(AverageMeter('Loss_' + str(i), ':.4e'))	
    model.eval()

    with torch.no_grad():
        for i, (data, labels, target_mask_list, mask_input_new, mask_input_next, mask_input_check) in enumerate(test_loader):
            data = data.cuda(args.local_rank)
            hidden = None

            total_count = [0. for i in range(step_num)]
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

                hidden = model.module.initialize(data, hidden_mask, hidden)
                set1, set2, hidden = model(data, hidden)
                #mu1, sig1, corr1, coef1 = set1
                #mu2, sig2, corr2, coef2 = set2
                mu1, sig1, corr1 = set1
                mu2, sig2, corr2 = set2
                coef1 = None
                coef2 = None
                nll1, cv1 = criterion(label_diff[:,:,:3], mu1, sig1, corr1, coef1, loss_out = True, cv_out = True) # position
                nll2, cv2 = criterion(label_diff[:,:,3:], mu2, sig2, corr2, coef2, loss_out = True, cv_out = True) # velocity
                nll = nll1+nll2
                test_loss = torch.sum(nll * label_mask.squeeze()) / int(torch.sum(label_mask))
                test_losses_list[-1].update(test_loss.item(), int(torch.sum(label_mask)))

                test_loss_real = 0.
                for j in range(step_num):
                    if torch.sum(target_mask_list[n][j] > 0):
                        test_loss_tmp = torch.sum(nll * target_mask_list[n][j].cuda(args.local_rank)) / int(torch.sum(target_mask_list[n][j]))
                        test_losses_list[j].update(test_loss_tmp.item(), int(torch.sum(target_mask_list[n][j])))
                        test_loss_real += torch.sum(nll * target_mask_list[n][j].cuda(args.local_rank)) * total_count[-1] / total_count[j]
                    #print(mu.squeeze().shape, data[:,:,-1].unsqueeze(-1).shape)

                sample_pos = sampler(mu1, cv1)
                sample_vel = sampler(mu2, cv2)

                if type(sample_pos) == type(None) or type(sample_vel) == type(None) :
                    if args.local_rank == 0:
                            print('tuzim')
                    sample = label_diff
                else:
                    if args.mode_num == 1: #sample shape = batch_num * agent_num * mode_num * state_nu
                        sample = torch.cat((sample_pos.cuda(args.local_rank), sample_vel.cuda(args.local_rank)), dim = -1).squeeze(-2)
                    else:
                        selected_sample1 = torch.zeros(sample_pos.shape[0], sample_pos.shape[1], sample_pos.shape[3]).to(data.device)
                        selected_sample2 = torch.zeros(sample_vel.shape[0], sample_vel.shape[1], sample_vel.shape[3]).to(data.device)
                        shape = coef1.shape
                        coef1 = coef1.view(-1, 3)
                        coef2 = coef2.view(-1, 3)
                        x1 = torch.multinomial(coef1, 1, replacement = True).view(shape[0], shape[1])
                        x2 = torch.multinomial(coef2, 1, replacement = True).view(shape[0], shape[1])
                        for i in range(shape[0]):
                            for j in range(shape[1]):
                                selected_sample1[i][j] = sample_pos[i][j][x1[i][j]]
                                selected_sample2[i][j] = sample_vel[i][j][x2[i][j]]
                        
                        sample = torch.cat((selected_sample1, selected_sample2), dim = -1).squeeze(-2)

                next_data = sample + data[:,:,:-1]
                for j in range(labels.shape[0]):
                    next_first_mask = (data_mask[j]-label_mask[j]).squeeze(-1).bool()
                    next_data[j][next_first_mask] = label[j][next_first_mask]
                
                data = torch.cat((next_data, 2*data_mask-1), dim = -1)
                # data = torch.cat((mu.squeeze(), data[:,:,-1].unsqueeze(-1)), dim = -1)
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
