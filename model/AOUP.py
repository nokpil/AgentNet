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
from torch.utils.data import (
    DataLoader,
)  # (testset, batch_size=4,shuffle=False, num_workers=4)
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
import tracemalloc

import src.DataStructure as DS
from src.mypytorch import *

parser = argparse.ArgumentParser(description="Pytorch VAINS Training")

parser.add_argument(
    "-j",
    "--workers",
    default=0,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 4)",
)
parser.add_argument(
    "--epochs", default=2, type=int, metavar="N", help="number of total epochs to run"
)
parser.add_argument(
    "--start-epoch",
    default=0,
    type=int,
    metavar="N",
    help="manual epoch number (useful on restarts)",
)
parser.add_argument(
    "-b",
    "--batch-size",
    default=16,
    type=int,
    metavar="N",
    help="mini-batch size (default: 256), this is the total "
    "batch size of all GPUs on the current node when "
    "using Data Parallel or Distributed Data Parallel",
)
parser.add_argument(
    "--lr",
    "--learning-rate",
    default=5e-4,
    type=float,
    metavar="LR",
    help="initial learning rate",
    dest="lr",
)
parser.add_argument(
    "--wd",
    "--weight-decay",
    default=0,
    type=float,
    metavar="W",
    help="weight decay (default: 0)",
    dest="weight_decay",
)

parser.add_argument(
    "--world-size", default=4, type=int, help="number of nodes for distributed training"
)
parser.add_argument(
    "--rank", default=-1, type=int, help="node rank for distributed training"
)
parser.add_argument(
    "--dist-url",
    default="env://",
    type=str,
    help="url used to set up distributed training",
)
parser.add_argument(
    "--dist-backend", default="nccl", type=str, help="distributed backend"
)

parser.add_argument("--local_rank", type=int)
parser.add_argument(
    "--model-type", default="vains", type=str, help="model type : vains, mlp"
)
parser.add_argument(
    "--block-type",
    default="mlp",
    type=str,
    help="mlp : simple multi-layer perceptron, res : skip-connection",
)
parser.add_argument(
    "--att-type",
    default="single",
    type=str,
    help="single : single attention, multi : multi-head attention, distort : Tanh distortion layer",
)
parser.add_argument(
    "--att-dim", default="10", type=int, help="Dimension of attention vector"
)
parser.add_argument(
    "--att-num", default=1, type=int, help='For "multi", works as number of heads.'
)
parser.add_argument("--mode-num", type=int, help="Number of gaussian mixture mode.")
parser.add_argument(
    "--dropout", default=0.0, type=float, help="Rate of dropout on attention."
)
parser.add_argument("--eval-type", default="p", type=str, help="p, v, pv")
parser.add_argument("--checkpoint", default="no", type=str, help="no, cp")
parser.add_argument(
    "--indicator", default="", type=str, help="Additional specification for file name."
)


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    tracemalloc.start()
    best_test_loss = 10000
    args = parser.parse_args()
    print(torch.cuda.device_count(), args.local_rank)
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend="nccl", init_method="env://")

    # Plugin parameters
    pp = {}
    system = AOUP()

    # Size parameters

    if system.name == "AOUP":
        pp["agent_num"] = 100
        pp["dt"] = 1 / 100.0
        pp["data_step"] = 8
        pp["label_step"] = 12
        pp["state_num"] = 4
        pp["answer_num"] = 2
        pp["const_num"] = 1
        system.assign_pp(pp)
        # system.assign_Tconst(train_size, test_size, Tconst_list = [[0.3, 1.0, 0.1], [0.1, 1.5, 0.1]])

    # Data loading code
    file_name = (
        system.rule_name
        + "_A"
        + str(system.agent_num)
        + "_dt"
        + str(int(1 / system.dt))
    )
    indicator = (
        "_MT"
        + str(args.model_type)
        + "_BT"
        + str(args.block_type)
        + "_AN"
        + str(args.att_num)
        + "_Dk"
        + str(args.mode_num)
        + "_DO"
        + str(args.dropout)
        + "_ET"
        + str(args.eval_type)
        + "_"
        + args.indicator
    )
    train_set = DS.Customset("./data/AOUP/" + file_name, train=True)
    test_set = DS.Customset("./data/AOUP/" + file_name, train=False)

    if args.local_rank == 0:
        print(file_name + indicator)

    train_sampler = DistributedSampler_LSTM(train_set)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=args.workers,
        sampler=train_sampler,
    )
    test_sampler = DistributedSampler_LSTM(test_set)
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=args.workers,
        sampler=test_sampler,
    )

    if args.eval_type == "p":
        data_mask = [0, 1, 2, 3, 6]  # x, y, vx, vy, R
        answer_mask = [0, 1]  # x, y
    elif args.eval_type == "v":
        data_mask = [0, 1, 2, 3, 6]  # x, y, vx, vy, R
        answer_mask = [2, 3]  # vx, vy
    elif args.eval_type == "pv":
        data_mask = [0, 1, 2, 3, 6]  # x, y, vx, vy, R
        answer_mask = [0, 1, 2, 3]  # x, y, vx, vy

    train_set.train_data = train_set.train_data[:, :, :, data_mask]
    test_set.test_data = test_set.test_data[:, :, :, data_mask]

    train_set.train_labels = train_set.train_labels[:, :, :, answer_mask]
    test_set.test_labels = test_set.test_labels[:, :, :, answer_mask]

    if args.model_type == "mlp":
        D_in_lstm = system.state_num
        D_hidden_lstm = 128

        D_in_dec = D_hidden_lstm  # x, y, z / vx. vy, vz / hdg , hdg_rate, V/ group
        D_hidden_dec = 256
        D_out_dec = 256
        D_hidden_stat = 128
        D_agent = system.agent_num

        D_s = 3
        D_k = args.mode_num

        cfg_init = [D_in_lstm, D_hidden_lstm]
        cfg_lstm = [D_in_lstm, D_hidden_lstm]
        cfg_dec = [D_in_dec, D_hidden_dec, D_hidden_dec, D_hidden_dec, D_out_dec]

        cfg_mu = [D_out_dec, D_hidden_stat, 2 * D_k]
        cfg_sig = [D_out_dec, D_hidden_stat, 2 * D_k]
        cfg_corr = [D_out_dec, D_hidden_stat, D_k]
        cfg_coef = [D_out_dec, D_hidden_stat, D_k]

        model = Module_MLP_AOUP(
            cfg_init,
            cfg_lstm,
            cfg_dec,
            cfg_mu,
            cfg_sig,
            cfg_corr,
            cfg_coef,
            D_agent,
            args.block_type,
            args.eval_type,
        ).cuda()

    elif args.model_type == "gat":
        D_in_lstm = system.state_num
        D_hidden_lstm = 128

        D_in_enc = D_hidden_lstm
        D_hidden_enc = 128
        D_out_enc = args.att_dim * args.att_num

        D_att = args.att_dim
        D_att_num = args.att_num

        D_in_dec = D_out_enc * 2
        D_hidden_dec = 256
        D_out_dec = 256
        D_hidden_stat = 128

        D_agent = system.agent_num
        D_k = args.mode_num

        cfg_init = [D_in_lstm, D_hidden_lstm]
        cfg_lstm = [D_in_lstm, D_hidden_lstm]

        cfg_enc = [D_in_enc, D_hidden_enc, D_out_enc]
        cfg_att = [D_att * 2 + system.const_num, 64, 32, 1]
        cfg_dec = [D_in_dec, D_hidden_dec, D_out_dec]

        cfg_mu = [D_out_dec, D_hidden_stat, 2 * D_k]
        cfg_sig = [D_out_dec, D_hidden_stat, 2 * D_k]
        cfg_corr = [D_out_dec, D_hidden_stat, D_k]
        cfg_coef = [D_out_dec, D_hidden_stat, D_k]

        model = Module_GAT_AOUP_ns(
            cfg_init,
            cfg_lstm,
            cfg_enc,
            cfg_att,
            cfg_dec,
            cfg_mu,
            cfg_sig,
            cfg_corr,
            cfg_coef,
            D_att,
            D_att_num,
            D_agent,
            args.block_type,
            args.att_type,
            args.dropout,
        ).cuda()
    else:
        print("hello")
    # define loss function (criterion) and optimizer

    criterion = gmm_criterion(system.answer_num)
    sampler = gmm_sample(system.answer_num)

    optimizer = torch.optim.Adam(
        model.parameters(), args.lr, weight_decay=args.weight_decay
    )
    # optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum = 0.9, weight_decay = args.weight_decay)
    scheduler = RLRP(optimizer, "min", factor=0.5, patience=30, min_lr=1e-8, verbose=1)
    cudnn.benchmark = True

    if args.checkpoint == "cp":
        print("cp entered")
        checkpoint = torch.load(file_name + "_" + indicator + "_checkpoint.pth")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        epochs = args.epochs + start_epoch
    else:
        start_epoch = args.start_epoch
        epochs = args.epochs

    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[args.local_rank], output_device=args.local_rank
    )

    # snapshot_old = tracemalloc.take_snapshot()
    with torch.autograd.detect_anomaly():
        for epoch in range(start_epoch, epochs):
            # train_sampler.set_epoch(epoch)

            # train for one epoch
            if args.local_rank == 0:
                print("============== Epoch {} =============".format(epoch))
            train_loss, train_count = train(
                train_loader,
                model,
                criterion,
                optimizer,
                epoch,
                scheduler,
                sampler,
                args,
            )

            # evaluate on test set

            test_loss, test_count = test(test_loader, model, criterion, sampler, args)
            if args.local_rank == 0:
                train_string = "Epoch {} / Train Loss : [Total : {}] {}, ".format(
                    str(epoch), str(train_count[-1]), str(train_loss[-1])
                )
                test_string = "Epoch {} / Test Loss : [Total : {}] {}, ".format(
                    str(epoch), str(test_count[-1]), str(test_loss[-1])
                )
                for i in range(system.label_step):
                    train_string += " [{} : {}] {}, ".format(
                        str(i + 1), str(train_count[i]), str(train_loss[i])
                    )
                    test_string += " [{} : {}] {}, ".format(
                        str(i + 1), str(test_count[i]), str(test_loss[i])
                    )
                print(train_string)
                print(test_string)

            # remember best acc@1 and save checkpoint
            is_best = test_loss[-1] < best_test_loss
            best_test_loss = min(test_loss[-1], best_test_loss)

            if args.local_rank == 0 and is_best:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.module.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": test_loss,
                    },
                    file_name + "_" + indicator + "_checkpoint.pth",
                )


def train(train_loader, model, criterion, optimizer, epoch, scheduler, sampler, args):
    g_const = 1
    train_losses_list = []
    data_num = train_loader.dataset.train_data.shape[1]
    step_num = train_loader.dataset.train_labels.shape[1]
    for i in range(step_num + 1):  # labels.shape[1]
        train_losses_list.append(AverageMeter("Loss_" + str(i), ":.4e"))
    train_losses_list.append(AverageMeter("Total_Loss" + str(i), ":.4e"))
    model.train()

    for i, (data, labels) in enumerate(train_loader):
        data = data.cuda(args.local_rank)
        # R_const = data[:,0,:,-1].unsqueeze(-1)
        R_const = (
            data[:, 0, :, -1].unsqueeze(-1).repeat(1, 1, data.shape[-2]).unsqueeze(-1)
        )  # one constant
        data = data[:, :, :, :-1]
        hidden, cell = model.module.initialize(data[:, 0])

        for n in range(data_num - 1):  # data.shape[1]
            hidden, cell = model(data[:, n], hidden, cell, R_const, test=False)

        data = data[:, -1]
        total_train_loss = 0
        for n in range(step_num):  # labels.shape[1]
            label = labels.cuda(args.local_rank)[:, n]
            label_diff = (label - data) * g_const

            set1, set2, hidden, cell = model(data, hidden, cell, R_const, test=True)
            mu1, sig1, corr1, coef1 = set1
            mu2, sig2, corr2, coef2 = set2
            nll1, cv1 = criterion(
                label_diff[:, :, :2],
                mu1,
                sig1,
                corr1,
                coef1,
                loss_out=True,
                cv_out=True,
            )  # position
            nll2, cv2 = criterion(
                label_diff[:, :, 2:],
                mu2,
                sig2,
                corr2,
                coef2,
                loss_out=True,
                cv_out=True,
            )  # velocity

            train_loss = torch.mean(nll1 + nll2)
            train_losses_list[-1].update(train_loss.item(), np.prod(nll1.shape))
            train_losses_list[n].update(train_loss.item(), np.prod(nll1.shape))
            total_train_loss = total_train_loss + train_loss

            optimizer.zero_grad()
            train_loss.backward(retain_graph=True)
            optimizer.step()

            # Teacher forcing (depends on epoch)
            if args.indicator[-2:] == "tf":
                forcing_period = 50
                sample_pos = sampler(mu1, cv1)
                sample_vel = sampler(mu2, cv2)
                sample = torch.cat((sample_pos, sample_vel), dim=-1)
                if type(sample) == type(None):
                    print("tuzim")
                    with open("singular1.pkl", "wb") as f:
                        pickle.dump((mu1, sig1, corr1, nll1, cv1), f)
                    with open("singular2.pkl", "wb") as f:
                        pickle.dump(
                            (mu2.cpu(), sig2.cpu(), corr2.cpu(), nll2.cpu(), cv2.cpu()),
                            f,
                        )
                    break
                else:
                    sample = sample.squeeze(-2).cuda(args.local_rank)
                # next_data_mask = torch.bernoulli(torch.ones((sample.shape[0], sample.shape[1], 1))*F.relu(torch.tensor(min(1, (n+2)-epoch/forcing_period)))).repeat(1, 1, sample.shape[2]).cuda(args.local_rank)
                next_data_mask = (
                    torch.bernoulli(
                        torch.ones((sample.shape[0], sample.shape[1], 1))
                        * F.relu(torch.tensor(1 - epoch / forcing_period))
                    )
                    .repeat(1, 1, sample.shape[2])
                    .cuda(args.local_rank)
                )

                # next_data = next_data_mask * label + (1-next_data_mask) * sample
                next_data = (
                    next_data_mask * label_diff + (1 - next_data_mask) * sample
                ) / g_const + data
            else:
                next_data = label

            if args.eval_type == "p":
                data = torch.cat((next_data, data[:, :, :2] - next_data), dim=-1)
            elif args.eval_type == "v":
                data = torch.cat((data[:, :, :2] + next_data, next_data), dim=-1)
            elif args.eval_type == "pv":
                data = next_data

        # optimizer.zero_grad()
        # total_train_loss.backward()
        # optimizer.step()

    scheduler.step(
        np.mean([train_losses_list[i].avg for i in range(len(train_losses_list))]),
        epoch,
    )
    return (
        [train_losses_list[i].avg for i in range(len(train_losses_list))],
        [train_losses_list[i].count for i in range(len(train_losses_list))],
    )


def test(test_loader, model, criterion, sampler, args):
    g_const = 1
    test_losses_list = []
    data_num = test_loader.dataset.test_data.shape[1]
    step_num = test_loader.dataset.test_labels.shape[1]
    for i in range(step_num + 1):
        test_losses_list.append(AverageMeter("Loss_" + str(i), ":.4e"))
    model.eval()

    with torch.no_grad():
        for i, (data, labels) in enumerate(test_loader):

            data = data.cuda(args.local_rank)
            R_const = (
                data[:, 0, :, -1]
                .unsqueeze(-1)
                .repeat(1, 1, data.shape[-2])
                .unsqueeze(-1)
            )  # one constant
            data = data[:, :, :, :-1]
            hidden, cell = model.module.initialize(data[:, 0])

            for n in range(data_num - 1):  # data.shape[1]
                hidden, cell = model(data[:, n], hidden, cell, R_const, test=False)
            data = data[:, -1]

            for n in range(step_num):  # labels.shape[1]
                label = labels.cuda(args.local_rank)[:, n]
                label_diff = (label - data) * g_const

                set1, set2, hidden, cell = model(data, hidden, cell, R_const, test=True)

                mu1, sig1, corr1, coef1 = set1
                mu2, sig2, corr2, coef2 = set2
                nll1, cv1 = criterion(
                    label_diff[:, :, :2],
                    mu1,
                    sig1,
                    corr1,
                    coef1,
                    loss_out=True,
                    cv_out=True,
                )  # position
                nll2, cv2 = criterion(
                    label_diff[:, :, 2:],
                    mu2,
                    sig2,
                    corr2,
                    coef2,
                    loss_out=True,
                    cv_out=True,
                )  # velocity

                test_loss = torch.mean(nll1 + nll2)
                test_losses_list[-1].update(test_loss.item(), np.prod(nll1.shape))
                test_losses_list[n].update(test_loss.item(), np.prod(nll1.shape))

                sample_pos = sampler(mu1, cv1)
                sample_vel = sampler(mu2, cv2)
                sample = torch.cat((sample_pos, sample_vel), dim=-1)

                if type(sample) == type(None):
                    print("tuzim")
                    with open("singular.pkl", "wb") as f:
                        pickle.dump(
                            (data, label, label_diff, mu1, sig1, corr1, nll1, L1), f
                        )
                    break
                next_data = sample.squeeze(-2).cuda(args.local_rank) / g_const + data

                if args.eval_type == "p":
                    data = torch.cat((next_data, data[:, :, :2] - next_data), dim=-1)
                elif args.eval_type == "v":
                    data = torch.cat((data[:, :, :2] + next_data, next_data), dim=-1)
                elif args.eval_type == "pv":
                    data = next_data

    return (
        [test_losses_list[i].avg for i in range(len(test_losses_list))],
        [test_losses_list[i].count for i in range(len(test_losses_list))],
    )


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
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
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


if __name__ == "__main__":
    print("started!")  # For test
    main()
