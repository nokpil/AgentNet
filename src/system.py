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