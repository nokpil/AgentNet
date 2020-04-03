
# coding: utf-8

# ## Data Process for ModAdd data

# In[1]:


import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
from PIL import Image
from torch.utils.data.dataset import Dataset
from numpy import binary_repr
import os
import pickle
import collections
from copy import deepcopy

def DataGen(file_name, rule, seq_len, sample_len, train):
    
    if train:
        file_name = file_name+'_train.txt'
    else:
        file_name = file_name+'_test.txt'
     
    f = open(file_name,'w')
    for i in range(int(sample_len)):
        x = rule(seq_len)
        for k in x:
            f.write(str(k)+' ')
        f.write('\n')
    f.close() 
    
    
def DataGen_cont(file_name, rule, seq_len, sample_len, train):
    
    if train:
        file_name = file_name+'_train.txt'
    else:
        file_name = file_name+'_test.txt'
     
    f = open(file_name,'w')
    x = rule(sample_len)
    for d in x:
        for k in d:
            f.write(str(k)+' ')
        f.write('\n')
    f.close() 
    

def DataGen_Image(file_name, X, train):
    
    if train:
        file_name = file_name+'_train.pkl'
    else:
        file_name = file_name+'_test.pkl'
    
    with open(file_name, 'rb') as f:
        pickle.dump(X, f)

        
class Customset(Dataset):
    
    def __init__(self, file_name, train=True, transform=None, target_transform=None):
        self.file_name = file_name
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        
        if self.train:
            with open(file_name + '_train.pkl', 'rb') as f:
                data = pickle.load(f)
                self.train_data = torch.FloatTensor(data['Image'])
                self.train_labels = torch.FloatTensor(data['Label'])
        else:
            with open(file_name + '_test.pkl', 'rb') as f:
                data = pickle.load(f)
                self.test_data = torch.FloatTensor(data['Image'])
                self.test_labels = torch.FloatTensor(data['Label'])
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)
        

class Customset_con(Dataset):
    
    def __init__(self, file_name, train=True, transform=None, target_transform=None):
        self.file_name = file_name
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        
        if self.train:
            with open(file_name + '_train.pkl', 'rb') as f:
                data = pickle.load(f)
                self.train_data = torch.FloatTensor(data['Image'])
                self.train_labels = torch.FloatTensor(data['Label'])
        else:
            with open(file_name + '_test.pkl', 'rb') as f:
                data = pickle.load(f)
                self.test_data = torch.FloatTensor(data['Image'])
                self.test_labels = torch.FloatTensor(data['Label'])
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)
        
    
class Customset_direct(Dataset):
    def __init__(self, data, label):
        # Read the csv file
        self.data = torch.Tensor(data)
        self.labels = torch.Tensor(label)  # Second column is the labels
        
    def __getitem__(self, index):
        single_data = self.data[index]
        #single_data = self.data_info[index,:-1].unsqueeze(0)
        single_label = self.labels[index]
        return single_data, single_label

    def __len__(self):
        return len(self.data)
    
    
class CustomImageset(Dataset):
    
    def __init__(self, file_name, train=True, transform=None, target_transform=None, pp = 0):
        self.file_name = file_name
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.pp = pp
        
        if self.train:
            with open(file_name + '_train.pkl', 'rb') as f:
                data = pickle.load(f)
                self.train_data = 255*np.array(data['Image']).astype(np.uint8)
                self.train_labels = np.array(data['Label']).astype(np.int64)
        else:
            with open(file_name + '_test.pkl', 'rb') as f:
                data = pickle.load(f)
                self.test_data = 255*np.array(data['Image']).astype(np.uint8)
                self.test_labels = np.array(data['Label']).astype(np.int64)
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        if self.pp>0:
            s = img.shape[0]-self.pp
            img = np.tile(img, (3,3))[s:-s, s:-s]   
            
        img = Image.fromarray(img, mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

class ib_RandomSampler():

    def __init__(self, data_source, batch_size, category_size_list):
        self.data_source = data_source
        self.batch_size = batch_size
        self.category_size_list = category_size_list
        batch_num_list = []
        for i in range(len(category_size_list)):
            batch_num_list.append(int(self.category_size_list[i]/batch_size))
        self.batch_num_list = batch_num_list

        category_list = np.split(np.arange(len(data_source)), np.cumsum(self.category_size_list)[:-1])
        for i in range(len(category_list)):
            category_list[i] = category_list[i][:self.batch_num_list[i]*batch_size]
        self.category_list = category_list

    def __iter__(self):
        out_list = np.array([])
        bn_copy = deepcopy(self.batch_num_list)
        for i in range(len(self.category_list)):
            np.random.shuffle(self.category_list[i])
        while True:
            if np.sum(np.nonzero(bn_copy)[0]) == 0:
                break
            else:
                f = np.random.choice(np.nonzero(bn_copy)[0])
                cat = self.category_list[f][self.batch_size*(bn_copy[f]-1):self.batch_size*bn_copy[f]]
                out_list = np.concatenate((out_list,cat)).astype('int64')
                bn_copy[f]-=1
        
        return iter(out_list)

    def __len__(self):
        return len(self.data_source)
