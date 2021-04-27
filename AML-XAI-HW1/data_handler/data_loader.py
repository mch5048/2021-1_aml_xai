import copy
import logging

import numpy as np
import torch
import torch.utils.data as td
from sklearn.utils import shuffle
from PIL import Image
from torch.autograd import Variable
import torchvision.transforms.functional as trnF


class Loader(td.Dataset):
    def __init__(self, data, labels, transform=None, data_dict=None):
        
        self.data = data
        self.labels = labels
        self.transform=transform
        self.data_dict = data_dict

    def __len__(self):
        return self.labels.shape[0]
    
    def __getitem__(self, index):
        
        img = self.data[index]
        try:
            # For CIFAR100
            img = Image.fromarray(img)
        except:
            # For MNIST
            img = Image.fromarray(img.numpy(), mode='L')
        if self.transform is not None:
            img = self.transform(img)

        return img, self.labels[index]

def make_ContinualLoaders(data, labels, task_info, transform=None, shuffle_idx=None, data_dict=None):
    if shuffle_idx is not None:
        labels = shuffle_idx[labels]
    sort_index = np.argsort(labels)
    data = data[sort_index]
    labels = np.array(labels)
    labels = labels[sort_index]
    
    loaders = []
    start = 0
    for t, ncla in task_info:
        start_idx = np.argmin(labels<start) # start data index
        end_idx = np.argmax(labels>(start+ncla-1)) # end data index
        if end_idx == 0:
            end_idx = data.shape[0]
        
        loaders.append(Loader(data[start_idx:end_idx], 
                                    labels[start_idx:end_idx]%ncla, 
                                    transform=transform, 
                                    data_dict=data_dict))
        
        start += ncla
    
    return loaders