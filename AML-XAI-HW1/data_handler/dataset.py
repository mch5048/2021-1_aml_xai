from torchvision import datasets, transforms
import torch
import numpy as np
from arguments import get_args
import math
import time
args = get_args()


class Dataset():
    '''
    Base class to reprenent a Dataset
    '''

    def __init__(self, classes, name, tasknum):
        self.classes = classes
        self.name = name
        self.tasknum = tasknum
        self.train_data = None
        self.test_data = None
        self.loader = None

        
class CIFAR100(Dataset):
    def __init__(self):
        super().__init__(100, "CIFAR100", args.tasknum)

        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]
        
        self.task_info = []
        for t in range(self.tasknum):
            self.task_info.append((t, self.classes // self.tasknum))
        
        self.train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            ])

        train_dataset = datasets.CIFAR100("dat", train=True, transform=self.train_transform, download=True)
        self.train_data = train_dataset.data
        self.train_labels = np.array(train_dataset.targets)
        test_dataset = datasets.CIFAR100("dat", train=False, transform=self.test_transform, download=True)
        self.test_data = test_dataset.data
        self.test_labels = np.array(test_dataset.targets)
        self.loader = None        

class MNIST(Dataset):
    def __init__(self):
        super().__init__(10, "MNIST", args.tasknum)

        mean = [0.1307]
        std = [0.3081]
        
        self.task_info = []
        for t in range(self.tasknum):
            self.task_info.append((t, self.classes // self.tasknum))
        
        self.train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        
        train_dataset = datasets.MNIST('dat', train=True, transform=self.train_transform, download=True)
        self.train_data = train_dataset.train_data
        self.train_labels = np.array(train_dataset.train_labels)
        test_dataset = datasets.MNIST("dat", train=False, transform=self.test_transform, download=True)
        self.test_data = test_dataset.test_data
        self.test_labels = np.array(test_dataset.test_labels)
        self.loader = None        

