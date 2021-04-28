from __future__ import print_function

import copy
import logging

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.modules.loss import CrossEntropyLoss
import torch.utils.data as td
from PIL import Image
from tqdm import tqdm
import trainer

from torch import autograd
from torch.autograd import Variable


import networks

class Trainer(trainer.GenericTrainer):
    def __init__(self, model, args, optimizer, evaluator, task_info):
        super().__init__(model, args, optimizer, evaluator, task_info)
        
        self.lamb=args.lamb
        

    def train(self, train_loader, test_loader, t, device = None):
        
        self.device = device
        lr = self.args.lr
        self.setup_training(lr)
        # Do not update self.t
        if t>0: # update fisher before starting training new task
            self.update_frozen_model()
            self.update_fisher()
        
        # Now, you can update self.t
        self.t = t # It denotes current task
        
        self.train_iterator = torch.utils.data.DataLoader(train_loader, batch_size=self.args.batch_size, shuffle=True)
        self.test_iterator = torch.utils.data.DataLoader(test_loader, 100, shuffle=False)
        self.fisher_iterator = torch.utils.data.DataLoader(train_loader, batch_size=20, shuffle=True)
        
        
        for epoch in range(self.args.nepochs):
            self.model.train()
            self.update_lr(epoch, self.args.schedule)
            for samples in tqdm(self.train_iterator):
                data, target = samples
                data, target = data.to(device), target.to(device)
                batch_size = data.shape[0]

                output = self.model(data)[t]
                loss_CE = self.criterion(output,
                target)
                self.optimizer.zero_grad()
                (loss_CE).backward()
                self.optimizer.step()

            train_loss,train_acc = self.evaluator.evaluate(self.model, self.train_iterator, t, self.device)
            num_batch = len(self.train_iterator)
            print('| Epoch {:3d} | Train: loss={:.3f}, acc={:5.1f}% |'.format(epoch+1,train_loss,100*train_acc),end='')
            test_loss,test_acc=self.evaluator.evaluate(self.model, self.test_iterator, t, self.device)
            print(' Test: loss={:.3f}, acc={:5.1f}% |'.format(test_loss,100*test_acc),end='')
            print()
        
    def criterion(self, output, targets):
        """
        Arguments: output (The output logit of self.model), targets (Ground truth label)
        Return: loss function for the regularization-based continual learning
        
        For the hyperparameter on regularization, please use self.lamb
        """

        loss =  nn.CrossEntropyLoss()(output, targets)
        if self.t >= 1:
            losses = []
            try:
                for n, p in self.model.named_parameters():
                    if p.grad == None:
                        continue
                    # retrieve the consolidated mean and fisher information.
                    n = n.replace('.', '__')
                    mean = getattr(self.model, '{}_mean'.format(n))
                    fisher = getattr(self.model, '{}_fisher'.format(n))
                    # wrap mean and fisher in variables.
                    mean = Variable(mean)
                    fisher = Variable(fisher)
                    losses.append((fisher * (p - mean)**2).sum())
                loss += (self.lamb/2)*sum(losses)
            except AttributeError:
                loss += Variable(torch.zeros(1).sum())
        return loss
    
    def compute_diag_fisher(self):
        """
        Arguments: None. Just use global variables (self.model, self.criterion, ...)
        Return: Diagonal Fisher matrix. 
        
        This function will be used in the function 'update_fisher'
        """

        self.fisher_iterator

        param_names = []
        loglikelihood_grads = {}
        for data, label in self.fisher_iterator:
            data = Variable(data)
            label = Variable(label)
            loglikelihood = CrossEntropyLoss()(self.model(data)[self.t], label)
            loglikelihood.backward()
            for n, p in self.model.named_parameters():
                n = n.replace('.', '__')
                if p.grad == None:
                    continue
                loglikelihood_grads[n] = (loglikelihood_grads.get(n, 0) + (p.grad ** 2)).mean(0)
                param_names.append(n)

        fisher_diagonals = [g for g in loglikelihood_grads.values()]
        dict = {n: f.detach() for n, f in zip(param_names, fisher_diagonals)}
        return dict
        
    def update_fisher(self):
        
        """
        Arguments: None. Just use global variables (self.model, self.fisher, ...)
        Return: None. Just update the global variable self.fisher
        Use 'compute_diag_fisher' to compute the fisher matrix
        """
        
        self.fisher = self.compute_diag_fisher()
        
        for n, p in self.model.named_parameters():
            if p.grad == None:
                continue
            n = n.replace('.', '__')
            self.model.register_buffer('{}_mean'.format(n), p.data.clone())
            self.model.register_buffer('{}_fisher'
                                 .format(n), self.fisher[n].data.clone())

