import torch
import torch.nn as nn
import numpy as np

def compute_conv_output_size(Lin,kernel_size,stride=1,padding=0,dilation=1):
    return int(np.floor((Lin+2*padding-dilation*(kernel_size-1)-1)/float(stride)+1))

class Conv_Net(nn.Module):
    def __init__(self, inputsize, task_info):
        super().__init__()
        
        ncha,size,_=inputsize
        self.task_info = task_info
        
        self.conv1 = nn.Conv2d(ncha,32,kernel_size=3,padding=1)
        s = compute_conv_output_size(size,3, padding=1) # 32
        self.conv2 = nn.Conv2d(32,32,kernel_size=3,padding=1)
        s = compute_conv_output_size(s,3, padding=1) # 32
        s = s//2 # 16
        self.conv3 = nn.Conv2d(32,64,kernel_size=3,padding=1)
        s = compute_conv_output_size(s,3, padding=1) # 16
        self.conv4 = nn.Conv2d(64,64,kernel_size=3,padding=1)
        s = compute_conv_output_size(s,3, padding=1) # 16
        s = s//2 # 8
        self.conv5 = nn.Conv2d(64,128,kernel_size=3,padding=1)
        s = compute_conv_output_size(s,3, padding=1) # 8
        self.conv6 = nn.Conv2d(128,128,kernel_size=3,padding=1)
        s = compute_conv_output_size(s,3, padding=1) # 8
        s = s//2 # 4
        self.fc1 = nn.Linear(s*s*128,256) # 2048
        self.drop1 = nn.Dropout(0.25)
        self.drop2 = nn.Dropout(0.5)
        self.MaxPool = torch.nn.MaxPool2d(2)
        self.last=torch.nn.ModuleList()
        
        for t,n in self.task_info:
            self.last.append(torch.nn.Linear(256,n))
        self.relu = torch.nn.ReLU()

    def forward(self, x, avg_act = False):
        act1=self.relu(self.conv1(x))
        act2=self.relu(self.conv2(act1))
        h=self.drop1(self.MaxPool(act2))
        act3=self.relu(self.conv3(h))
        act4=self.relu(self.conv4(act3))
        h=self.drop1(self.MaxPool(act4))
        act5=self.relu(self.conv5(h))
        act6=self.relu(self.conv6(act5))
        h=self.drop1(self.MaxPool(act6))
        h=h.view(x.shape[0],-1)
        act7 = self.relu(self.fc1(h))
        h = self.drop2(act7)
        y = []
        for t,i in self.task_info:
            y.append(self.last[t](h))
        
        return y

    
class MLP_Net(torch.nn.Module):

    def __init__(self, inputsize, task_info, unitN = 400):
        super(MLP_Net,self).__init__()

        size,_=inputsize
        self.task_info=task_info
        self.relu=torch.nn.ReLU()
        self.drop=torch.nn.Dropout(0.5)
        self.fc1=torch.nn.Linear(size*size,unitN)
        self.fc2=torch.nn.Linear(unitN,unitN)
        self.relu = torch.nn.ReLU()
        
        self.last=torch.nn.ModuleList()
        for t,n in self.task_info:
            self.last.append(torch.nn.Linear(unitN,n))

    def forward(self,x):
        h=x.view(x.size(0),-1)
        h=self.drop(self.relu(self.fc1(h)))
        h=self.drop(self.relu(self.fc2(h)))
        y = []
        for t,i in self.task_info:
            y.append(self.last[t](h))
        
        return y    

def conv_net(task_info):
    inputsize = (3, 32, 32)
    model = Conv_Net(inputsize, task_info)
    
    return model


def MLP(task_info):
    inputsize = (28, 28)
    model = MLP_Net(inputsize, task_info)
    
    return model