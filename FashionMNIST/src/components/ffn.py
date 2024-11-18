import torch
import torchvision
import torchvision.transforms.v2 as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

'''

In this file you will write the model definition for a feedforward neural network. 

Please only complete the model definition and do not include any training code.

The model should be a feedforward neural network, that accepts 784 inputs (each image is 28x28, and is flattened for input to the network)
and the output size is 10. Whether you need to normalize outputs using softmax depends on your choice of loss function.

PyTorch documentation is available at https://pytorch.org/docs/stable/index.html, and will specify whether a given loss funciton 
requires normalized outputs or not.

'''


class FF_Net(nn.Module):
    def __init__(self):
        super(FF_Net, self).__init__()
        # First block
        self.ll1 = nn.Linear(28*28, 1024, bias=False)
        self.bn1 = nn.BatchNorm1d(1024)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(0.3)
        
        # Second block
        self.ll2 = nn.Linear(1024, 512, bias=False)
        self.bn2 = nn.BatchNorm1d(512)
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(0.2)
        
        # Third block
        self.ll3 = nn.Linear(512, 256, bias=False)
        self.bn3 = nn.BatchNorm1d(256)
        self.act3 = nn.GELU()
        self.drop3 = nn.Dropout(0.15)
        
        # Output layer
        self.ll4 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.drop1(self.act1(self.bn1(self.ll1(x))))
        x = self.drop2(self.act2(self.bn2(self.ll2(x))))
        x = self.drop3(self.act3(self.bn3(self.ll3(x))))
        x = self.ll4(x)
        return x