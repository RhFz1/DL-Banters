import torch.nn as nn

'''

In this file you will write the model definition for a convolutional neural network. 

Please only complete the model definition and do not include any training code.

The model should be a convolutional neural network, that accepts 28x28 grayscale images as input, and outputs a tensor of size 10.
The number of layers/kernels, kernel sizes and strides are up to you. 

Please refer to the following for more information about convolutions, pooling, and convolutional layers in PyTorch:

    - https://deeplizard.com/learn/video/YRhxdVk_sIs
    - https://deeplizard.com/resource/pavq7noze2
    - https://deeplizard.com/resource/pavq7noze3
    - https://setosa.io/ev/image-kernels/
    - https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html


Whether you need to normalize outputs using softmax depends on your choice of loss function. PyTorch documentation is available at
https://pytorch.org/docs/stable/index.html, and will specify whether a given loss funciton requires normalized outputs or not.

'''

# Enhanced CNN Model
class Conv_Net(nn.Module):
    def __init__(self):
        super(Conv_Net, self).__init__()
        # First convolutional block

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1) # shape [1, 32, 26, 26]
        self.bn1 = nn.BatchNorm2d(32)  # BatchNorm after conv1
        self.act1 = nn.GELU()
        self.pool1 = nn.MaxPool2d(2) # shape [1, 32, 13, 13]
        self.drop1 = nn.Dropout2d(0.3)
        
        # Second convolutional block
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # shape [1, 64, 11, 11]
        self.bn2 = nn.BatchNorm2d(64)  # BatchNorm after conv2
        self.act2 = nn.GELU()
        self.pool2 = nn.MaxPool2d(2) # shape [1, 64, 5, 5]
        self.drop2 = nn.Dropout2d(0.25)

        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.bn4 = nn.BatchNorm1d(512)  # BatchNorm after fc1
        self.act4 = nn.GELU()
        self.drop4 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        # Convolutional blocks
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.drop1(self.pool1(self.act1(x)))
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.drop2(self.pool2(self.act2(x)))
        
        # Flatten and fully connected layers
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.bn4(x)
        x = self.drop4(self.act4(x))
        x = self.fc2(x)
        return x

