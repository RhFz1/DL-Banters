import torch
import torchvision
from src.components.constants.transforms import train_transform, test_transform





def load_data(batch_size):

    trainset = torchvision.datasets.FashionMNIST(root='./assets/data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    testset = torchvision.datasets.FashionMNIST(root='./assets/data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size // 4, shuffle=False)

