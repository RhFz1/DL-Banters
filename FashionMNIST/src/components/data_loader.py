import torch
import torchvision
from src.constants.transforms import train_transform, test_transform


def load_data(batch_size):

    assert isinstance(batch_size, int), "batch_size should be an integer"
    assert batch_size % 4 == 0, "batch_size should be divisible by 4"

    trainset = torchvision.datasets.FashionMNIST(root='./assets/data', train=True, download=True, transform=train_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    testset = torchvision.datasets.FashionMNIST(root='./assets/data', train=False, download=True, transform=test_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size // 4, shuffle=False)

    return trainloader, testloader

