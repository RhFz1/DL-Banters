import torch
import torchvision.transforms.v2 as transforms


test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_transform = transforms.Compose([
                # Random horizontal flip
                transforms.RandomHorizontalFlip(p=0.2),
                transforms.RandomErasing(p=0.1),
                # Normalization
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])])