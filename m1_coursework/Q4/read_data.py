"""
4. Comparison of training process using different inputs
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from construct_dataset import TwoimageDataset, TwoSeparateImageDataset

# Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor()
])
mnist_data = datasets.MNIST(root="../mnist_data", train=True, download=True, transform=transform)

def prepare_dataset_dataloaders(Dataset, length):  # Flexible for any dataset
    train_set = Dataset(mnist_dataset=mnist_data, length=length, start=0, end=6)  # 60%
    validate_set = Dataset(mnist_dataset=mnist_data, length=length, start=6, end=8)  # 20%
    test_set = Dataset(mnist_dataset=mnist_data, length=length, start=8, end=10)  # 20%

    train_loader = DataLoader(train_set, batch_size=16, num_workers=2, shuffle=True)
    validate_loader = DataLoader(validate_set, batch_size=16, num_workers=2, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=16, num_workers=2, shuffle=True)

    return train_set, validate_set, test_set, train_loader, validate_loader, test_loader


