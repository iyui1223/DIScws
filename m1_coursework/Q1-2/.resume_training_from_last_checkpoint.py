'''
A make-shift code to resume computatinally heavy tuning process, 
from the checkpoint provided by last optuna tuning cycle.
as it does not remember the previous parameter tuning process, 
it is only useful if you decided to change the tuning framework.
So to speak, it is for hyper-hyper parameter tuning.
'''

import optuna
from construct_dataset import TwoimageDataset
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from Q2_tuning import DenseNet

# Load the model checkpoint
checkpoint = torch.load("best_model_checkpoint.pth")

# Reinitialize the model with saved hyperparameters
best_hyperparams = checkpoint['hyperparameters']
model = DenseNet(
    input_size=56 * 28,
    hidden1=best_hyperparams['hidden1'],
    hidden2=best_hyperparams['hidden2'],
    output_size=19
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# restart optimization from checkpoint
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
