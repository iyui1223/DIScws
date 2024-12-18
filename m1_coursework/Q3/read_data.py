import numpy as np
from torchvision import datasets, transforms
import sys
import os
q1_2_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../Q1-2"))
sys.path.append(q1_2_path)
from construct_dataset import TwoimageDataset # from Q1

def prepare_data():
    """
    Prepare the 2-combined MNIST dataset for scikit-learn classifiers.
    Returns:
        X_train, X_val, y_train, y_val: Flattened feature vectors and labels for training/validation.
    """
    transform = transforms.Compose([transforms.ToTensor()])
    mnist_data = datasets.MNIST(root="../mnist_data", train=True, download=True, transform=transform)

    train_set = TwoimageDataset(mnist_dataset=mnist_data, length=1000, start=0, end=8) # 80%
    validate_set = TwoimageDataset(mnist_dataset=mnist_data, length=1000, start=8, end=10) # 20%

    def flatten_dataset(dataset):
        X, y = [], []
        # randomly pick two images, combine them, return label and flatten image for scikit-learn input
        i = 0
        for item in dataset: 
            X.append(item["img"].numpy().flatten())
            y.append(item["labels"])
            i += 1
            if i > 1000-2:
                break
        return np.array(X), np.array(y)

    X_train, y_train = flatten_dataset(train_set)
    X_val, y_val = flatten_dataset(validate_set)
    return X_train, X_val, y_train, y_val
