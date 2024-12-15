import torch
from torch.utils.data import Dataset
import random

class TwoimageDataset(Dataset):
    def __init__(self, mnist_dataset, transform=None, length=1000, start=0, end=10, seed=42):
        """
        Custom dataset for pairing two random MNIST images, concatenating them, and creating a label.

        Args:
            mnist_dataset: The base MNIST dataset.
            transform: Transformations to apply to images.
            length: Number of pairs to generate.
            start: Starting proportion (e.g., 0 for 0%).
            end: Ending proportion (e.g., 6 for 60%).
            seed: Random seed for reproducibility.
        """
        self.mnist_dataset = mnist_dataset
        self.transform = transform
        self.length = length
        self.start = start
        self.end = end

        # Initialize random seed
        random.seed(seed)
        self.rng = random.Random(seed)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        """
        Fetch two random MNIST images, concatenate them vertically, and create a label.

        Returns:
            A dictionary containing:
            - 'img': Concatenated image (Tensor).
            - 'labels': Sum of the two MNIST digit labels.
        """
        # Define index range for the dataset slice
        istart = int(self.start / 10 * len(self.mnist_dataset))
        iend = int(self.end / 10 * len(self.mnist_dataset)) - 1

        # Randomly select two indices
        idx1 = self.rng.randint(istart, iend)
        idx2 = self.rng.randint(istart, iend)

        # Fetch images and labels
        image1, label1 = self.mnist_dataset[idx1]
        image2, label2 = self.mnist_dataset[idx2]

        # Apply transformations if provided
        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)

        # Concatenate images vertically
        image = torch.cat((image1, image2), dim=1)

        # Compute label
        label = label1 + label2

        return {
            "img": image,
            "labels": label
        }


class TwoSeparateImageDataset(Dataset):
    def __init__(self, mnist_dataset, transform=None, length=1000, start=0, end=10, seed=42):
        """
        Custom dataset for pairing two random MNIST images and creating a label as summation of two labels.

        Args:
            mnist_dataset: The base MNIST dataset.
            transform: Transformations to apply to images.
            length: Number of pairs to generate.
            start: Starting proportion (e.g., 0 for 0%).
            end: Ending proportion (e.g., 6 for 60%).
            seed: Random seed for reproducibility.
        """
        self.mnist_dataset = mnist_dataset
        self.transform = transform
        self.length = length
        self.start = start
        self.end = end

        # Initialize random seed
        random.seed(seed)
        self.rng = random.Random(seed)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        """
        Fetch two random MNIST images, and create a label as their sum.

        Returns:
            A dictionary containing:
            - 'img1': First image (Tensor).
            - 'img2': Second image (Tensor).
            - 'labels': Sum of the two MNIST digit labels.
        """
        # Define index range for the dataset slice
        istart = int(self.start / 10 * len(self.mnist_dataset))
        iend = int(self.end / 10 * len(self.mnist_dataset)) - 1

        # Randomly select two indices
        idx1 = self.rng.randint(istart, iend)
        idx2 = self.rng.randint(istart, iend)

        # Fetch images and labels
        image1, label1 = self.mnist_dataset[idx1]
        image2, label2 = self.mnist_dataset[idx2]

        # Apply transformations if provided
        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)

        # Compute label
        label = label1 + label2

        return {
            "img1": image1,
            "img2": image2,
            "labels": label
        }
