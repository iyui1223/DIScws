import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import random
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt
import os


class TwoimageDataset(Dataset):
    def __init__(self, mnist_dataset, transform=None, length=1000, start=0, end=10):
        """
        Custom dataset for pairing two random MNIST images, concatenating them, and creating a label.

        Args:
            mnist_dataset: The base MNIST dataset.
            length: Number of pairs to generate.
            start: Starting proportion (e.g., 0 for 0%).
            end: Ending proportion (e.g., 6 for 60%).
        """
        self.mnist_dataset = mnist_dataset
        self.transform = transform
        self.length = length
        self.start = start
        self.end = end

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
        idx1 = random.randint(istart, iend)
        idx2 = random.randint(istart, iend)

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


# Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor()
])
mnist_data = datasets.MNIST(root="../mnist_data", train=True, download=True, transform=transform)

train_set = TwoimageDataset(mnist_dataset=mnist_data, length=1000, start=0, end=6)  # 60%
validate_set = TwoimageDataset(mnist_dataset=mnist_data, length=333, start=6, end=8)  # 20%
test_set = TwoimageDataset(mnist_dataset=mnist_data, length=333, start=8, end=10)  # 20%

train_loader = DataLoader(train_set, batch_size=16, num_workers=2, shuffle=True)
validate_loader = DataLoader(validate_set, batch_size=16, num_workers=2, shuffle=True)
test_loader = DataLoader(test_set, batch_size=16, num_workers=2, shuffle=True)

# Test the DataLoader
batch = next(iter(train_loader))
print(batch["img"].shape, batch["labels"].shape)

# Create an output directory for saving images
output_dir = "."
os.makedirs(output_dir, exist_ok=True)

# Save concatenated images
for i in range(5):
    img_tensor = batch["img"][i]
    label = batch["labels"][i].item()

    # Convert tensor to PIL image
    img = to_pil_image(img_tensor)

    # Save the image using Matplotlib
    plt.figure()
    plt.imshow(img, cmap="gray")
    plt.axis("off")
    plt.title(f"Label (Sum): {label}")

    # Save to PNG file
    output_path = os.path.join(output_dir, f"pair_{i + 1}_label_{label}.png")
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    print(f"Saved Pair {i + 1}: Label (Sum): {label} as {output_path}")
