"""
2. Perform prediction using dense neural network.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from construct_dataset import TwoimageDataset

# Define the model
class DenseNet(nn.Module):
    def __init__(self):
        super(DenseNet, self).__init__()
        self.dense_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(56 * 28, 128),  # Input: Flattened 56x28 image
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 19)  # Output: 19 classes (0-18)
        )

    def forward(self, x):
        # Input shape: [batch_size, 1, 56, 28]
        return self.dense_net(x)


# Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor()
])
mnist_data = datasets.MNIST(root="mnist_data", train=True, download=True, transform=transform)

train_set = TwoimageDataset(mnist_dataset=mnist_data, length=1000, start=0, end=6)  # 60%
validate_set = TwoimageDataset(mnist_dataset=mnist_data, length=333, start=6, end=8)  # 20%
test_set = TwoimageDataset(mnist_dataset=mnist_data, length=333, start=8, end=10)  # 20%

train_loader = DataLoader(train_set, batch_size=16, num_workers=2, shuffle=True)
validate_loader = DataLoader(validate_set, batch_size=16, num_workers=2, shuffle=True)
test_loader = DataLoader(test_set, batch_size=16, num_workers=2, shuffle=True)

# Test the DataLoader
batch = next(iter(train_loader))
print(batch["img"].shape, batch["labels"].shape)

# Initialize the model
model = DenseNet()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 100
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for batch in train_loader:
        images, labels = batch["img"].to(device), batch["labels"].to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader):.4f}")

# Validate the model
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch in validate_loader:
        images, labels = batch["img"].to(device), batch["labels"].to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Validation Accuracy: {100 * correct / total:.2f}%")

# Test the model
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch in test_loader:
        images, labels = batch["img"].to(device), batch["labels"].to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")

# Save the model checkpoint
checkpoint_path = "model_checkpoint.pth"

torch.save({
    'epoch': epochs,  # Save the final epoch number
    'model_state_dict': model.state_dict(),  # Save the model weights
    'optimizer_state_dict': optimizer.state_dict(),  # Save the optimizer state
    'loss': running_loss / len(train_loader),  # Save the final loss
}, checkpoint_path)

print(f"Checkpoint saved to {checkpoint_path}")

