import pandas as pd

# Load the study results
study_results = pd.read_csv("optuna_study_results.csv")

# Get the best trial
best_trial = study_results.iloc[46 - 1]  # 46 was the best

# Extract hyperparameters
hidden1 = int(best_trial["params_hidden1"])
hidden2 = int(best_trial["params_hidden2"])
lr = float(best_trial["params_lr"])
batch_size = int(best_trial["params_batch_size"])
l1_lambda = float(best_trial["params_l1_lambda"])

from construct_dataset import TwoimageDataset
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim

# Data preparation
transform = transforms.Compose([transforms.ToTensor()])
mnist_data = datasets.MNIST(root="mnist_data", train=True, download=True, transform=transform)

train_set = TwoimageDataset(mnist_dataset=mnist_data, length=1000, start=0, end=6)
validate_set = TwoimageDataset(mnist_dataset=mnist_data, length=333, start=6, end=8)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
validate_loader = DataLoader(validate_set, batch_size=batch_size, shuffle=False)

# Define the model
class DenseNet(nn.Module):
    def __init__(self, input_size, hidden1, hidden2, output_size):
        super(DenseNet, self).__init__()
        self.dense_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, output_size)
        )

    def forward(self, x):
        return self.dense_net(x)


# Initialize the model
model = DenseNet(input_size=56 * 28, hidden1=hidden1, hidden2=hidden2, output_size=19)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Training loop
epochs = 300 
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for batch in train_loader:
        images, labels = batch["img"].to(device), batch["labels"].to(device)
        optimizer.zero_grad()
        outputs = model(images)

        # Calculate loss and L1 regularization
        loss = criterion(outputs, labels)
        l1_norm = sum(p.abs().sum() for p in model.parameters())
        loss += l1_lambda * l1_norm  # Add L1 penalty

        loss.backward()
        optimizer.step()

        # Track running loss and accuracy
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    # Calculate training accuracy
    train_accuracy = 100 * correct_train / total_train

    # Validation phase
    model.eval()
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for batch in validate_loader:
            images, labels = batch["img"].to(device), batch["labels"].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    # Calculate validation accuracy
    val_accuracy = 100 * correct_val / total_val

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader):.4f}, "
          f"Train Accuracy: {train_accuracy:.2f}%, Validation Accuracy: {val_accuracy:.2f}%")

# Save the model checkpoint
checkpoint_path = "best_model_checkpoint.pth"

torch.save({
    'model_state_dict': model.state_dict(),  # Save the model weights
    'optimizer_state_dict': optimizer.state_dict(),  # Save the optimizer state
    'hyperparameters': {
        'hidden1': hidden1,
        'hidden2': hidden2,
        'lr': lr,
        'batch_size': batch_size,
        'l1_lambda': l1_lambda
    }
}, checkpoint_path)

print(f"Best model checkpoint saved to {checkpoint_path}")
