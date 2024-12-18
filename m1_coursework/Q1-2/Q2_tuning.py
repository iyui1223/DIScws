import optuna
from construct_dataset import TwoimageDataset
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim


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


def objective(trial):
    # Hyperparameter space
    hidden1 = trial.suggest_int("hidden1", 64, 256)
    hidden2 = trial.suggest_int("hidden2", 32, 128)
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-1)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    l1_lambda = trial.suggest_loguniform("l1_lambda", 1e-6, 1e-2)

    # Data preparation
    transform = transforms.Compose([transforms.ToTensor()])
    mnist_data = datasets.MNIST(root="../mnist_data", train=True, download=True, transform=transform)

    train_set = TwoimageDataset(mnist_dataset=mnist_data, length=1000, start=0, end=6)
    validate_set = TwoimageDataset(mnist_dataset=mnist_data, length=333, start=6, end=8)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    validate_loader = DataLoader(validate_set, batch_size=batch_size, shuffle=False)

    # Model, criterion, optimizer
    model = DenseNet(input_size=56 * 28, hidden1=hidden1, hidden2=hidden2, output_size=19)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    epochs = 120
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            images, labels = batch["img"].to(device), batch["labels"].to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            # L1 Regularization
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            loss += l1_lambda * l1_norm  # Add L1 penalty to the loss

            loss.backward()
            optimizer.step()

    # Validation loop
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

    accuracy = 100 * correct / total
    return accuracy

# Create the Optuna study
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

# Print the best hyperparameters
print("Best hyperparameters:")
print(study.best_params)

# Save the study results
study.trials_dataframe().to_csv("optuna_study_results.csv", index=False)
