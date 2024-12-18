import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from read_data import prepare_dataset_dataloaders
from construct_dataset import TwoimageDataset, TwoSeparateImageDataset
from models import LinearClassifier56, SplitAndSum

def analyze_probabilities(outputs, labels):
    """
    Analyze classifier probabilities.
    Args:
        outputs (Tensor): Raw model outputs (logits).
        labels (Tensor): True labels for the batch.
    Returns:
        avg_entropy (float): Average entropy across the batch.
        correct_probs (List[float]): Probabilities for the correct class.
    """
    probabilities = F.softmax(outputs, dim=1)  # Convert logits to probabilities
    entropies = -torch.sum(probabilities * torch.log(probabilities + 1e-9), dim=1)  # Compute entropy per example
    avg_entropy = torch.mean(entropies).item()  # Average entropy for the batch

    correct_probs = probabilities[torch.arange(len(labels)), labels].tolist()  # Probabilities of correct class

    return avg_entropy, correct_probs

def train_and_evaluate_single(model, train_loader, validate_loader, test_loader, device, epochs=20):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.to(device)

    # Training Loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            images, labels = batch["img"].to(device), batch["labels"].to(device)

            # Zero gradients, backward pass, optimization
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader):.4f}")

    # Validation
    model.eval()
    correct, total = 0, 0
    avg_entropy_list = []
    correct_probabilities = []
    with torch.no_grad():
        for batch in validate_loader:
            images, labels = batch["img"].to(device), batch["labels"].to(device)
            outputs = model(images)

            # Analyze probabilities
            avg_entropy, correct_probs = analyze_probabilities(outputs, labels)
            avg_entropy_list.append(avg_entropy)
            correct_probabilities.extend(correct_probs)

            # Standard accuracy computation
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    validation_accuracy = 100 * correct / total
    validation_avg_entropy = np.mean(avg_entropy_list)
    print(f"Validation Accuracy: {validation_accuracy:.2f}%, Avg Entropy: {validation_avg_entropy:.4f}")

    # Testing
    correct, total = 0, 0
    avg_entropy_list = []
    correct_probabilities = []
    with torch.no_grad():
        for batch in test_loader:
            images, labels = batch["img"].to(device), batch["labels"].to(device)
            outputs = model(images)

            # Analyze probabilities
            avg_entropy, correct_probs = analyze_probabilities(outputs, labels)
            avg_entropy_list.append(avg_entropy)
            correct_probabilities.extend(correct_probs)

            # Standard accuracy computation
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_accuracy = 100 * correct / total
    test_avg_entropy = np.mean(avg_entropy_list)
    print(f"Test Accuracy: {test_accuracy:.2f}%, Avg Entropy: {test_avg_entropy:.4f}")

    return validation_accuracy, validation_avg_entropy, test_accuracy, test_avg_entropy

def train_and_evaluate_double(model, train_loader, validate_loader, test_loader, device, epochs=20):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.to(device)

    # Training Loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            images1, images2, labels = batch["img1"].to(device), batch["img2"].to(device), batch["labels"].to(device)

            # Zero gradients, backward pass, optimization
            optimizer.zero_grad()
            outputs = model(images1, images2)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader):.4f}")

    # Validation
    model.eval()
    correct, total = 0, 0
    avg_entropy_list = []
    correct_probabilities = []
    with torch.no_grad():
        for batch in validate_loader:
            images1, images2, labels = batch["img1"].to(device), batch["img2"].to(device), batch["labels"].to(device)
            outputs = model(images1, images2)

            # Analyze probabilities
            avg_entropy, correct_probs = analyze_probabilities(outputs, labels)
            avg_entropy_list.append(avg_entropy)
            correct_probabilities.extend(correct_probs)

            # Standard accuracy computation
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    validation_accuracy = 100 * correct / total
    validation_avg_entropy = np.mean(avg_entropy_list)
    print(f"Validation Accuracy: {validation_accuracy:.2f}%, Avg Entropy: {validation_avg_entropy:.4f}")

    # Testing
    correct, total = 0, 0
    avg_entropy_list = []
    correct_probabilities = []
    with torch.no_grad():
        for batch in test_loader:
            images1, images2, labels = batch["img1"].to(device), batch["img2"].to(device), batch["labels"].to(device)
            outputs = model(images1, images2)

            # Analyze probabilities
            avg_entropy, correct_probs = analyze_probabilities(outputs, labels)
            avg_entropy_list.append(avg_entropy)
            correct_probabilities.extend(correct_probs)

            # Standard accuracy computation
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_accuracy = 100 * correct / total
    test_avg_entropy = np.mean(avg_entropy_list)
    print(f"Test Accuracy: {test_accuracy:.2f}%, Avg Entropy: {test_avg_entropy:.4f}")

    return validation_accuracy, validation_avg_entropy, test_accuracy, test_avg_entropy

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 50
    dataset_sizes = [50, 100, 500, 1000, 1500, 2000]

    results = []

    for dataset_size in dataset_sizes:
        print(f"\nDataset Size: {dataset_size}\n")

        # LinearClassifier56 (Single Image)
        train_set, validate_set, test_set, train_loader, validate_loader, test_loader = prepare_dataset_dataloaders(
            TwoimageDataset, dataset_size
        )
        model56 = LinearClassifier56()
        val_acc, val_entropy, test_acc, test_entropy = train_and_evaluate_single(model56, train_loader, validate_loader, test_loader, device, epochs)
        results.append(("LinearClassifier56", dataset_size, val_acc, val_entropy, test_acc, test_entropy))

        # SplitAndSum (Two Images)
        train_set, validate_set, test_set, train_loader, validate_loader, test_loader = prepare_dataset_dataloaders(
            TwoSeparateImageDataset, dataset_size
        )
        model28 = SplitAndSum()
        val_acc, val_entropy, test_acc, test_entropy = train_and_evaluate_double(model28, train_loader, validate_loader, test_loader, device, epochs)
        results.append(("SplitAndSum", dataset_size, val_acc, val_entropy, test_acc, test_entropy))

    # Print Results
    print("\nSummary of Results:")
    print(f"{'Model':<20} {'Dataset Size':<15} {'Validation Acc (%)':<20} {'Validation Entropy':<20} {'Test Acc (%)':<15} {'Test Entropy':<15}")
    for model_name, size, val_acc, val_entropy, test_acc, test_entropy in results:
        print(f"{model_name:<20} {size:<15} {val_acc:<20.2f} {val_entropy:<20.4f} {test_acc:<15.2f} {test_entropy:<15.4f}")

if __name__ == "__main__":
    main()
