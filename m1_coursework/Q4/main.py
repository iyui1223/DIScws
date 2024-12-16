import torch
import torch.nn as nn
import torch.optim as optim
from read_data import prepare_dataset_dataloaders
from construct_dataset import TwoimageDataset, TwoSeparateImageDataset
from models import LinearClassifier56, SplitAndSum


# Training and Evaluation for Single-Image Input (LinearClassifier56)
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
    with torch.no_grad():
        for batch in validate_loader:
            images, labels = batch["img"].to(device), batch["labels"].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    validation_accuracy = 100 * correct / total
    print(f"Validation Accuracy: {validation_accuracy:.2f}%")

    # Testing
    correct, total = 0, 0
    with torch.no_grad():
        for batch in test_loader:
            images, labels = batch["img"].to(device), batch["labels"].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_accuracy = 100 * correct / total
    print(f"Test Accuracy: {test_accuracy:.2f}%")

    return validation_accuracy, test_accuracy


# Training and Evaluation for Two-Image Input (SplitAndSum)
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
    with torch.no_grad():
        for batch in validate_loader:
            images1, images2, labels = batch["img1"].to(device), batch["img2"].to(device), batch["labels"].to(device)
            outputs = model(images1, images2)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    validation_accuracy = 100 * correct / total
    print(f"Validation Accuracy: {validation_accuracy:.2f}%")

    # Testing
    correct, total = 0, 0
    with torch.no_grad():
        for batch in test_loader:
            images1, images2, labels = batch["img1"].to(device), batch["img2"].to(device), batch["labels"].to(device)
            outputs = model(images1, images2)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_accuracy = 100 * correct / total
    print(f"Test Accuracy: {test_accuracy:.2f}%")

    return validation_accuracy, test_accuracy


# Main Loop
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
        val_acc, test_acc = train_and_evaluate_single(model56, train_loader, validate_loader, test_loader, device, epochs)
        results.append(("LinearClassifier56", dataset_size, val_acc, test_acc))

        # SplitAndSum (Two Images)
        train_set, validate_set, test_set, train_loader, validate_loader, test_loader = prepare_dataset_dataloaders(
            TwoSeparateImageDataset, dataset_size
        )
        model28 = SplitAndSum()
        val_acc, test_acc = train_and_evaluate_double(model28, train_loader, validate_loader, test_loader, device, epochs)
        results.append(("SplitAndSum", dataset_size, val_acc, test_acc))

    # Print Results
    print("\nSummary of Results:")
    print(f"{'Model':<20} {'Dataset Size':<15} {'Validation Acc (%)':<20} {'Test Acc (%)':<15}")
    for model_name, size, val_acc, test_acc in results:
        print(f"{model_name:<20} {size:<15} {val_acc:<20.2f} {test_acc:<15.2f}")


# Run the main function
if __name__ == "__main__":
    main()
