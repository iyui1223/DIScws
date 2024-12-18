import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from construct_dataset import TwoimageDataset

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.cm as cm

def apply_tSNE(features, labels1, labels2, perplexity=30, title="t-SNE Visualization", save_path="temp.png"):
    """
    Apply t-SNE on input features and plot in 2D.
    Args:
        features: Feature dataset to apply t-SNE on (e.g., raw pixels or hidden layer outputs).
        labels1: First digit labels for the concatenated image.
        labels2: Second digit labels for the concatenated image.
        perplexity: t-SNE perplexity parameter.
        title: Title for the plot.
        save_path: Path to save the plot as a PNG file.
    """
    print(f"Running t-SNE with perplexity={perplexity} on {len(features)} samples.")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    reduced_features = tsne.fit_transform(features)

    # Generate a color map with 10 colors (one for each digit)
    colors = cm.get_cmap('tab10', 10) 

    plt.figure(figsize=(10, 8))
    for i in range(len(reduced_features)):
        plt.scatter(
            reduced_features[i, 0],
            reduced_features[i, 1],
            color=colors(labels1[i]),  # Color by labels1
            alpha=0.6,
            s=70
        )

    for i in range(len(reduced_features)):
        plt.scatter(
            reduced_features[i, 0],
            reduced_features[i, 1],
            color=colors(labels2[i]),  # Color by labels1
            s=20
        )


    # Add a legend for label1 colors
    for digit in range(10):
        plt.scatter([], [], color=colors(digit), label=f"{digit}")
    plt.legend(title="Digits", loc="best", fontsize=8)

    plt.title(title)
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.grid(True)
    plt.savefig(save_path) 
    print(f"t-SNE plot saved as {save_path}")
    plt.close()

def main():
    # Parameters
    perplexity = 10

    # Prepare the dataset and data loader
    transform = transforms.Compose([transforms.ToTensor()])
    mnist_dataset = datasets.MNIST(root="../mnist_data", train=True, download=True, transform=transform)

    # Create TwoimageDataset with ~200 samples
    dataset = TwoimageDataset(mnist_dataset=mnist_dataset, length=10-00, start=0, end=10)
    dataloader = DataLoader(dataset, batch_size=1000, shuffle=False)

    # Extract raw images and labels for t-SNE
    for batch in dataloader:
        raw_images = batch["img"]
        labels1 = batch["labels1"]
        labels2 = batch["labels2"]
        break  # Only take the first (and only) batch since we want ~200 samples

    # Reshape raw images for t-SNE (flatten 56x28 into vectors)
    raw_images_flattened = raw_images.view(raw_images.size(0), -1).numpy()

    # Perform t-SNE on raw MNIST input
    print("Performing t-SNE on MNIST input...")
    apply_tSNE(raw_images_flattened, labels1.numpy(), labels2.numpy(),
               perplexity=perplexity, title="t-SNE on Two-Image MNIST Input", save_path="tsne_mnist_input.png")

    # Load the best-performing model
    checkpoint = torch.load("../Q1-2/best_model_checkpoint.pth")
    best_hyperparams = checkpoint['hyperparameters']
    print("Loaded hyperparameters:", best_hyperparams)

    # Define the : same structure with Q2 tuning but without wrapper
    class DenseNet(nn.Module):
        def __init__(self, input_size, hidden1, hidden2, output_size):
            super(DenseNet, self).__init__()
            self.dense_net = nn.Sequential(
                nn.Flatten(),
                nn.Linear(input_size, hidden1),
                nn.ReLU(),
                nn.Linear(hidden1, hidden2),
                nn.ReLU(),
                nn.Linear(hidden2, output_size)  # Output layer included here
            )

        def forward(self, x, feature_only=False):
            """
            Forward pass through the network.

            Args:
                x: Input tensor.
                feature_only: If True, return features from hidden2 layer instead of final output.

            Returns:
                Output tensor (final layer) or features (hidden2 layer).
            """
            if feature_only:
                x = self.dense_net[0](x)  # Flatten
                x = self.dense_net[1](x)  # Linear(input_size -> hidden1)
                x = self.dense_net[2](x)  # ReLU
                x = self.dense_net[3](x)  # Linear(hidden1 -> hidden2)
                x = self.dense_net[4](x)  # ReLU
                return x  # Hidden2 output
            else:
                return self.dense_net(x)  # Full forward pass

    # Reinitialize the model
    model = DenseNet(
        input_size=56 * 28,
        hidden1=best_hyperparams['hidden1'],
        hidden2=best_hyperparams['hidden2'],
        output_size=19
    )

    # Load the saved model state
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Get hidden layer outputs for t-SNE
    with torch.no_grad(): # no parameter update
        hidden2_output = model(raw_images, feature_only=True).numpy()

    # Perform t-SNE on hidden layer outputs
    print("Performing t-SNE on hidden layer outputs...")
    apply_tSNE(hidden2_output, labels1.numpy(), labels2.numpy(),
               perplexity=perplexity, title="t-SNE on Hidden Layer Outputs", save_path="tsne_hidden_output.png")


if __name__ == "__main__":
    main()
