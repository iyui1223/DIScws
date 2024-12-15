import torch
import torch.nn as nn
# Define the model
class LinearClassifier56(nn.Module):
    def __init__(self):
        super(LinearClassifier56, self).__init__()
        self.weaklinear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(56 * 28, 128),  # Input: Flattened 56x28 image
            nn.ReLU(),
        )

    def forward(self, x):
        # Input shape: [batch_size, 1, 56, 28]
        return self.weaklinear(x)


class SplitAndSum(nn.Module):
    def __init__(self):
        super(SplitAndSum, self).__init__()

        # Shared network for processing each image
        self.shared_weak_linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 10),  # For each image (28x28)
            nn.ReLU(),
        )

        # Final linear layer to compute the sum with fixed weights
        self.sum_layer = nn.Linear(20, 19, bias=False)  # Fixed summation layer, no bias
        self._initialize_sum_layer()

    def _initialize_sum_layer(self):
        """
        Initialize the weights of the summation layer to fixed values such that:
        The weights encode the deterministic addition of two digits.
        """
        # Create a weight matrix of shape (19, 20)
        weights = torch.zeros(19, 20)

        for i in range(10):  # First digit
            for j in range(10):  # Second digit
                sum_idx = i + j
                weights[sum_idx, i] = 1  # From the first network output
                weights[sum_idx, j + 10] = 1  # From the second network output

        # Assign fixed weights to the sum_layer
        self.sum_layer.weight = nn.Parameter(weights, requires_grad=False)

    def forward(self, img1, img2):
        """
        Forward pass for SplitAndSum.
        
        Args:
        - img1: Tensor of shape [batch_size, 1, 28, 28] (first image)
        - img2: Tensor of shape [batch_size, 1, 28, 28] (second image)
        
        Returns:
        - output: Tensor of shape [batch_size, 19] (output for sum classification)
        """

        # Process each image through the shared network
        out1 = self.shared_weak_linear(img1)  # Output: [batch_size, 10]
        out2 = self.shared_weak_linear(img2)  # Output: [batch_size, 10]

        # Concatenate the outputs and compute the sum using the fixed summation layer
        combined_out = torch.cat((out1, out2), dim=1)  # Output: [batch_size, 20]
        output = self.sum_layer(combined_out)  # Final 19-class output

        return output

    