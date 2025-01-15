import torch
from torch import nn
from torch.nn import functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class MyAwesomeModel(nn.Module):
    """Dynamic model definition."""

    def __init__(
        self,
        input_channels: int = 1,
        conv_layers: list = [(32, 3, 1), (64, 3, 1), (128, 3, 1)],  # [(out_channels, kernel_size, stride), ...]
        fc_layers: list = [64, 10],  # Fully connected layer sizes
        activation: str = "ReLU",
        dropout_p: float = 0.2,
    ) -> None:
        super().__init__()

        # Store layers
        self.conv_layers = nn.ModuleList()
        self.fc_layers = nn.ModuleList()
        self.activations = {"ReLU": nn.ReLU, "LeakyReLU": nn.LeakyReLU, "Tanh": nn.Tanh, "Sigmoid": nn.Sigmoid}
        self.activation = self.activations.get(activation, nn.ReLU)()

        # Create convolutional layers
        in_channels = input_channels
        for out_channels, kernel_size, stride in conv_layers:
            self.conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride))
            in_channels = out_channels

        # Create fully connected layers
        for i in range(len(fc_layers) - 1):
            self.fc_layers.append(nn.Linear(fc_layers[i], fc_layers[i + 1]))

        # Other layers
        self.dropout = nn.Dropout(p=dropout_p)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # Forward through convolutional layers
        for conv in self.conv_layers:
            x = self.activation(conv(x))
            x = F.max_pool2d(x, 2, 2)

        # Flatten the tensor
        x = x.view(x.shape[0], -1)

        # Forward through fully connected layers
        for i, fc in enumerate(self.fc_layers):
            if i < len(self.fc_layers) - 1:  # Apply dropout and activation for all but the last layer
                x = self.dropout(self.activation(fc(x)))
            else:
                x = self.logsoftmax(fc(x))

        return x

if __name__ == "__main__":
    model = MyAwesomeModel()
    print(f"Model architecture: {model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    dummy_input = torch.randn(1, 1, 28, 28)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")