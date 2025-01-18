import torch
from torch import nn
from torch.nn import functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

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

class ResNetModel(nn.Module):
    def __init__(self, input_channels=1, model_type="resnet18", num_classes=36, pretrained=True):
        
        super(ResNetModel, self).__init__()

        if model_type == "resnet18":
            self.resnet = models.resnet18(pretrained=pretrained)
        elif model_type == "resnet34":
            self.resnet = models.resnet34(pretrained=pretrained)
        elif model_type == "resnet50":
            self.resnet = models.resnet50(pretrained=pretrained)
        else:
            raise ValueError(f"Invalid model_type: {model_type}. Choose from 'resnet18', 'resnet34', 'resnet50'.")
        
        for param in self.resnet.parameters():
            param.requires_grad = False
        
        self.resnet.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

        # unfreeze the parameters of self.resnet.conv1 and self.resnet.fc
        for param in self.resnet.conv1.parameters():
            param.requires_grad = True
        for param in self.resnet.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.resnet(x)
    

class MobileNetModel(nn.Module):
    def __init__(self, input_channels=1, model_type="mobilenetV2", num_classes=36, pretrained=True):
        
        super(MobileNetModel, self).__init__()

        if model_type == "mobilenetV2":
            self.mobilenet = models.mobilenet_v2(pretrained=pretrained)
            first_layer = self.mobilenet.features[0]
        elif model_type == "mobilenetV3L":
            self.mobilenet = models.mobilenet_v3_large(pretrained=pretrained)
            first_layer = self.mobilenet.features[0][0]
        elif model_type == "mobilenetV3S":
            self.mobilenet = models.mobilenet_v3_small(pretrained=pretrained)
            first_layer = self.mobilenet.features[0][0]
        else:
            raise ValueError(f"Invalid model_type: {model_type}. Choose from 'mobilenetV2', 'mobilenetV3L', 'mobilenetV3S'.")
        
        # Extract the first layer (ConvBNReLU)
        first_layer = self.mobilenet.features[0]
        conv_layer = first_layer[0]  # Access the Conv2d layer inside ConvBNReLU

        # Replace the Conv2d layer for single-channel input (grayscale)
        new_conv = nn.Conv2d(
            in_channels=input_channels,  # Change to single-channel input or 3 channels
            out_channels=conv_layer.out_channels,
            kernel_size=conv_layer.kernel_size,
            stride=conv_layer.stride,
            padding=conv_layer.padding,
            bias=conv_layer.bias
        )

        # Create a new ConvBNReLU with the modified Conv2d
        self.mobilenet.features[0] = nn.Sequential(
            new_conv,
            first_layer[1]  # Keep the original BatchNorm and activation
        )
        
        self.mobilenet.classifier[1] = nn.Linear(self.mobilenet.last_channel, num_classes)

    def forward(self, x):
        return self.mobilenet(x)
    
if __name__ == "__main__":
    model = MyAwesomeModel()
    print(f"Model architecture: {model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    dummy_input = torch.randn(1, 1, 28, 28)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")