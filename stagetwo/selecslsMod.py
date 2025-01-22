import torch
import torch.nn as nn
import torch.nn.functional as F

class SelecSLSNet(nn.Module):
    def __init__(self):
        super(SelecSLSNet, self).__init__()

        # Initial convolutional layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        # Selective Long and Short Range Skip Connections
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        # Fully connected layers for regression
        self.fc1 = nn.Linear(256 * 120 * 160, 1024)  # Adjusted for input size (3, 480, 640)
        self.fc2 = nn.Linear(1024, 42)  # Output size matches (3, 14)

    def forward(self, x):
        # Convolutional layers with ReLU and BatchNorm
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # Flatten the tensor for fully connected layers
        x = x.reshape(x.size(0), -1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        # Reshape to match output label size (batch_size, 3, 14)
        x = x.view(-1, 3, 14)
        return x
