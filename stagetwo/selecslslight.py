import torch
import torch.nn as nn
import torch.nn.functional as F

class LightweightSelecSLS(nn.Module):
    def __init__(self):
        super(LightweightSelecSLS, self).__init__()

        # Initial convolutional layer with reduced filters
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # Reduced number of filters in subsequent layers
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Global average pooling instead of fully connected layers
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Final fully connected layer
        self.fc = nn.Linear(128, 42)  # 42 = 3 * 14 for output

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = self.global_pool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        return x.view(-1, 3, 14)