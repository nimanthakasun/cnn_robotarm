import torch
import torch.nn as nn
import torch.nn.functional as F


class PartRegressor2D(nn.Module):
    def __init__(self, num_joints=14):
        super(PartRegressor2D, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, num_joints, kernel_size=1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.conv3(x)
        return x  # Output shape: (batch_size, 14, 480, 640)


class ModifiedSelecSLS(nn.Module):
    def __init__(self, num_joints=14):
        super(ModifiedSelecSLS, self).__init__()

        self.conv1 = nn.Conv2d(3 + num_joints, 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, num_joints * 3)

    def forward(self, x_image, x_heatmaps):
        x = torch.cat((x_image, x_heatmaps), dim=1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x.view(-1, 3, 14)


class MotionCapturePipeline(nn.Module):
    def __init__(self):
        super(MotionCapturePipeline, self).__init__()

        self.part_regressor_2d = PartRegressor2D()
        self.selec_sls_3d = ModifiedSelecSLS()

    def forward(self, x_image):
        heatmaps_2d = self.part_regressor_2d(x_image)
        output_3d = self.selec_sls_3d(x_image, heatmaps_2d)
        return output_3d

