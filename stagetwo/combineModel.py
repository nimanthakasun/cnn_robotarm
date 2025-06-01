import torch
import torch.nn as nn
import torch.nn.functional as F
from xarray.util.generate_ops import inplace


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
        # self.selec_sls_3d = ModifiedSelecSLS()
        self.selec_sls_3d = SelecSLSMod()

    def forward(self, x_image):
        heatmaps_2d = self.part_regressor_2d(x_image)
        output_3d = self.selec_sls_3d(x_image, heatmaps_2d)
        return output_3d

class SelecSLSMod(nn.Module):
    def __init__(self, num_joints=14):
        super(SelecSLSMod, self).__init__()
        input_channels = 3 + num_joints

        # Initial conv
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # Second conv
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Third conv (no downsampling)
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Fourth conv (downsample again)
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        # self.conv5 = nn.Sequential(
        #     nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True)
        # )

        # Combine selected features with 1x1 conv (like SelecSLS)
        self.conv_combine = nn.Conv2d(32 + 64 + 64 + 128, 128, kernel_size=1)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, num_joints * 3)

    def forward(self, x_image, x_heatmaps):
        x = torch.cat((x_image, x_heatmaps), dim=1)  # (B, 3+14, H, W)

        out1 = self.conv1(x)   # (B, 32, H/2, W/2)
        out2 = self.conv2(out1)# (B, 64, H/4, W/4)
        out3 = self.conv3(out2)# (B, 64, H/4, W/4)
        out4 = self.conv4(out3)# (B, 128, H/8, W/8)

        # Resize all outputs to the same spatial dimension before concat
        out1_resized = F.interpolate(out1, size=out4.shape[2:], mode='bilinear', align_corners=False)
        out2_resized = F.interpolate(out2, size=out4.shape[2:], mode='bilinear', align_corners=False)
        out3_resized = F.interpolate(out3, size=out4.shape[2:], mode='bilinear', align_corners=False)

        # Concatenate selected feature maps
        concat = torch.cat([out1_resized, out2_resized, out3_resized, out4], dim=1)  # (B, 288, H/8, W/8)

        # Compress with 1x1 conv
        combined = F.relu(self.conv_combine(concat))  # (B, 128, H/8, W/8)

        x = self.global_pool(combined)                # (B, 128, 1, 1)
        x = x.view(x.size(0), -1)                     # (B, 128)
        x = self.fc(x)                                # (B, 42)
        return x.view(-1, 3, 14)                      # (B, 3, 14)