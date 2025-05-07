import torch
import torch.nn as nn
import torch.nn.functional as F

class EnhancedPartRegressor2D(nn.Module):
    def __init__(self, num_joints=14):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.res_blocks = nn.Sequential(
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 512, stride=2)
        )
        self.final_conv = nn.Conv2d(512, num_joints, 1)

    def forward(self, x):
        x = self.conv1(x)  # (B,64,240,320)
        x = self.res_blocks(x)  # (B,512,30,40)
        return self.final_conv(x)  # (B,14,30,40)


class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.downsample = nn.Sequential(
            nn.Conv2d(in_c, out_c, 1, stride),
            nn.BatchNorm2d(out_c)
        ) if stride != 1 or in_c != out_c else nn.Identity()

    def forward(self, x):
        identity = self.downsample(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + identity)


class EnhancedSelecSLS(nn.Module):
    def __init__(self, num_joints=14):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Conv2d(3 + num_joints, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.encoder = nn.Sequential(
            DepthSepConv(256, 512, stride=2),
            DepthSepConv(512, 1024, stride=2),
            SpatialAttention()
        )
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            DepthSepConv(1024, 512),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            DepthSepConv(512, 256)
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, num_joints * 3)
        )

    def forward(self, x_img, x_heat):
        x = torch.cat([x_img, x_heat], 1)
        x = self.input_proj(x)
        x = self.encoder(x)
        x = self.decoder(x)
        return self.head(x).view(-1, 3, 14)


class DepthSepConv(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_c, in_c, 3, stride, 1, groups=in_c)
        self.pointwise = nn.Conv2d(in_c, out_c, 1)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        return F.relu(self.bn(self.pointwise(self.depthwise(x))))


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3)

    def forward(self, x):
        avg = torch.mean(x, 1, keepdim=True)
        max_val, _ = torch.max(x, 1, keepdim=True)
        att = torch.sigmoid(self.conv(torch.cat([avg, max_val], 1)))
        return x * att


class MotionCaptureSystem(nn.Module):
    def __init__(self):
        super().__init__()
        self.part_regressor = EnhancedPartRegressor2D()
        self.pose_estimator = EnhancedSelecSLS()

    def forward(self, x):
        heatmaps = self.part_regressor(x)
        return self.pose_estimator(x, heatmaps)
