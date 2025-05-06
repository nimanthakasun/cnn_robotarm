import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """A basic residual block with optional downsampling."""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # If dimensions change or stride!=1, use a 1x1 conv to match
        self.downsample = (nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
            nn.BatchNorm2d(out_channels)
        ) if (stride != 1 or in_channels != out_channels) else None)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return F.relu(out)

class EnhancedPartRegressor2D(nn.Module):
    """Deep U-Net style 2D keypoint heatmap regressor with residual blocks."""
    def __init__(self, in_channels=3, num_joints=14, base_filters=64):
        super().__init__()
        # Initial conv to increase channel dimension
        self.conv_start = nn.Sequential(
            nn.Conv2d(in_channels, base_filters, 7, 2, padding=3, bias=False),  # 480x640 -> 240x320
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True)
        )
        # Encoder: downsample twice
        self.enc1 = ResidualBlock(base_filters, base_filters, stride=1)
        self.down1 = ResidualBlock(base_filters, base_filters*2, stride=2)     # 240x320 -> 120x160
        self.enc2 = ResidualBlock(base_filters*2, base_filters*2, stride=1)
        self.down2 = ResidualBlock(base_filters*2, base_filters*4, stride=2)  # 120x160 -> 60x80
        self.enc3 = ResidualBlock(base_filters*4, base_filters*4, stride=1)
        # Decoder: upsample twice
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(base_filters*4, base_filters*2, 4, 2, padding=1, bias=False),  # 60x80 -> 120x160
            nn.BatchNorm2d(base_filters*2),
            nn.ReLU(inplace=True)
        )
        self.dec1 = ResidualBlock(base_filters*2, base_filters*2, stride=1)
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(base_filters*2, base_filters, 4, 2, padding=1, bias=False),    # 120x160 -> 240x320
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True)
        )
        self.dec2 = ResidualBlock(base_filters, base_filters, stride=1)
        # Final upsampling to restore original size
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(base_filters, base_filters, 4, 2, padding=1, bias=False),      # 240x320 -> 480x640
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True)
        )
        self.dec3 = ResidualBlock(base_filters, base_filters, stride=1)
        # Output heatmaps: one per joint
        self.out_conv = nn.Conv2d(base_filters, num_joints, 1, 1, 0)

    def forward(self, x):
        # Encoder
        x = self.conv_start(x)      # [B, base_filters, 240, 320]
        e1 = self.enc1(x)          # [B, base_filters, 240, 320]
        d1 = self.down1(e1)        # [B, base_filters*2, 120, 160]
        e2 = self.enc2(d1)         # [B, base_filters*2, 120, 160]
        d2 = self.down2(e2)        # [B, base_filters*4, 60, 80]
        e3 = self.enc3(d2)         # [B, base_filters*4, 60, 80]
        # Decoder with skip-connections
        u1 = self.up1(e3)          # [B, base_filters*2, 120, 160]
        u1 = u1 + e2              # skip connection
        c1 = self.dec1(u1)         # [B, base_filters*2, 120, 160]
        u2 = self.up2(c1)          # [B, base_filters, 240, 320]
        u2 = u2 + e1              # skip connection
        c2 = self.dec2(u2)         # [B, base_filters, 240, 320]
        u3 = self.up3(c2)          # [B, base_filters, 480, 640]
        c3 = self.dec3(u3)         # [B, base_filters, 480, 640]
        heatmaps = self.out_conv(c3)  # [B, num_joints, 480, 640]
        return heatmaps


class Enhanced3DRegressor(nn.Module):
    """Convolutional network for regressing 3D joints from image+heatmaps."""
    def __init__(self, in_channels=17, num_joints=14, base_filters=64):
        super().__init__()
        # Initial 1x1 conv to mix input channels (image + heatmaps)
        self.mix_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_filters, 1, bias=False),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True)
        )
        # Several residual downsampling blocks
        self.layer1 = ResidualBlock(base_filters, base_filters, stride=1)       # [B,64,480,640]
        self.layer2 = ResidualBlock(base_filters, base_filters*2, stride=2)     # [B,128,240,320]
        self.layer3 = ResidualBlock(base_filters*2, base_filters*2, stride=1)   # [B,128,240,320]
        self.layer4 = ResidualBlock(base_filters*2, base_filters*4, stride=2)   # [B,256,120,160]
        self.layer5 = ResidualBlock(base_filters*4, base_filters*4, stride=1)   # [B,256,120,160]
        self.layer6 = ResidualBlock(base_filters*4, base_filters*8, stride=2)   # [B,512,60,80]
        self.layer7 = ResidualBlock(base_filters*8, base_filters*8, stride=1)   # [B,512,60,80]
        # Global pooling and final regressor
        self.avgpool = nn.AdaptiveAvgPool2d(1)   # [B,512,1,1]
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),  # helps prevent overfitting on small data
            nn.Linear(base_filters*8, num_joints * 3)
        )

    def forward(self, image, heatmaps):
        # Expect image: [B,3,480,640], heatmaps: [B,14,480,640]
        x = torch.cat((image, heatmaps), dim=1)   # [B,17,480,640]
        x = self.mix_conv(x)                     # [B,64,480,640]
        x = self.layer1(x)                       # [B,64,480,640]
        x = self.layer2(x)                       # [B,128,240,320]
        x = self.layer3(x)                       # [B,128,240,320]
        x = self.layer4(x)                       # [B,256,120,160]
        x = self.layer5(x)                       # [B,256,120,160]
        x = self.layer6(x)                       # [B,512,60,80]
        x = self.layer7(x)                       # [B,512,60,80]
        x = self.avgpool(x)                      # [B,512,1,1]
        x = self.fc(x)                           # [B, 42]
        joints3d = x.view(x.size(0), 3, -1)      # [B, 3, 14]
        return joints3d


class MotionCapturePipeline(nn.Module):
    def __init__(self):
        super().__init__()
        self.part2d = EnhancedPartRegressor2D(in_channels=3, num_joints=14)
        # Enhanced3DRegressor expects 17 input channels (3 from image + 14 heatmaps)
        self.regress3d = Enhanced3DRegressor(in_channels=17, num_joints=14)

    def forward(self, images):
        # images: [B,3,480,640]
        heatmaps = self.part2d(images)             # [B,14,480,640]
        coords3d = self.regress3d(images, heatmaps)  # [B,3,14]
        return coords3d
