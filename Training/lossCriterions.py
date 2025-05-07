import torch
import torch.nn as nn

class PoseLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.geo = GeometricConsistencyLoss()

    def forward(self, pred, target):
        return 0.8 * self.mse(pred, target) + 0.2 * self.geo(pred)


class GeometricConsistencyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Predefined limb length ratios
        self.limb_weights = {
            'shoulder_elbow': 0.15,
            'elbow_wrist': 0.12,
            # ... define other limb relationships
        }

    def forward(self, poses):
        # Calculate limb length consistency
        loss = 0
        for pair, weight in self.limb_weights.items():
            j1, j2 = pair.split('_')
            limb_lengths = torch.norm(poses[:, j1] - poses[:, j2], dim=-1)
            loss += weight * torch.var(limb_lengths)
        return loss
