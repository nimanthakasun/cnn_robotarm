import torch
import torch.nn as nn

class PoseMetrics(nn.Module):
    def __init__(self, joint_weights=None):
        super().__init__()
        self.joint_weights = joint_weights or torch.ones(14)

    def forward(self, pred, target):
        # MPJPE (Mean Per Joint Position Error)
        l2_dist = torch.norm(pred - target, dim=1)  # (B,14)
        mpjpe = torch.mean(l2_dist)

        # PA-MPJPE (Procrustes Aligned)
        aligned = self.procrustes(pred, target)
        pa_mpjpe = torch.mean(torch.norm(aligned - target, dim=1))

        # Acceleration Error (temporal smoothness)
        if pred.dim() == 3:  # (B,3,14)
            accel_error = torch.mean(torch.norm(pred[:, :, 2:] - 2 * pred[:, :, 1:-1] + pred[:, :, :-2], dim=1))
        else:
            accel_error = torch.tensor(0.0)

        return {'mpjpe': mpjpe, 'pa_mpjpe': pa_mpjpe, 'accel_error': accel_error}

    def procrustes(pred, target):
        aligned_pred = torch.zeros_like(pred)

        for i in range(pred.size(0)):
            # Center both poses
            pred_centered = pred[i] - pred[i].mean(dim=1, keepdim=True)
            target_centered = target[i] - target[i].mean(dim=1, keepdim=True)


            # Compute rotation using SVD
            H = pred_centered @ target_centered.T
            U, S, V = torch.svd(H)

            # Ensure proper rotation (no reflection)
            R = V @ U.T
            if torch.det(R) < 0:
                V[:, -1] *= -1
                R = V @ U.T

            # Apply optimal rotation
            aligned_pred[i] = R @ pred_centered + target[i].mean(dim=1, keepdim=True)

        return aligned_pred


class AlignedPoseLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        # Align predictions to ground truth
        aligned_pred = PoseMetrics.procrustes(pred, target)

        # Calculate aligned MSE
        aligned_loss = self.mse(aligned_pred, target)

        # Optional: Add original MSE for stability
        raw_loss = self.mse(pred, target)

        return 0.7 * aligned_loss + 0.3 * raw_loss

