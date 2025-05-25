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

    @staticmethod
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
                V_reflected = V.clone()
                V_reflected[:, -1] = -V_reflected[:, -1]
                R = V_reflected @ U.T

            # Apply optimal rotation
            aligned_pred[i] = R @ pred_centered + target[i].mean(dim=1, keepdim=True)

        return aligned_pred


class CombinedPoseLoss(nn.Module):
    def __init__(self, heatmap_weight=1.0, mpjpe_weight=1.0, pa_mpjpe_weight=0.5, accel_weight=0.1):
        super().__init__()
        self.heatmap_loss = nn.MSELoss()
        self.pose_metrics = PoseMetrics()
        self.heatmap_weight = heatmap_weight
        self.mpjpe_weight = mpjpe_weight
        self.pa_mpjpe_weight = pa_mpjpe_weight
        self.accel_weight = accel_weight

    def forward(self,pred_2d, pred_3d, target_2d, target_3d):
        losses = {}

        # 1. Heatmap Loss (MSE)
        losses['heatmap_loss'] = self.heatmap_loss(pred_2d, target_2d)

        # 2. 3D Metrics
        metrics = self.pose_metrics(pred_3d, target_3d)
        losses.update(metrics)

        # 3. Weighted Sum
        total = (
                self.heatmap_weight * losses['heatmap_loss'] +
                self.mpjpe_weight * losses['mpjpe'] +
                self.pa_mpjpe_weight * losses['pa_mpjpe'] +
                self.accel_weight * losses['accel_error']
        )

        losses['total'] = total
        return losses

