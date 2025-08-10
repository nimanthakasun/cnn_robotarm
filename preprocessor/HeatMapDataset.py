import torch
from torch.utils.data import DataLoader, Dataset

class HeatMapDataset(Dataset):
    def __init__(self):
        frames, heatmaps = self.read_existing_dataset_1()
        self.video_frames = frames
        self.heatmap_label = heatmaps

    def __len__(self):
        return  len(self.video_frames)

    def __getitem__(self, idx):
        frame = self.video_frames[idx]
        label = self.heatmap_label[idx]
        return torch.tensor(frame, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

    def read_existing_dataset(self):
        heatmap_size = (480, 640)
        image_size = (640, 480)
        sigma = 4
        ExistingDataset = "n_guestures_1_2_C1.pt"
        loaded_dataset = torch.load(ExistingDataset)
        if len(loaded_dataset.labels) > len(loaded_dataset.video_frames):
            loaded_dataset.labels = loaded_dataset.labels[:len(loaded_dataset.video_frames)]
            print("Modified label length for HeatmapGen: ", len(loaded_dataset.labels))

        print("HeatmapGen Input: \n", loaded_dataset)
        heatmaps = []
        for i, (vid_frames, vid_labels) in enumerate(loaded_dataset):
            print("Label lenght", len(vid_labels))
            print(vid_labels)
            B, _, J = vid_labels.shape
            W, H = image_size
            vid_labels = vid_labels.permute(0, 2, 1)  # (B, 14, 3)

            x = vid_labels[..., 0]
            y = vid_labels[..., 1]

            # Normalize to [-1, 1]
            x_norm = x / x.abs().max(dim=1, keepdim=True)[0].clamp(min=1e-5)
            y_norm = y / y.abs().max(dim=1, keepdim=True)[0].clamp(min=1e-5)

            # Rescale to image size
            u = (x_norm + 1) * W / 2
            v = (y_norm + 1) * H / 2

            # for i in range(len(loaded_dataset.video_frames)):
            B, J, _ = torch.stack([u, v], dim=-1)
            H, W = heatmap_size
            device = vid_labels.device

            # Create mesh grid for heatmap
            y_range = torch.arange(0, H, device=device).view(1, 1, H, 1)
            x_range = torch.arange(0, W, device=device).view(1, 1, 1, W)

            y_range = y_range.expand(B, J, H, W)
            x_range = x_range.expand(B, J, H, W)

            # Get joint coordinates
            x = vid_labels[:, :, 0].view(B, J, 1, 1)
            y = vid_labels[:, :, 1].view(B, J, 1, 1)

            # Compute Gaussian heatmaps
            heatmaps.append(torch.exp(-((x_range - x) ** 2 + (y_range - y) ** 2) / (2 * sigma ** 2)))

        return loaded_dataset.video_frames, heatmaps

    def read_existing_dataset_1(self):
        heatmap_size = (480, 640)
        image_size = (640, 480)
        sigma = 4
        ExistingDataset = "n_guestures_1_2_C1.pt"

        loaded_dataset = torch.load(ExistingDataset)

        # Make sure labels match video_frames length
        if len(loaded_dataset.labels) > len(loaded_dataset.video_frames):
            loaded_dataset.labels = loaded_dataset.labels[:len(loaded_dataset.video_frames)]
            print("Modified label length for HeatmapGen:", len(loaded_dataset.labels))

        print("HeatmapGen Input: \n", loaded_dataset)
        heatmaps = []

        for vid_frames, vid_labels in zip(loaded_dataset.video_frames, loaded_dataset.labels):
            # vid_labels shape: (3, 14) → (coords, joints)
            # We only need x and y for heatmaps
            x = vid_labels[0]  # shape: (14,)
            y = vid_labels[1]  # shape: (14,)

            # Rescale from normalized coords [-1, 1] to image size
            W, H = image_size
            u = (x + 1) * W / 2
            v = (y + 1) * H / 2

            # Create meshgrid for Gaussian
            Hm, Wm = heatmap_size
            device = vid_labels.device
            y_range = torch.arange(0, Hm, device=device).view(Hm, 1)
            x_range = torch.arange(0, Wm, device=device).view(1, Wm)

            # For each joint, create a Gaussian
            joint_heatmaps = []
            for j in range(len(x)):
                xj, yj = u[j], v[j]
                g = torch.exp(-((x_range - xj) ** 2 + (y_range - yj) ** 2) / (2 * sigma ** 2))
                joint_heatmaps.append(g)

            # Stack joints → shape: (14, H, W)
            joint_heatmaps = torch.stack(joint_heatmaps, dim=0)

            heatmaps.append(joint_heatmaps)

        # Return same video frames, but with heatmaps instead of raw labels
        return loaded_dataset.video_frames, heatmaps


