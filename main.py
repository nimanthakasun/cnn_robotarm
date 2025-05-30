import os
import json
import torch

from preprocessor.DataSet import VideoDataset
from torch.utils.data import DataLoader, TensorDataset, random_split
# from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch.nn as nn
import torch.optim as optim
import gc
import sys
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from stagetwo import selecsls, selecslsMod, selecslslight, combineModel
from torchinfo import summary
from pyomeca import analogs, Markers
from preprocessor.FrameExtracter import FrameExtractor

# marker_locations = ["LSHO","LUPA","LELB","LWRA","LWRB","LFRA","LFIN","RSHO","RUPA","RELB","RWRA","RWRB","RFRA","RFIN"]
# marker_location = ["LSHO"]
# dataset_paths = ['dataset_tensor_1.pt', 'dataset_tensor_2.pt', 'dataset_tensor_3.pt']
dataset_paths = ['dataset_tensor_4.pt', 'dataset_tensor_5.pt', 'dataset_tensor_6.pt']

from stagetwo.selecslsMod import SelecSLSNet
from stagetwo.selecslslight import LightweightSelecSLS
from stagetwo.combineModel import MotionCapturePipeline
from stagetwo.advanced import MotionCapturePipelineAdvanced
from stagetwo.enhancedAdv import MotionCaptureSystem

from Training import advanceTrainingScheme
from Training import lossCriterions
from Training import poseMatrix

# ###############################################   Dataset creation   ###################################################
def create_dataset():
    dataset_tensor = VideoDataset()
    sample_frame,sample_label = dataset_tensor[0]
    print("Video shape: ", sample_frame.shape)
    print("Label shape: ", sample_label.shape)
    print("Saving Dataset:")
    torch.save(dataset_tensor, "dataset_tensor_6.pt")
    print(dataset_tensor.__len__())

def dataset_details(path):
    loaded_dataset = torch.load(path)
    print("Length of frames set: ", len(loaded_dataset.video_frames))
    print("Length of labels set: ", len(loaded_dataset.labels))

    sample_frame, sample_label = loaded_dataset[25]
    print("Video shape: ", sample_frame.shape)
    print("Label shape: ", sample_label.shape)

    print("Video tensor shape: ", np.asarray(loaded_dataset.video_frames).shape)
    print("Label tensor shape: ", np.asarray(loaded_dataset.labels).shape)

    print (sample_label)

def prepare_dataset(dataset_path):
    loaded_dataset = torch.load(dataset_path)

    if len(loaded_dataset.labels) > len(loaded_dataset.video_frames):
        loaded_dataset.labels = loaded_dataset.labels[:len(loaded_dataset.video_frames)]
        print("Modified label length: ", len(loaded_dataset.labels))

    print(loaded_dataset)

    train_size = int(0.8 * len(loaded_dataset))
    test_size = len(loaded_dataset) - train_size

    train_dataset, test_dataset = random_split(loaded_dataset, [train_size, test_size])
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    # Set batch size
    batch_size = 8
    workers = os.cpu_count()

    # Create DataLoaders for training and testing
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)

    return train_loader, test_loader

def orthographic_projection(joints_3d, image_size=(640, 480)):

    B, _, J = joints_3d.shape
    W, H = image_size
    joints_3d = joints_3d.permute(0, 2, 1)  # (B, 14, 3)

    x = joints_3d[..., 0]
    y = joints_3d[..., 1]

    # Normalize to [-1, 1]
    x_norm = x / x.abs().max(dim=1, keepdim=True)[0].clamp(min=1e-5)
    y_norm = y / y.abs().max(dim=1, keepdim=True)[0].clamp(min=1e-5)

    # Rescale to image size
    u = (x_norm + 1) * W / 2
    v = (y_norm + 1) * H / 2

    return torch.stack([u, v], dim=-1)

def generate_heatmaps_2d(joints_2d, heatmap_size=(480, 640), sigma=4):
    B, J, _ = joints_2d.shape
    H, W = heatmap_size
    device = joints_2d.device

    # Create mesh grid for heatmap
    y_range = torch.arange(0, H, device=device).view(1, 1, H, 1)
    x_range = torch.arange(0, W, device=device).view(1, 1, 1, W)

    y_range = y_range.expand(B, J, H, W)
    x_range = x_range.expand(B, J, H, W)

    # Get joint coordinates
    x = joints_2d[:, :, 0].view(B, J, 1, 1)
    y = joints_2d[:, :, 1].view(B, J, 1, 1)

    # Compute Gaussian heatmaps
    heatmaps = torch.exp(-((x_range - x) ** 2 + (y_range - y) ** 2) / (2 * sigma ** 2))

    return heatmaps

def train_model(model, loader, optimizer, criterion, device):
# Taining part
    model.train()
    total_loss = 0
    optimizer.zero_grad()

    loss_log = {
        'mpjpe': 0.0,
        'pa_mpjpe': 0.0,
        'accel_error': 0.0,
        'total': 0.0,
        'heatmap_loss': 0.0
        # 'pose_loss': 0.0,
    }

    num_batches = 0

    for i, (video_frames, labels) in enumerate(loader):
        video_frames_rearranged = video_frames.permute(0, 3, 1, 2)
        del video_frames
        video_frames_rearranged = video_frames_rearranged.float().to(device)
        labels = labels.float().to(device)

        # Forward pass
        optimizer.zero_grad()

        # Old way
        # outputs = model(video_frames_rearranged)
        # loss = torch.sqrt(criterion(outputs, labels))

        # New way
        joints_2d = orthographic_projection(labels, (480, 640))
        gt_heatmaps_2d = generate_heatmaps_2d(joints_2d, heatmap_size=(480, 640), sigma=4)
        heatmaps_2d = model.part_regressor_2d(video_frames_rearranged)
        output_3d = model.selec_sls_3d(video_frames_rearranged, heatmaps_2d)

        loss = criterion(heatmaps_2d, output_3d, gt_heatmaps_2d, labels)

        loss['total'].backward()
        optimizer.step()

        for key in loss:
            if isinstance(loss[key], torch.Tensor):
                loss_log[key] += loss[key].detach().item()

        num_batches += 1

        del loss, heatmaps_2d, output_3d

    # Print average loss for the epoch
    avg_loss =  {key: val / num_batches for key, val in loss_log.items()}
    print(avg_loss)

    return avg_loss

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    metrics_total = {
        'mpjpe': 0.0,
        'pa_mpjpe': 0.0,
        'accel_error': 0.0,
        'loss': 0.0
    }
    num_batches = 0

    with torch.no_grad():
        for i, (video_frames, labels) in enumerate(dataloader):
            video_frames_rearranged = video_frames.permute(0, 3, 1, 2)
            del video_frames
            video_frames_rearranged = video_frames_rearranged.float().to(device)
            labels = labels.float().to(device)

            # Optional: Generate 2D heatmaps from 3D labels
            joints_2d = orthographic_projection(labels, (video_frames_rearranged.shape[3], video_frames_rearranged.shape[2]))  # [B, 14, 2]
            gt_heatmaps_2d = generate_heatmaps_2d(joints_2d, heatmap_size=(video_frames_rearranged.shape[2], video_frames_rearranged.shape[3]), sigma=4)

            # Model forward
            heatmaps_2d = model.part_regressor_2d(video_frames_rearranged)
            output_3d = model.selec_sls_3d(video_frames_rearranged, heatmaps_2d)

            # Compute loss and metrics
            loss_dict = criterion(heatmaps_2d, output_3d, gt_heatmaps_2d, labels)

            metrics_total['mpjpe'] += loss_dict['mpjpe'].item()
            metrics_total['pa_mpjpe'] += loss_dict['pa_mpjpe'].item()
            metrics_total['accel_error'] += loss_dict['accel_error'].item()
            metrics_total['loss'] += loss_dict['total'].item()
            num_batches += 1

    # Average across batches
    for key in metrics_total:
        metrics_total[key] /= num_batches
    print(metrics_total)

    return metrics_total

def infer_model(h_device, saved_state_dict):
    print("Model in Inference mode")
    model.load_state_dict(torch.load(saved_state_dict, map_location=device))
    model.to(h_device)
    model.eval()

    frm_extrct = FrameExtractor()
    frames_for_infer, total_frames, frame_w, frame_he, frame_rate, frame_shape = frm_extrct.extract_frames("../Datasets/Output_1.avi")

    inference_outputs = []
    for i in range(0, len(frames_for_infer), batch_size):
        batch_frames = frames_for_infer[i:i + batch_size]

        # Preprocess frames
        input_tensor = []
        for frame in batch_frames:
            # frame = cv2.resize(frame, (640, 480))  # Resize if needed
            frame = frame.astype(np.float32) / 255.0  # Normalize to [0, 1]
            frame = np.transpose(frame, (2, 0, 1))  # HWC → CHW
            input_tensor.append(frame)

        input_tensor = np.stack(input_tensor)  # Shape: [B, 3, H, W]
        input_tensor = torch.tensor(input_tensor).float().to(device)

        # Inference
        with torch.no_grad():
            heatmaps_2d = model.part_regressor_2d(input_tensor)
            output_3d = model.selec_sls_3d(input_tensor, heatmaps_2d)  # [B, 3, 14]
            inference_outputs.append(output_3d.cpu())

    # Combine all results
    results_3d_out = torch.cat(inference_outputs, dim=0).numpy()

    display_3d_joint_demo(frames_for_infer,results_3d_out, 30)

    for t, joints in enumerate(results_3d_out):
        print(f"Frame {t}:")
        for j in range(joints.shape[1]):
            x, y, z = joints[:, j]
            print(f"  Reference Point {j}: X={x:.2f}, Y={y:.2f}, Z={z:.2f}")


def display_3d_joint_demo(frames, results_3d, fps=10):
    """
    Display a side-by-side view of video frames and corresponding 3D joint coordinates.

    Parameters:
        frames (List[np.ndarray]): List of video frames (H, W, 3)
        results_3d (List[torch.Tensor] or np.ndarray): Corresponding joint predictions (N, 3, 14)
        fps (int): Frames per second for display
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    frame_disp = ax1.imshow(np.zeros_like(frames[0]))
    text_box = ax2.text(0.01, 0.99, '', transform=ax2.transAxes,
                        fontsize=12, verticalalignment='top', family='monospace')

    ax1.axis('off')
    ax1.set_title("Video Frame")
    ax2.axis('off')
    ax2.set_title("3D Joint Coordinates")

    def update(idx):
        frame = frames[idx]
        joints = results_3d[idx]
        if isinstance(joints, torch.Tensor):
            joints = joints.cpu().detach().numpy()

        frame_disp.set_data(frame)

        text = "Joint Coordinates (X, Y, Z):\n\n"
        for i in range(joints.shape[1]):
            x, y, z = joints[:, i]
            text += f"Joint {i:02d}: X={x:7.2f}, Y={y:7.2f}, Z={z:7.2f}\n"

        text_box.set_text(text)
        return frame_disp, text_box

    ani = FuncAnimation(fig, update, frames=len(frames),
                        interval=1000 / fps, blit=False)
    # HTML(anim.to_jshtml())
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    model_selection = sys.argv[1]
    epoch_input = sys.argv[2]
    lr_input = sys.argv[3]
    train_mod = sys.argv[4]

    # Set batch size
    batch_size = 8
    workers = os.cpu_count()
    learning_rate = float(lr_input)
    num_epochs = int(epoch_input)

    # Plotting stuff support
    train_losses = []
    eval_losses = []

    train_mpjpe = []
    eval_mpjpe = []

    train_pa_mpjpe = []
    eval_pa_mpjpe = []

    train_accel_error = []
    eval_accel_error = []

    # Dataset creation and details stuff
    # create_dataset()
    # dataset_details('dataset_tensor_4.pt')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    match model_selection:
        case "normal":
            model = SelecSLSNet()
            print("Normal model selected")
        case "light":
            model = LightweightSelecSLS()
            print("Light model selected")
        case "combined":
            model = MotionCapturePipeline()
            print("Combined model selected")
        case "advanced":
            model = MotionCapturePipelineAdvanced()
            print("Advanced model selected")
        case "enhanced":
            model = MotionCaptureSystem()
            print("Enhanced model selected")
        case _:
            model = SelecSLSNet()
            print("Normal model selected - In Default")

    # Model details
    # summary(model)
    # print(model)

    # loss function - old
    # criterion = nn.MSELoss()

    if train_mod == "train":
        # loss function - new
        criterion = poseMatrix.CombinedPoseLoss()
        optimizer = optim.Adam(model.parameters(), lr = learning_rate)
        model.to(device)

        for dataset_path in dataset_paths:
            # model.train()
            epoch_loss = 0
            train_loader, test_loader = prepare_dataset(dataset_path)
            for epoch in range(num_epochs):
                # del test_loader
                avgerage_loss = train_model(model, train_loader, optimizer, criterion, device)
                val_loss = evaluate_model(model, test_loader, criterion, device)

                # Training Related parameters
                train_losses.append(avgerage_loss['total'])
                train_mpjpe.append(avgerage_loss['mpjpe'])
                train_pa_mpjpe.append(avgerage_loss['pa_mpjpe'])
                train_accel_error.append(avgerage_loss['accel_error'])

                # Evaluation Related Parameters
                eval_losses.append(val_loss['loss'])
                eval_mpjpe.append(val_loss['mpjpe'])
                eval_pa_mpjpe.append(val_loss['pa_mpjpe'])
                eval_accel_error.append(val_loss['accel_error'])

                # epoch_loss += avgerage_loss
            del train_loader, test_loader

        # Save model
        torch.save(model.state_dict(), "mocap_model.pth")

        # Plotting functions
        epochs = range(1, len(train_losses) + 1)

        plt.figure(figsize=(12, 8))

        plt.subplot(2, 2, 1)
        plt.plot(epochs, train_losses, label='Train Total Loss')
        plt.plot(epochs, eval_losses, label='Eval Total Loss')
        plt.title('Total Loss')
        plt.legend()

        plt.subplot(2, 2, 2)
        plt.plot(epochs, train_mpjpe, label='Train MPJPE')
        plt.plot(epochs, eval_mpjpe, label='Eval MPJPE')
        plt.title('MPJPE (mm)')
        plt.legend()

        plt.subplot(2, 2, 3)
        plt.plot(epochs, train_pa_mpjpe, label='Train PA-MPJPE')
        plt.plot(epochs, eval_pa_mpjpe, label='Eval PA-MPJPE')
        plt.title('PA-MPJPE (mm)')
        plt.legend()

        plt.subplot(2, 2, 4)
        plt.plot(epochs, train_accel_error, label='Train Accel Error')
        plt.plot(epochs, eval_accel_error, label='Eval Accel Error')
        plt.title('Acceleration Error (mm/frame²)')
        plt.legend()

        plt.tight_layout()
        plt.show()
    elif train_mod == "datagen":
        # Dataset creation and details stuff
        print("Dataset creation mode")
        # create_dataset()
        # dataset_details('dataset_tensor_4.pt')
    elif train_mod == "infer":
        infer_model(device,"../Datasets/mocap_model_2.pth")
    else:
        print("Wrong Execution Mode")


        # Print average loss for the epoch
        # print(f"Epoch [{epoch + 1}/{num_epochs}], Average Loss in ds: {epoch_loss/len(dataset_paths):.4f}")