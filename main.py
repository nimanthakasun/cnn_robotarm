import os
import json
import torch
from preprocessor.DataSet import VideoDataset
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import torch.nn as nn
import torch.optim as optim
import gc
import sys
from stagetwo import selecsls, selecslsMod, selecslslight, combineModel
from torchinfo import summary
from pyomeca import analogs, Markers
from preprocessor import DataExtracter

# marker_locations = ["LSHO","LUPA","LELB","LWRA","LWRB","LFRA","LFIN","RSHO","RUPA","RELB","RWRA","RWRB","RFRA","RFIN"]
# marker_location = ["LSHO"]
dataset_paths = ['dataset_tensor_1.pt', 'dataset_tensor_2.pt', 'dataset_tensor_3.pt']

from stagetwo.selecslsMod import SelecSLSNet
from stagetwo.selecslslight import LightweightSelecSLS
from stagetwo.combineModel import MotionCapturePipeline

# ###############################################   Dataset creation   ###################################################
def create_dataset():
    dataset_tensor = VideoDataset()
    sample_frame,sample_label = dataset_tensor[0]
    print("Video shape: ", sample_frame.shape)
    print("Label shape: ", sample_label.shape)
    print("Saving Dataset:")
    torch.save(dataset_tensor, "dataset_tensor_3.pt")
    print(dataset_tensor.__len__())

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
    workers = 2

    # Create DataLoaders for training and testing
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)

    return train_loader, test_loader

def train_model(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    batch_iteration = 0

    optimizer.zero_grad()
    for i, (video_frames, labels) in enumerate(loader):
        video_frames_rearranged = video_frames.permute(0, 3, 1, 2)
        del video_frames
        video_frames_rearranged = video_frames_rearranged.float().to(device)
        labels = labels.float().to(device)

        # Forward pass
        outputs = model(video_frames_rearranged)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss = loss / accumulation_steps
        loss.backward()
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.detach().item()
        del loss, outputs

    # Print average loss for the epoch
    return total_loss / len(train_loader)
    # print(f"Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss:.4f}")

if __name__ == '__main__':
    model_selection = sys.argv[1]
    # Set batch size
    batch_size = 8
    workers = 2
    learning_rate = 0.05
    num_epochs = 250
    accumulation_steps = 4

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
        case _:
            model = SelecSLSNet()
            print("Normal model selected - In Default")

    #loss function
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    model.to(device)
    for epoch in range(num_epochs):
        # model.train()
        epoch_loss = 0

        for dataset_path in dataset_paths:
            # Forward pass

            train_loader, test_loader = prepare_dataset(dataset_path)
            del test_loader
            avgerage_loss = train_model(model, train_loader, optimizer, criterion, device)
            epoch_loss += avgerage_loss
            del train_loader

        # Print average loss for the epoch
        print(f"Epoch [{epoch + 1}/{num_epochs}], Average Loss: {epoch_loss/len(dataset_paths):.4f}")