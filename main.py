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

from stagetwo.selecslsMod import SelecSLSNet
from stagetwo.selecslslight import LightweightSelecSLS
from stagetwo.combineModel import MotionCapturePipeline

if __name__ == '__main__':
    with open('common_configs.json', 'r') as configfile:
        configuration_data = json.load(configfile)

    model_selection = sys.argv[1]

    # mocap_folder_path = '../HumanEva/S1/Mocap_Data'
    # mocap_c3d_path = os.path.join(mocap_folder_path, 'Box_1.c3d')
    # dims, axis, channel, time, attrs = DataExtracter.get_mocap_params(DataExtracter.load_c3d(mocap_c3d_path))
    # print(len(time.data))

    # c3d_data_xarray = Markers.from_c3d(os.path.join('../HumanEva/S1/Mocap_Data', 'Box_1.c3d'), prefix_delimiter=":")
    # marker_array = c3d_data_xarray.sel(channel=marker_locations).data
    # zeros_deleted = np.delete(marker_array, 3, 0)
    #
    # downsampled_moap = zeros_deleted[:,:,::3]
    # print(type(marker_array))
    # print(marker_array.shape)
    # print("--------------- Before Delete ---------------")
    # print(marker_array)
    # print("--------------- After Delete ---------------")
    # print(zeros_deleted)
    # print(zeros_deleted.shape)
    # print("--------------- After Downsample ---------------")
    # print(downsampled_moap)
    # print(downsampled_moap.shape)

###############################################   Dataset creation   ###################################################
#     dataset_tensor = VideoDataset()
#     sample_frame,sample_label = dataset_tensor[0]
#     print("Video shape: ", sample_frame.shape)
#     print("Label shape: ", sample_label.shape)
#     print("Saving Dataset:")
#     torch.save(dataset_tensor, "dataset_tensor_3.pt")
#     print(dataset_tensor.__len__())

###############################################   Dataset using - Init #################################################
    loaded_dataset = torch.load('dataset_tensor_2.pt')
    video_frames = loaded_dataset.video_frames
    labels = loaded_dataset.labels

    print("Video array shape: ", np.array(video_frames).shape)
    print("Label array shape: ", np.array(labels).shape)
    print("Video length: ", len(video_frames))
    print("Label length: ", len(labels))
    one_frame, one_label = loaded_dataset[0]
    print("Video frame shape: ", one_frame.shape[1])
    print("Label frame shape: ", one_label.shape[1])
    print("Length of Video frame array", len(loaded_dataset.video_frames))
    print("Length of labels frame array", len(loaded_dataset.labels))

    if len(loaded_dataset.labels) > len(loaded_dataset.video_frames):
        loaded_dataset.labels = loaded_dataset.labels[:len(loaded_dataset.video_frames)]
        print("Modified label length: ", len(loaded_dataset.labels))

    del video_frames, labels
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

    # input_size = loaded_dataset.
    # sample_images, sample_labels = next(iter(train_loader))
    # print(f"Shape of one image: {sample_images[10].shape}")  # Shape of a single image
    # print(f"Shape of one label: {sample_labels[10].shape}")  # Shape of a single label

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

    # model.load_state_dict(torch.load("selec_sls_motion_capture.pth"))

#     print("--------------- Normal-----------")
#     summary(model)
#     print("--------------- Normal-----------")
#     summary(model2)

    learning_rate = 0.05
    num_epochs = 250
    accumulation_steps = 4

    #loss function
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        batch_iteration = 0

        optimizer.zero_grad()
        for i, (video_frames, labels) in enumerate(train_loader):
            video_frames_rearranged = video_frames.permute(0, 3, 1, 2)
            del video_frames
            video_frames_rearranged = video_frames_rearranged.float().to(device)
            labels = labels.float().to(device)

            # Forward pass
            outputs = model(video_frames_rearranged)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss = loss/accumulation_steps
            loss.backward()
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.detach().item()
            del loss, outputs

        # Print average loss for the epoch
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), "selec_sls_motion_capture.pth")

    print("Training completed!")

# def load_dataset(dataset_path):
#     return torch.load(dataset_path)
#
# def train_model(model, loader, optimizer, criterion, device):
#     model.train()
#     total_loss = 0
#     total_batches = 0
#
#     video_frames = loaded_dataset.video_frames
#     labels = loaded_dataset.labels
#
#     for batch