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
from stagetwo import selecsls, selecslsMod, selecslslight
from torchinfo import summary

from stagetwo.selecslsMod import SelecSLSNet
from stagetwo.selecslslight import LightweightSelecSLS

if __name__ == '__main__':
    with open('common_configs.json', 'r') as configfile:
        configuration_data = json.load(configfile)

    model_selection = sys.argv[1]

# Dataset creation
#     dataset_tensor = VideoDataset()
#     sample_frame,sample_label = dataset_tensor[0]
#     print("Video shape: ", sample_frame.shape)
#     print("Label shape: ", sample_label.shape)
#     print("Saving Dataset:")
#     torch.save(dataset_tensor, "dataset_tensor.pt")
#     print(dataset_tensor.__len__())

# Dataset using
    loaded_dataset = torch.load('dataset_tensor.pt')
    video_frames = loaded_dataset.video_frames
    labels = loaded_dataset.labels
    video_length = len(loaded_dataset)
    print("Dataset length: ", video_length)
    print("Video array shape: ", np.array(video_frames).shape)
    print("Label array shape: ", np.array(labels).shape)
    print("Video length: ", len(video_frames))
    print("Label length: ", len(labels))
    one_frame, one_label = loaded_dataset[0]
    print("Video frame shape: ", one_frame.shape[1])
    print("Label frame shape: ", one_label.shape[1])

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
    workers = 4

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
        case _:
            model = SelecSLSNet()
            print("Normal model selected - In Default")

#     print("--------------- Normal-----------")
#     summary(model)
#     print("--------------- Normal-----------")
#     summary(model2)

    learning_rate = 0.01
    num_epochs = 20
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


    # model = selecsls.Net('SelecSLS60')
    # criterion = nn.MSELoss(reduction='none')
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    #
    # print("--------------------- Model goes from here -------------------")
    # # print(model)
    # # summary(model)
    #
    # epochs = 5
    # for epoch in range(epochs):
    #     model.train()
    #     total_loss = 0.0
    #     for video_frames, labels in train_loader:
    #         video_frames = video_frames.float()  # Convert to float if not already
    #         labels = labels.float()  # Convert labels to float
    #
    #         # print("Video array shape in loop: ", np.array(video_frames).shape)
    #         video_frames_rearranged = video_frames.permute(0, 3, 1, 2)
    #
    #         print("Input shape:", video_frames_rearranged.shape)
    #         print("Label shape:", labels.shape)
    #
    #         # print("Changed array shape in loop: ", np.array(video_frames_rearranged).shape)
    #         # Forward pass
    #         # outputs = model(video_frames_rearranged)
    #         # labels = labels.view(outputs.shape)
    #         # loss = criterion(outputs, labels)
    #
    #         # Backward pass and optimization
    #         # optimizer.zero_grad()
    #         outputs = model(video_frames_rearranged)
    #         print("Output shape:", outputs.shape)
    #         break
    #         loss = criterion(outputs, labels)
    #         loss.backward()
    #         optimizer.step()
    #
    #         # total_loss += loss.item()
    #
    #     # print(f"Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(train_loader):.4f}")
    #     print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")