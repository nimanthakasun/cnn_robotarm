import os
import json
import torch
from preprocessor.DataSet import VideoDataset

if __name__ == '__main__':
    with open('common_configs.json', 'r') as configfile:
        configuration_data = json.load(configfile)

    # print(marker_array)
    # print(frame_count, frame_rate)
    # print("Frame array size: ", frame_array.shape)
    # Window.show_window()

    #cv2.imshow("Original", frame_array[1])
    #cv2.imshow("Background", bckg_array[1])
    #cv2.imshow("BG Removed", no_bckg_frame)

    dataset_tensor = VideoDataset()

    sample_frame,sample_label = dataset_tensor[0]
    print("Video shape: ", sample_frame.shape)
    print("Label shape: ", sample_label.shape)
    print("Saving Dataset:")
    torch.save(dataset_tensor, "dataset_tensor.pt")
    print(dataset_tensor.__len__())