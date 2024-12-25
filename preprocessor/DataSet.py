import os
import torch
from torch.utils.data import DataLoader, Dataset
from datetime import datetime
from preprocessor import FrameExtracter, BackgroundRemover,DataExtracter
import numpy as np

COLOR = True

class VideoDataset(Dataset):
    def __init__(self):
        extracted_frames, c3d_samples = self.load_dataset()
        self.video_frames = extracted_frames
        self.labels = c3d_samples

    def __len__(self):
        return len(self.video_frames)

    def __getitem__(self, idx):
        frame = self.video_frames[idx]
        label = self.labels[idx]

        return torch.tensor(frame,dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

    def get_paths(self):
        folder_path = '../HumanEva/S1/Image_Data'
        bckg_folder_path = '../HumanEva/Background'
        mocap_folder_path = '../HumanEva/S1/Mocap_Data'
        cur_path = os.getcwd()

        if COLOR:
            # Color Video
            video_path = os.path.join(folder_path, 'Box_1_(C1).avi')
            bckg_video_path = os.path.join(bckg_folder_path, 'Background_1_(C1).avi')
        else:
            # BW video
            video_path = os.path.join(folder_path, 'Box_1_(BW1).avi')
            bckg_video_path = os.path.join(bckg_folder_path, 'Background_1_(BW1).avi')

        mocap_c3d_path = os.path.join(mocap_folder_path, 'Box_1.c3d')
        return video_path, bckg_video_path, mocap_c3d_path

    def load_from_image_source(self, video_path, bckg_video_path):
        no_bckg_frame = []
        print("Start scene frame extraction:", datetime.now().strftime("%H:%M:%S"), '\n')
        frame_array, frame_count, frame_width, frame_height, frame_rate, frame_shape= FrameExtracter.extract_frames(video_path,2)
        print("End scene Frame calculation:", datetime.now().strftime("%H:%M:%S"), " Frame array of shape: ", frame_shape, '\n')

        print("Start background frame extraction:", datetime.now().strftime("%H:%M:%S"), '\n')
        bckg_array, bgframe_count, bgframe_width, bgframe_height, bgframe_rate, bgframe_shape = FrameExtracter.extract_frames(bckg_video_path,2)
        print("End background frame extraction:", datetime.now().strftime("%H:%M:%S"), " Background Frame array of shape: ", bgframe_shape, '\n')

        print("Start background removal:", datetime.now().strftime("%H:%M:%S"), '\n')
        for i in range(frame_count-2):
            no_bckg_frame.append(BackgroundRemover.remove_background(frame_array[i], bckg_array[0]))

        print("End background removal:", datetime.now().strftime("%H:%M:%S"), '\n')

        return no_bckg_frame

    def load_from_mocap_source(self, mocap_path):
        c3d_file = DataExtracter.load_c3d(mocap_path)
        marker_array = []
        for i in range(5):
            marker_array.append(DataExtracter.get_marker_array(c3d_file, i))

        return marker_array

    def load_dataset(self):
        video_path, bckg_video_path, mocap_path = self.get_paths()

        image_numpy_array = self.load_from_image_source(video_path, bckg_video_path)
        mocap_numpy_array = self.load_from_mocap_source(mocap_path)

        return image_numpy_array, mocap_numpy_array