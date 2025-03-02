import os
import torch
from torch.utils.data import DataLoader, Dataset
from datetime import datetime
from preprocessor.FrameExtracter import FrameExtractor
from preprocessor import BackgroundRemover,DataExtracter
import numpy as np

COLOR = True

class VideoDataset(Dataset):
    def __init__(self):
        extracted_frames, c3d_samples = self.load_dataset()
        self.video_frames = extracted_frames
        self.labels = c3d_samples

    def __len__(self):
        return  len(self.video_frames)

    def __getitem__(self, idx):
        frame = self.video_frames[idx]
        label = self.labels[idx]

        frame = frame/255.0

        return  torch.tensor(frame,dtype=torch.float32), torch.tensor(label, dtype=torch.float32) # torch.tensor(frame,dtype=torch.float32),

    def get_paths(self):
        folder_path = '../HumanEva/S1/Image_Data'
        bckg_folder_path = '../HumanEva/Background'
        mocap_folder_path = '../HumanEva/S1/Mocap_Data'
        cur_path = os.getcwd()

        if COLOR:
            # Color Video
            video_path = os.path.join(folder_path, 'Box_1_(C3).avi')
            bckg_video_path = os.path.join(bckg_folder_path, 'Background_1_(C3).avi')
        else:
            # BW video
            video_path = os.path.join(folder_path, 'Box_1_(BW2).avi')
            bckg_video_path = os.path.join(bckg_folder_path, 'Background_1_(BW1).avi')

        mocap_c3d_path = os.path.join(mocap_folder_path, 'Box_1.c3d')
        return video_path, bckg_video_path, mocap_c3d_path

    def load_from_image_source(self, video_path, bckg_video_path):
        no_bckg_frame = []
        frm_extrct = FrameExtractor()
        print("Start scene frame extraction:", datetime.now().strftime("%H:%M:%S"), '\n')
        frame_array, frame_count, frame_width, frame_height, frame_rate, frame_shape= frm_extrct.extract_frames(video_path,2)
        print("End scene Frame calculation:", datetime.now().strftime("%H:%M:%S"), " Frame array of shape: ", frame_shape, '\n')

        print("Start background frame extraction:", datetime.now().strftime("%H:%M:%S"), '\n')
        bckg_array, bgframe_count, bgframe_width, bgframe_height, bgframe_rate, bgframe_shape = frm_extrct.extract_frames(bckg_video_path,2)
        print("End background frame extraction:", datetime.now().strftime("%H:%M:%S"), " Background Frame array of shape: ", bgframe_shape, '\n')

        print("Start background removal:", datetime.now().strftime("%H:%M:%S"), '\n')
        for i in range(frame_count-2):
            no_bckg_frame.append(BackgroundRemover.remove_background(frame_array[i], bckg_array[0]))

        print("End background removal:", datetime.now().strftime("%H:%M:%S"), '\n')

        return frame_count, no_bckg_frame

    def load_from_mocap_source(self, mocap_path, source_frame_count):
        c3d_file = DataExtracter.load_c3d(mocap_path)
        dims, axis, channel, time, attrs = DataExtracter.get_mocap_params(DataExtracter.load_c3d(mocap_path))
        mocap_sample_count = len(time.data)
        ratio = mocap_sample_count//source_frame_count
        marker_array = []
        for i in range(0,source_frame_count):
            marker_array.append(DataExtracter.get_marker_array(c3d_file, i, ratio))

        return marker_array
        # full_marker_array =  DataExtracter.get_marker_array(c3d_file, source_frame_count)
        # shape_of_full_array = full_marker_array.shape
        # mocap_length = shape_of_full_array[2]
        # return self.downsample_mocap(source_frame_count,mocap_length,full_marker_array)

    def load_dataset(self):
        video_path, bckg_video_path, mocap_path = self.get_paths()

        srcframecount, image_numpy_array = self.load_from_image_source(video_path, bckg_video_path)
        mocap_numpy_array = self.load_from_mocap_source(mocap_path, srcframecount)
        return image_numpy_array, mocap_numpy_array

    # def downsample_mocap(self, frame_count, mocap_sample_count, mocap_array):
    #     ratio = mocap_sample_count//frame_count
    #     # return mocap_array[:,:,::int(ratio)]
    #     averaged_mocap = []
    #     # for i in range(frame_count):
    #     #     start = i * ratio
    #     #     end = start + ratio
    #     #     averaged_sample = np.mean(mocap_array[start:end], axis=0)  # Average over the range
    #     #     averaged_mocap.append(averaged_sample)
    #     #
    #     # return np.array(averaged_mocap).transpose(2, 0, 1)
    #
    #     for i in range(frame_count):
    #         start = i * ratio
    #         end = start + ratio
    #         averaged_sample = np.mean(mocap_array[start:end], axis=0)  # Average over the range
    #         averaged_mocap.append(averaged_sample)
    #
    #     start = i * ratio