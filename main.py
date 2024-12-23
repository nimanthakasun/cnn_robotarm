from scipy.io import whosmat

from preprocessor import FrameExtracter, BackgroundRemover,CropandResize, DataExtracter
import os
import cv2
import numpy as np
import scipy.io as sio
from pyomeca import analogs, Markers
from UI import Window
from datetime import datetime

COLOR = True

if __name__ == '__main__':
    folder_path = '../HumanEva/S1/Image_Data'
    bckg_folder_path = '../HumanEva/Background'
    mat_folder_path = '../HumanEva/S1/Mocap_Data'
    cur_path = os.getcwd()

    if COLOR:
        #Color Video
        video_path = os.path.join(folder_path, 'Box_1_(C1).avi')
        bckg_video_path = os.path.join(bckg_folder_path, 'Background_1_(C1).avi')
    else:
        #BW video
        video_path = os.path.join(folder_path, 'Box_1_(BW1).avi')
        bckg_video_path = os.path.join(bckg_folder_path, 'Background_1_(BW1).avi')

    print("Start scene frame extraction:", datetime.now().strftime("%H:%M:%S"), '\n')
    # frame_array, frame_count, frame_width, frame_height, frame_rate= FrameExtracter.extract_frames(video_path,2)
    print("End scene Frame calculation:", datetime.now().strftime("%H:%M:%S"), '\n')

    print("Start background frame extraction:", datetime.now().strftime("%H:%M:%S"), '\n')
    # bckg_array, bgframe_count, bgframe_width, bgframe_height, bgframe_rate = FrameExtracter.extract_frames(bckg_video_path,2)
    print("End background frame extraction:", datetime.now().strftime("%H:%M:%S"), '\n')

    print("Start background removal:", datetime.now().strftime("%H:%M:%S"), '\n')
    # no_bckg_frame = BackgroundRemover.remove_background(frame_array[1], bckg_array[1])
    print("End background removal:", datetime.now().strftime("%H:%M:%S"), '\n')
    mocap_path = os.path.join(mat_folder_path, 'Box_1.mat')
    mocap_c3d_path = os.path.join(mat_folder_path, 'Box_1.c3d')
    mat_contents = sio.loadmat(mocap_path)

    c3d_file = DataExtracter.load_c3d(mocap_c3d_path)
    dimensions, axises, channels, times, attibutes = DataExtracter.get_mocap_params(c3d_file)
    #print(dimensions, axises, channels, times, attibutes)
    marker_array = DataExtracter.get_marker_array(c3d_file,3)
    # print(marker_array)
    # print(frame_count, frame_rate)
    # print("Frame array size: ", frame_array.shape)
    # Window.show_window()

    #cv2.imshow("Original", frame_array[1])
    #cv2.imshow("Background", bckg_array[1])
    #cv2.imshow("BG Removed", no_bckg_frame)


