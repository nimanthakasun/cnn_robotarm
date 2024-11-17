from scipy.io import whosmat
#from tensorflow.tools.docs.doc_controls import header

from preprocessor import FrameExtracter, BackgroundRemover,CropandResize
import os
import cv2
import numpy as np
import scipy.io as sio
from pyomeca import analogs, Markers

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

    frame_array = FrameExtracter.extract_frames(video_path,2)
    bckg_array = FrameExtracter.extract_frames(bckg_video_path,2)

    no_bckg_frame = BackgroundRemover.remove_background(frame_array[1], bckg_array[1])

    #cv2.imshow("Original", frame_array[1])
    #cv2.imshow("Background", bckg_array[1])

    #cv2.imshow("BG Removed", no_bckg_frame)

    mocap_path = os.path.join(mat_folder_path, 'Box_1.mat')
    mocap_c3d_path = os.path.join(mat_folder_path, 'Box_1.c3d')
    mat_contents = sio.loadmat(mocap_path)

    markers = Markers.from_c3d(mocap_c3d_path, prefix_delimiter=":")
    #print(markers)
    print(markers.sel(channel="LASI", time=10))