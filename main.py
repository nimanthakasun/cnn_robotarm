from preprocessor import FrameExtracter, BackgroundRemover
import os
import cv2

COLOR = True

if __name__ == '__main__':
    folder_path= 'C:/Users/kasun/Downloads/Compressed/S1/Image_Data'
    bckg_folder_path = 'C:/Users/kasun/Downloads/Compressed/Background'
    cur_path = os.getcwd()
    #relative_path = '../../../../Downloads/Compressed/S1/Image_Data/Box_1_(BW1).avi'

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

    cv2.imshow("Original", frame_array[1])
    cv2.imshow("Background", bckg_array[1])

    cv2.imshow("BG Removed", no_bckg_frame)


    cv2.waitKey(0)