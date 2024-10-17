import cv2
import numpy as np

def extract_frames(video_path, num_frames=16):
    """
    Extract frames from a video file.

    Args:
    - video_path (str): Path to the video file.
    - num_frames (int): Number of frames to extract (default is 16).

    Returns:
    - frames (np.array): Array of extracted frames.
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    frames = []

    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Get the video height and width
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    # Calculate the interval at which frames will be extracted
    frame_interval = max(total_frames // num_frames, 1)

    # Iterate through the frames and extract them
    for i in range(num_frames):
        # Set the frame position to the current frame index
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * frame_interval)
        # Read the frame
        ret, frame = cap.read()

        # Break the loop if the end of the video is reached
        if not ret:
            break
        # Resize the frame to 112x112 pixels
        '''
        frame = cv2.resize(frame, (112, 112))
        '''

        # Append the frame to the frames list
        frames.append(frame)

    # Release the video capture object
    cap.release()

    # Fill any missing frames with blank (zero) frames
    '''
    while len(frames) < num_frames:
        frames.append(np.zeros((frame_height, frame_width, 3), np.uint8))
    '''
    # Convert the frames list to a NumPy array
    return np.array(frames)