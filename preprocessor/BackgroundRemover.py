import cv2

def remove_background(scene_frame, background_frame):
    """
    Removes background from a frame

    Args:
    - scene_frame (str): Original frame.
    - background_frame (int): Image of the background or background frame.

    Returns:
    - processed_frame (np.array):Background removed frame.
    """
    gs_scene = grayscale_converter(scene_frame)
    gs_bckg = grayscale_converter(background_frame)

    inv_mask = create_invmask(gs_scene, gs_bckg)
    cv2.imshow('Mask',inv_mask)
    return cv2.bitwise_and(scene_frame, scene_frame, mask = inv_mask)

def grayscale_converter(source_frame):
    """
        Converts a frame into gray scale

        Args:
        - scene_frame (str): Original frame.

        Returns:
        - processed_frame (np.array):Grayscale frame.
    """

    return cv2.cvtColor(source_frame, cv2.COLOR_BGR2GRAY)

def create_invmask(source_frame, background_frame):
    """
        Creates an inverted mask of the background

        Args:
        - source_frame (str): Original frame.

        Returns:
        - processed_frame (np.array):Background removed frame.
    """
    difference = cv2.absdiff(source_frame, background_frame)
    cv2.imshow('Absolute Diff', difference)
    _, thresh = cv2.threshold(difference, 8, 255, cv2.THRESH_BINARY)
    #return cv2.bitwise_not(thresh)
    return thresh

