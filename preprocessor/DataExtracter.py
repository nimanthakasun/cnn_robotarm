from numpy.core.records import ndarray
from pyomeca import analogs, Markers
import numpy as np
marker_locations = ["LSHO","LUPA","LELB","LWRA","LWRB","LFRA","LFIN","RSHO","RUPA","RELB","RWRA","RWRB","RFRA","RFIN"]

def load_c3d(source_path):
    """
    Extract data from a video file.

    Args:
    - video_path (str): Path to the video file.
    - num_frames (int): Number of frames to extract (default is 16).

    Returns:
    - frames (np.array): Array of extracted frames.
    """
    return Markers.from_c3d(source_path, prefix_delimiter=":")

def get_marker_array(xarray_object, timestamp, ratio):
    intr_filtered_markers = xarray_object.sel(channel=marker_locations)
    filtered_markers = intr_filtered_markers.isel(time=(timestamp*ratio)).data
    return np.delete(filtered_markers, 3, 0)

def get_mocap_params(xarray_object):
    return xarray_object.dims, xarray_object.axis, xarray_object.channel, xarray_object.time, xarray_object.attrs

def get_ndarray_params(ndarray_object):
    return ndarray_object.ndim, ndarray_object.shape, ndarray_object.size, ndarray_object.dtype

# def get_marker_array_full(xarray_object, scr_frm_cnt):
#     ratio = mocap_sample_count//scr_frm_cnt
#     intr_filtered_markers = xarray_object.sel(channel=marker_locations)
#
#     return np.delete(intr_filtered_markers, 3, 0)