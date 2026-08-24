'''_____________________________________________________IMPORTS____________________________________________________________'''

import numpy as np
import xarray as xr
import cv2
from pathlib import Path
import pandas as pd
from data_handling import load_preprocessed_data
from tqdm import tqdm
import datetime


'''_____________________________________________________FUNCTIONS____________________________________________________________'''

def get_times_video(video_folder: str, abs_start_frame: int = 0, abs_end_frame: int = None):
    ''' Retrieve numpy array of timestamps for a slice of frames from a video folder.'''

    # STEP 0: Validate inputs
    assert abs_start_frame >= 0, "abs_start_frame must be non-negative."
    assert abs_end_frame is None or abs_end_frame > abs_start_frame, "abs_end_frame must be greater than abs_start_frame."

    # STEP 1: Order file name in the video_folder and get the splice specified by the absolute start and end frame indices
    video_folder = Path(video_folder)
    frame_titles = sorted(list(video_folder.glob('*.jpg')))[abs_start_frame:abs_end_frame]

    # STEP 2: Extract timestamps from the file names and convert to datetime objects
    timestamps = []
    for frame_title in frame_titles:
        # Extract the timestamp from the file name (assuming the format is consistent)
        timestamp_str = '_'.join(frame_title.stem.split('_')[-2:])  # Get the last part of the stem

        # Create a datetime object from the timestamp string
        timestamp = datetime.datetime.strptime(timestamp_str, "%Y-%m-%d_%H-%M-%S-%f")  # Convert to datetime object
        timestamps.append(timestamp)

    return np.array(timestamps)

def get_times_mocap(ts_path:str, csv_path:str, rel_start_frame:int = 0, rel_end_frame:int = None, mocap_fps:int = 25):
    ''' Retrieve numpy array of timestamps for a slice of frames within a single MoCap batch.'''

    # STEP 0: Validate inputs
    assert rel_start_frame >= 0, "rel_start_frame must be non-negative."
    assert rel_end_frame is None or rel_end_frame > rel_start_frame, "rel_end_frame must be greater than rel_start_frame."

    # STEP 1: Find the row containing information for the specified batch_name in the ts_path CSV file
    ts_df = pd.read_csv(ts_path)
    batch_name = csv_path.split('/')[-1].split('.')[0] + '.qtm' # Extract batch name from the csv_path
    batch_row = ts_df[ts_df['relative_path'] == batch_name]

    if not len(batch_row):
        raise ValueError(f"Batch name '{batch_name}' not found in the timestamp CSV file.")

    # STEP 2: Extract the initial timestamp from the batch_row
    timestamp = datetime.datetime.strptime(batch_row['capture_start_local'].values[0], "%Y-%m-%dT%H:%M:%S.%f")  # Convert to datetime object

    # STEP 3: Determine the length of the batch to assure the requested slice is within bounds
    with open(csv_path, mode = 'r') as mocap:
        mocap_df = pd.read_csv(mocap)
        mocap_length = len(np.unique(mocap_df['frame']))

    assert rel_end_frame is None or rel_end_frame <= mocap_length, f"rel_end_frame must be less than or equal to the length of the MoCap batch ({mocap_length})."
    assert rel_start_frame < mocap_length, f"rel_start_frame must be less than the length of the MoCap batch ({mocap_length})."

    # STEP 4: Generate timestamps for the specified slice of frames
    if rel_end_frame is None:
        rel_end_frame = mocap_length

    timestamps = [timestamp + datetime.timedelta(seconds=i/mocap_fps) for i in range(rel_start_frame, rel_end_frame)]

    return np.array(timestamps)

def downsample_mocap_for_video(vid_times:np.ndarray, mocap_times:np.ndarray, video_fps:int = 5):
    ''' Finds indices of MoCap timestamps that are closest to each video timestamp. Returns a list of indices of MoCap timestamps for downsampling the MoCap data.'''

    # STEP 0: Validate inputs
    assert len(vid_times) > 0, "vid_times must not be empty."
    assert len(mocap_times) > 0, "mocap_times must not be empty."

    # STEP 1: Convert datetime arrays to numpy arrays of seconds since epoch for easier comparison
    vid_seconds = np.array([t.timestamp() for t in vid_times])
    mocap_seconds = np.array([t.timestamp() for t in mocap_times])

    # STEP 2: Find the closest MoCap timestamp for each video timestamp
    matched_indices = []
    dists = []
    for vid_time in vid_seconds:
        closest_index = np.argmin(np.abs(mocap_seconds - vid_time))
        matched_indices.append(closest_index)
        dists.append(np.abs(mocap_seconds[closest_index] - vid_time))

    # Check that MoCap timestamps are within 1/video_fps seconds of the video timestamps
    matched_indices = np.array(matched_indices)
    valid_mask = np.array(dists) <= 1/video_fps  # Create a boolean mask for valid matches

    return matched_indices, valid_mask

vid_time = get_times_video('/original/20230329/video/', abs_start_frame=2530, abs_end_frame=2540)
mocap_time = get_times_mocap('/mocap/20230329/qtm_capture_times.csv', '/mocap/20230329/csvs/10K_Marching_0049.csv', rel_start_frame=0, rel_end_frame=50, mocap_fps=25)

matched, valid = downsample_mocap_for_video(vid_time, mocap_time, video_fps=5)
print("Video timestamps:", vid_time)
print("MoCap timestamps:", mocap_time)
print("Matched MoCap indices:", matched)
print("Valid matches:", valid)
