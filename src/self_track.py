import numpy as np
import xarray as xr
from collections import OrderedDict
from typing import Any
from scipy.optimize import linear_sum_assignment
from data_handling import detections_h5_to_xr_dataset
from line_profiler import profile

def wrap_angle(theta):
    """Wrap angle to [-pi, pi)."""
    return (theta + np.pi) % (2 * np.pi) - np.pi

class StopAndGoKalmanFilterWithOrientation:
    """
    Kalman filter for animals with:
    - long stationary periods (observed)
    - intermittent motion
    - explicit orientation state
    """

    def __init__(self, x0:float, y0:float, theta0:float, dt:float=1.0, pos_var:float=0.2, vel_var:float=5.0, ang_var:float=0.05, ang_vel_var:float=0.5, meas_var_pos:float=1.0, meas_var_theta:float=0.1, vel_damping:float=0.7, ang_damping:float=0.7, good_pred_factor:float=0.8, bad_pred_factor:float=1.2, miss_factor:float=1.3):
        '''
        x0, y0, theta0: initial state
        dt: time step
        pos_var, vel_var, ang_var, ang_vel_var: process noise variances for position, velocity, angle, and angular velocity
        meas_var_pos, meas_var_theta: measurement noise variances for position and angle
        vel_damping, ang_damping: factors to damp velocity and angular velocity to handle intermittent motion
        good_pred_factor, bad_pred_factor, miss_factor: factors to update process noise based on prediction performance
        '''
        
        self.dt = dt
        self.vel_damping = vel_damping
        self.ang_damping = ang_damping # Damps angular speed

        # State: x, y, vx, vy, theta, omega (dtheta)
        self.x = np.array([x0, y0, 0.0, 0.0, theta0, 0.0])

        self.P = np.diag([1, 1, 10, 10, 0.5, 4.0]) # State uncertainty matrix (current uncertainty beliefs)

        self.F = np.array([[1, 0, dt, 0,  0,  0],
                           [0, 1, 0, dt,  0,  0],
                           [0, 0, vel_damping, 0,  0,  0],
                           [0, 0, 0, vel_damping,  0,  0],
                           [0, 0, 0, 0,  1, dt],
                           [0, 0, 0, 0,  0, ang_damping]])

        # Measurement models
        self.H = np.array([[1, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0],
                           [0, 0, 0, 0, 1, 0]])

        # Process noise (high movement noise versus low)
        self.Q_high = np.diag([pos_var, pos_var, vel_var, vel_var, ang_var, ang_vel_var])
        self.Q_low = np.diag([pos_var * 0.1, pos_var * 0.1, vel_var * 0.01, vel_var * 0.01, ang_var * 0.1, ang_vel_var * 0.1]) # Reduced uncertainties when animal isn't moving
        self.motion_uncertainty = 1 # Blending factor
        self.Q = self.Q_high.copy() # Initial value

        # Measurement noise
        self.R = np.diag([meas_var_pos, meas_var_pos, meas_var_theta])

        self.I = np.eye(6)

        self.prev_z = np.array([x0, y0, theta0])
        self.missed = 0

        # Factors for updating Q
        self.good_pred_factor = good_pred_factor
        self.bad_pred_factor = bad_pred_factor
        self.miss_factor = miss_factor

    def predict(self):
        self.x = self.F @ self.x
        self.x[4] = wrap_angle(self.x[4])
        self.P = self.F @ self.P @ self.F.T + self.Q

    def innovation(self, z): # Pre-fit residual
        """Return innovation vector (with wrapped angle)."""
        y = np.asarray(z) - self.H @ self.x
        y[2] = wrap_angle(y[2])
        return y

    def innovation_cov(self): # Pre-fit residual covariance
        return self.H @ self.P @ self.H.T + self.R
    
    def mahalanobis_distance(self, z):
        y = self.innovation(z)
        S = self.innovation_cov()
        return float(y.T @ np.linalg.inv(S) @ y)
    
    def gating_cost(self, z, gate=9.21):
        """
        Returns inf if outside gate, else Mahalanobis distance. Using Chi squared because of Gaussian assumptions of Kalman filter.
        gate=9.21 ~ chi2(3) 99% (3 degrees of freedom, Chi squared is probability distribution of sum of distance squared, 99% chance it's less than 9.21)
        """
        d2 = self.mahalanobis_distance(z)
        return d2 if d2 <= gate else 1e6
    
    def update_Q(self, z:np.ndarray | None, detected:bool = True):
        '''Update Q based on error.'''
        if detected:
            nis = self.mahalanobis_distance(z) # Normalized innovation squared, used to blend Q values, ~ Chi-squared (k=3)
            if nis < 3.5: # Within 2 STDs (68%)
                self.motion_uncertainty *= self.good_pred_factor # Model predicts motion well, decrease uncertainty
            elif nis > 8.0: # More than ~95%
                self.motion_uncertainty = min(1, self.motion_uncertainty*self.bad_pred_factor)
            # else: leave unchanged

        else:
            self.motion_uncertainty = min(1, self.motion_uncertainty*self.miss_factor) # Increase uncertainty, missed detection

        self.Q = (1 - self.motion_uncertainty)*self.Q_low + self.motion_uncertainty*self.Q_high # Blended Q

    def update(self, z):
        """
        z : (x, y, theta) -> current measurement
        """
        z = np.asarray(z)

        # Pre-fit
        self.predict()
        y = self.innovation(z)
        S = self.innovation_cov()
        K = self.P @ self.H.T @ np.linalg.inv(S) # Optimal Kalman gain

        # Post-fit
        self.x = self.x + K @ y # Post-fit estimate
        self.x[4] = wrap_angle(self.x[4])
        self.P = (self.I - K @ self.H) @ self.P # Post-fit uncertainty matrix

        self.prev_z = z
        self.missed = 0

        # Update Q for next time step
        self.update_Q(z, detected = True)

    def miss(self):
        """
        Call when no detection is assigned.
        """
        self.missed += 1
        self.predict() # Update estimates and state uncertainty matrix

        # Update Q for next time step
        self.update_Q(None, detected = False)

    # Accessors

    def get_position(self):
        return self.x[0:2].copy()

    def get_orientation(self):
        return self.x[4]

    def get_uncertainty_radius(self):
        return np.sqrt(self.P[0, 0] + self.P[1, 1])

    def get_angle_uncertainty(self):
        return np.sqrt(self.P[4, 4])

class TrackState:
    """Enumeration class representing the possible states of an object being tracked.

    Attributes:
        New (int): State when the object is newly detected.
        Tracked (int): State when the object is successfully tracked in subsequent frames.
        Lost (int): State when the object is no longer tracked.
        Removed (int): State when the object is removed from tracking.

    Examples:
        >>> state = TrackState.New
        >>> if state == TrackState.New:
        ...     print("Object is newly detected.")
    """

    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3

class BaseTrack:
    """Base class for object tracking, providing foundational attributes and methods.

    Attributes:
        _count (int): Class-level counter for unique track IDs.
        track_id (int): Unique identifier for the track.
        is_activated (bool): Flag indicating whether the track is currently active.
        state (TrackState): Current state of the track.
        history (OrderedDict): Ordered history of the track's states.
        features (list): List of features extracted from the object for tracking.
        curr_feature (Any): The current feature of the object being tracked.
        score (float): The confidence score of the tracking.
        start_frame (int): The frame number where tracking started.
        frame_id (int): The most recent frame ID processed by the track.
        time_since_update (int): Frames passed since the last update.
        location (tuple): The location of the object in the context of multi-camera tracking.

    Methods:
        end_frame: Returns the ID of the last frame where the object was tracked.
        next_id: Increments and returns the next global track ID.
        activate: Abstract method to activate the track.
        predict: Abstract method to predict the next state of the track.
        update: Abstract method to update the track with new data.
        mark_lost: Marks the track as lost.
        mark_removed: Marks the track as removed.
        reset_id: Resets the global track ID counter.

    Examples:
        Initialize a new track and mark it as lost:
        >>> track = BaseTrack()
        >>> track.mark_lost()
        >>> print(track.state)  # Output: 2 (TrackState.Lost)
    """

    _count = 0

    def __init__(self):
        """Initialize a new track with a unique ID and foundational tracking attributes."""
        self.track_id = 0
        self.is_activated = False
        self.state = TrackState.New
        self.history = OrderedDict()
        self.features = []
        self.curr_feature = None
        self.score = 0
        self.start_frame = 0
        self.frame_id = 0
        self.time_since_update = 0
        self.location = (np.inf, np.inf)

    @property
    def end_frame(self) -> int:
        """Return the ID of the most recent frame where the object was tracked."""
        return self.frame_id

    @staticmethod
    def next_id() -> int:
        """Increment and return the next unique global track ID for object tracking."""
        BaseTrack._count += 1
        return BaseTrack._count

    def activate(self, *args: Any) -> None:
        """Activate the track with provided arguments, initializing necessary attributes for tracking."""
        raise NotImplementedError

    def predict(self) -> None:
        """Predict the next state of the track based on the current state and tracking model."""
        raise NotImplementedError

    def update(self, *args: Any, **kwargs: Any) -> None:
        """Update the track with new observations and data, modifying its state and attributes accordingly."""
        raise NotImplementedError

    @staticmethod
    def reset_id() -> None:
        """Reset the global track ID counter to its initial value."""
        BaseTrack._count = 0

class STrack(BaseTrack): # Single tracklet

    def __init__(self, centroid_pos:np.ndarray, theta:float, frame_id:int, max_frames_lost:int, kf_kwargs:dict = {}):
        self.pos = centroid_pos
        self.theta = theta
        self.frame_id = frame_id

        self.history = OrderedDict()
        self.time_since_update = 0
        self.max_frames_lost = max_frames_lost # Number of frames to keep tracklet alive without updates before marking as removed
        self.kf_kwargs = kf_kwargs

        self.kalman_filter = None

    def activate(self):
        """Activate a new tracklet and initialize its state."""
        self.track_id = self.next_id()
        self.tracklet_len = 1
        self.state = TrackState.Tracked

        self.kalman_filter = StopAndGoKalmanFilterWithOrientation(x0 = self.pos[0], y0 = self.pos[1], theta0 = self.theta, **self.kf_kwargs)

    def predict(self):
        """Predict the next state of the object using the Kalman filter."""
        self.kalman_filter.predict() 
        return self.kalman_filter.x # x_centroid, y_centroid, theta

    def detect(self, frame_id: int, z:np.ndarray):
        """Update the tracklet's history with a new detection and update its state and attributes."""

        self.state = TrackState.Tracked
        self.frame_id = frame_id
        self.time_since_update = 0
        self.tracklet_len += 1

        # Update Kalman filter with new measurement and update position and orientation estimates
        self.kalman_filter.update(z)

        # Store history of actual position, theta
        self.history[frame_id] = z

    def miss(self, frame_id: int):
        """Handle a missed detection by marking the track as lost and updating its state."""
        self.state = TrackState.Lost
        self.time_since_update += 1
        self.frame_id = frame_id

        if self.time_since_update > self.max_frames_lost:
            self.state = TrackState.Removed
        else:
            self.kalman_filter.miss() # Update estimates and increase uncertainty

class MAYOLO:
    """MAYOLO: A tracking algorithm loosely adapted from BYTETracker but meant to be used with precomputed detections (useful when tiling is necessary for getting detections). Adaptions specialize BYTETracker to use orientation to help maintain identities. 
    A custom Kalman filter is used to handle long stationary periods and intermittent motion, which are common in animal tracking.

    This class maintains the state of tracked, lost, and removed tracks over frames and performs data association.

    Attributes:
        tracked_stracks (list[STrack]): List of successfully activated tracks.
        lost_stracks (list[STrack]): List of lost tracks.
        removed_stracks (list[STrack]): List of removed tracks.
        detections_ds (xr.Dataset): Dataset containing precomputed detections for each frame, including centroid coordinates, head and tail positions, and confidence scores.
        frame_id (int): The current frame ID.
        max_frames_lost (int): The maximum frames for a track to be removed permanently.
        kf_kwargs (dict): Keyword arguments for initializing the Kalman filter, allowing customization of process and measurement noise parameters.
        kalman_filter (StopAndGoKalmanFilterWithOrientation): Kalman Filter object.

    Methods:
        update: Update object tracker with new detections.
        get_cost_matrix: Calculate the cost matrix between tracks and detections using Mahalanobis gating on (x, y, orientation) states.
        reset_id: Reset the ID counter of STrack.
        reset: Reset the tracker by clearing all tracks.
        joint_stracks: Combine two lists of stracks.
        sub_stracks: Filter out the stracks present in the second list from the first list.
    """

    def __init__(self, detections_ds: xr.Dataset, init_frame: int = 0, max_frames_lost: int = 5, kf_kwargs: dict = {}):
        """Initialize a MAYOLO instance for object tracking.

        Args:
            detections_ds (xr.Dataset): A dataset containing precomputed detections for each frame, including centroid coordinates, head and tail positions, and confidence scores. 
            The dataset should be structured with dimensions for frames and detections, and contain variables such as 'centroid_x', 'centroid_y', and 'theta'.
        """

        self.lost_stracks: list[STrack] = []
        self.removed_stracks: list[STrack] = []
        self.detections_ds = detections_ds
        self.kf_kwargs = kf_kwargs
        self.frame_id = init_frame
        self.max_frames_lost = max_frames_lost
        self.reset()

        # Initialize tracked_stracks with detections from the first frame
        detections = self.detections_ds.isel(frame=self.frame_id)

        active_stracks: list[STrack] = []
        for idx in range(detections.sizes['id']):
            det_x = detections['centroid_x'].isel(id=idx).values
            det_y = detections['centroid_y'].isel(id=idx).values
            theta = detections['theta'].isel(id=idx).values

            new_track = STrack(np.array([det_x, det_y]), theta, self.frame_id, self.max_frames_lost, self.kf_kwargs)
            new_track.activate()
            active_stracks.append(new_track)
        
        # Initialize list of tracked stracks with the active stracks from the first frame
        self.tracked_stracks: list[STrack] = active_stracks

    @profile
    def update(self):
        """Update the tracker with new detections and return the current list of tracked objects."""

        # Update frame ID and prepare lists for tracking state changes
        self.frame_id += 1
        active_stracks: list[STrack] = []
        lost_stracks: list[STrack] = []
        removed_stracks: list[STrack] = []
        unmatched_detections: list[int] = []

        # Prepare detections for the current frame
        detections = self.detections_ds.isel(frame=self.frame_id).dropna(dim='id') #, subset=['centroid_x', 'centroid_y', 'theta']) # Drop detections with NaN values

        # Step 1: Predict the current location of all tracks (both active and lost) using their Kalman filters, and prepare for data association.
        for track in self.tracked_stracks + self.lost_stracks:
            track.predict()

        # Step 2: Construct cost matrix based on Mahalanobis distance between predicted track states and detections
        cost_matrix, flipped = self.get_cost_matrix(self.tracked_stracks + self.lost_stracks, detections)

        # Step 3: Associate data using the cost matrix and update track states based on matches. Handle unconfirmed tracks separately to avoid incorrect associations.
        track_idcs, detection_idcs = linear_sum_assignment(cost_matrix) # Cost matrix is (num_tracks, num_detections), so this returns indices of matched tracks and detections

        # Step 4: Update matched tracks with assigned detections
        invalid_match = 0
        reactivated_tracks = 0
        for track_idx, detection_idx in zip(track_idcs, detection_idcs):

            track = (self.tracked_stracks + self.lost_stracks)[track_idx]
            
            if cost_matrix[track_idx, detection_idx] == 1e6: # If the cost is above the gating threshold but it was still matched, consider it an unmatched track and detection

                # Deal with track
                track.miss(self.frame_id)
                if track.state == TrackState.Lost:
                    lost_stracks.append(track)
                else:
                    removed_stracks.append(track)

                # Deal with detection
                unmatched_detections.append(detection_idx)
                invalid_match += 1
                continue

            if track.state == TrackState.Lost:
                reactivated_tracks += 1

            det_x = detections['centroid_x'].isel(id=detection_idx).values
            det_y = detections['centroid_y'].isel(id=detection_idx).values
            theta = detections['theta'].isel(id=detection_idx).values

            if flipped[track_idx, detection_idx]: # If this detection was flipped for head/tail confusion, flip it again before updating the track
                theta = wrap_angle(theta + np.pi)

            track.detect(self.frame_id, np.array([det_x, det_y, theta])) # Updates track with new detections and state becomes "Tracked"
            active_stracks.append(track) # Add to active tracks after update
        print('% invalid matches:', round(invalid_match/len(track_idcs), 2))
        print('Num reactivated tracks:', reactivated_tracks)

        # Step 5: Mark unmatched tracks as lost/removed based on how long they've been unmatched
        num_tracks, num_detections = cost_matrix.shape

        for idx in range(num_tracks):
            if idx not in track_idcs:
                track = (self.tracked_stracks + self.lost_stracks)[idx]
                track.miss(self.frame_id)
                if track.state == TrackState.Lost:
                    lost_stracks.append(track)
                else:
                    removed_stracks.append(track)

        # Step 6: Collect the indices for all unmatched detections and initialize new tracks
        for idx in range(num_detections):
            if idx not in detection_idcs:
                unmatched_detections.append(idx)

        for idx in unmatched_detections:
            det_x = detections['centroid_x'].isel(id=idx).values
            det_y = detections['centroid_y'].isel(id=idx).values
            theta = detections['theta'].isel(id=idx).values

            new_track = STrack(np.array([det_x, det_y]), theta, self.frame_id, self.max_frames_lost, self.kf_kwargs)
            new_track.activate()
            active_stracks.append(new_track)
        
        # Step 7: Update the track lists for the next iteration
        self.tracked_stracks = active_stracks
        self.lost_stracks = lost_stracks
        self.removed_stracks += removed_stracks

        print(f"Frame {self.frame_id}: {detections.sizes['id']} detections, {len(active_stracks)} active tracks, {len(lost_stracks)} lost tracks, {len(removed_stracks)} removed tracks")
    
    def get_cost_matrix(self, tracks: list[STrack], detections: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
        """Calculate a cost matrix between tracks and detections using Mahalanobis gating."""
        
        # Extract all detection data into a single array (M, 3)
        det_coords = np.column_stack([detections['centroid_x'].values, detections['centroid_y'].values, detections['theta'].values])
        
        # Initialize cost matrix and flipped matrix
        num_tracks = len(tracks)
        num_dets = len(det_coords)
        cost_matrix = np.full((num_tracks, num_dets), 1e6) # (N, M)
        flipped_matrix = np.zeros((num_tracks, num_dets), dtype=bool)

        # Iterate only over tracks (O(N))
        for i, track in enumerate(tracks):

            # Extract KF properties once per track
            pred_state = track.kalman_filter.H @ track.kalman_filter.x # H is (3, 6), x is (6,), so Hx is (3,)
            S = track.kalman_filter.innovation_cov() # (3, 3)
            S_inv = np.linalg.inv(S)
            
            # Vectorized head/tail flip check
            angles = det_coords[:, 2]
            angle_diff = np.abs(wrap_angle(pred_state[2] - angles)) # Calculate angle difference between predicted orientation and all detection orientations
            
            is_flipped = np.isclose(angle_diff, np.pi, atol=np.deg2rad(15)) # Flip if angle difference is close to 180 degrees
            flipped_matrix[i, :] = is_flipped
            
            # Adjust theta in the detection copy for this track
            current_dets = det_coords.copy()
            current_dets[is_flipped, 2] = wrap_angle(current_dets[is_flipped, 2] + np.pi)
            
            # Vectorized Mahalanobis
            y = current_dets - pred_state # (M, 3)
            y[:, 2] = wrap_angle(y[:, 2])
            
            # Batch Mahalanobis calculation
            d2 = np.einsum('ij,jk,ik->i', y, S_inv, y)
            
            # Apply gate
            gate = 9.21 # 99% confidence for chi-squared with 3 degrees of freedom
            valid = d2 <= gate
            cost_matrix[i, valid] = d2[valid]

        # print(np.sum(cost_matrix < np.inf), "valid associations out of", num_tracks * num_dets)
            
        return cost_matrix, flipped_matrix

    @staticmethod
    def reset_id():
        """Reset the ID counter for STrack instances to ensure unique track IDs across tracking sessions."""
        STrack.reset_id()

    def reset(self):
        """Reset the tracker by clearing all tracked, lost, and removed tracks and reinitializing the Kalman filter."""
        self.tracked_stracks: list[STrack] = []
        self.lost_stracks: list[STrack] = []
        self.removed_stracks: list[STrack] = []
        self.frame_id = 0
        self.reset_id()

    @staticmethod
    def joint_stracks(tlista: list[STrack], tlistb: list[STrack]) -> list[STrack]:
        """Combine two lists of STrack objects into a single list, ensuring no duplicates based on track IDs."""
        exists = {}
        res = []
        for t in tlista:
            exists[t.track_id] = 1
            res.append(t)
        for t in tlistb:
            tid = t.track_id
            if not exists.get(tid, 0):
                exists[tid] = 1
                res.append(t)
        return res

    @staticmethod
    def sub_stracks(tlista: list[STrack], tlistb: list[STrack]) -> list[STrack]:
        """Filter out the stracks present in the second list from the first list."""
        track_ids_b = {t.track_id for t in tlistb}
        return [t for t in tlista if t.track_id not in track_ids_b]
    

def track_kps(detections_h5_path: str, output_dir: str, start_frame:int = 0, end_frame: int | None = None, max_frames_lost: int = 5, kf_kwargs: dict = {}):
    """Track objects in a video using the MAYOLO tracking algorithm with precomputed detections.

    Args:
        detections_h5_path (str): Path to the HDF5 file containing precomputed detections for each frame, structured as an xarray Dataset.
        output_dir (str): Directory where the tracking results will be saved.
        start_frame (int, optional): The starting frame ID for tracking. Defaults to 0.
        end_frame (int | None, optional): The ending frame ID for tracking. If None, tracking will continue until the last frame in the dataset. Defaults to None."""
    
    # Load detections from HDF5 file into an xarray Dataset
    detections_ds = detections_h5_to_xr_dataset(detections_h5_path)

    # Initialize the MAYOLO tracker with the loaded detections
    tracker = MAYOLO(detections_ds, init_frame=start_frame, max_frames_lost=max_frames_lost, kf_kwargs=kf_kwargs)

    # Loop through frames and update the tracker, saving results as needed
    for i in range(start_frame + 1, end_frame or detections_ds.sizes['frame']):
        tracker.update()
    
    print('Number of tracked stracks:', len(tracker.tracked_stracks))
    print('Number of lost stracks:', len(tracker.lost_stracks))
    print('Number of removed stracks:', len(tracker.removed_stracks))