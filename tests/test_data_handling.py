import h5py
import numpy as np
import pytest

from src.data_handling import kp_detections_to_xr


def test_kp_detections_to_xr_preserves_point_order_across_frames(tmp_path):
    h5_path = tmp_path / "detections.h5"

    with h5py.File(h5_path, "w") as f:
        for frame_idx, centroids in enumerate(
            [np.array([[1.0, 10.0], [2.0, 20.0]]), np.array([[5.0, 50.0], [6.0, 60.0]])]
        ):
            grp = f.create_group(f"f{frame_idx}")
            grp.create_dataset("centroid", data=centroids)
            grp.create_dataset("head", data=np.array([[100.0, 100.0], [200.0, 200.0]]))
            grp.create_dataset("tail", data=np.array([[300.0, 300.0], [400.0, 400.0]]))

    ds = kp_detections_to_xr(str(h5_path), calibration_path=None, start_frame=0, end_frame=2, subsample=1)

    assert ds["centroid_x"].values.tolist() == [[1.0, 2.0], [5.0, 6.0]]
    assert ds["centroid_y"].values.tolist() == [[10.0, 20.0], [50.0, 60.0]]
