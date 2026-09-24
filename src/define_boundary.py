import numpy as np
import matplotlib.pyplot as plt
import cv2

def fit_circle_least_squares(x, y):
    """
    Fit circle to points using linear least squares.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    A = np.c_[2*x, 2*y, np.ones_like(x)]
    b = x**2 + y**2

    c, residuals, _, _ = np.linalg.lstsq(A, b, rcond=None)

    cx, cy = c[0], c[1]
    r = np.sqrt(c[2] + cx**2 + cy**2)

    return np.array([cx, cy]), r


def get_arena_circle_from_clicks(dewarped_img_path:str, world_bounds:dict, px_per_m:float, n_clicks=6):
    """
    Display (dewarped) image and let user click boundary points to estimate arena circle.
    """

    assert n_clicks >= 3, "At least 3 clicks are required to fit a circle."

    # Load image
    image = cv2.imread(dewarped_img_path, cv2.IMREAD_GRAYSCALE)

    # Create display figure for user to click
    fig, ax = plt.subplots()
    ax.imshow(image, cmap="gray")
    ax.set_title(f"Click {n_clicks} points along the arena boundary")
    pts = plt.ginput(n_clicks, timeout=0)
    plt.close(fig)

    # First use pixel units
    pts_px = np.array(pts)
    x_px = pts_px[:, 0]
    y_px = pts_px[:, 1]

    center_px, r_px = fit_circle_least_squares(x_px, y_px)

    # Now convert to world units using the calibration
    xmin = world_bounds["xmin"]
    ymax = world_bounds["ymax"]
    center_wrld = np.array([center_px[0] / px_per_m + xmin, ymax - center_px[1] / px_per_m])
    r_wrld = r_px / px_per_m

    return center_px, r_px, center_wrld, r_wrld

if __name__ == "__main__":

    print('Warning: must be used outside of VSCode to allow for interactive clicking on the image.')
    img_path = '/Users/mayadagher/Documents/Locusts/hangar/original/20230329/video_dewarped/65MP01_10Kmarching_01_2023-03-29_10-10-24-124_dewarped.png'
    # img_path = '/original/20230329/video/65MP01_10Kmarching_01_2023-03-29_10-10-24-124.jpg'
    # calibration_path = '/intrinsics/arena_board_calibration/calibration_official.yaml'
    world_bounds = {'xmin': -2.5905300000000007, 'xmax': 2.5905300000000007, 'ymin': -2.5905300000000007, 'ymax': 2.5905300000000007}
    px_per_m = 1454.2180013974423

    center_px, r_px, center_wrld, r_wrld = get_arena_circle_from_clicks(img_path, world_bounds, px_per_m, n_clicks=6)

    print(f"Estimated arena center (pixels): ({center_px[0]:.2f}, {center_px[1]:.2f}), radius: {r_px:.2f}")
    print(f"Estimated arena center (world units): ({center_wrld[0]:.2f}, {center_wrld[1]:.2f}), radius: {r_wrld:.2f}")