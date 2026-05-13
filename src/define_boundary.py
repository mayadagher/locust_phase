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

    return cx, cy, r


def get_arena_circle_from_clicks(img_path, n_clicks=6):
    """
    Display image and let user click boundary points to estimate arena circle.

    Parameters
    ----------
    img_path : str
        Path to the image file
    n_clicks : int
        Number of boundary points to click (>=3)

    Returns
    -------
    cx, cy, r
    """

    assert n_clicks >= 3, "At least 3 clicks are required to fit a circle."

    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    fig, ax = plt.subplots()
    ax.imshow(image, cmap="gray")
    ax.set_title(f"Click {n_clicks} points along the arena boundary")

    pts = plt.ginput(n_clicks, timeout=0)
    plt.close(fig)

    pts = np.array(pts)
    x = pts[:, 0]
    y = pts[:, 1]

    cx, cy, r = fit_circle_least_squares(x, y)

    return cx, cy, r

if __name__ == "__main__":

    img_path = '/Users/mayadagher/Documents/Locusts/hangar/original/20230329/video/65MP01_10Kmarching_01_2023-03-29_10-10-24-124.jpg'
    cx, cy, r = get_arena_circle_from_clicks(img_path)
    print(f"Estimated arena center: ({cx:.2f}, {cy:.2f}), radius: {r:.2f}")