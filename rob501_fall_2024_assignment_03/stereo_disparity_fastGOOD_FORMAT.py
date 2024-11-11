# COPY PASTE THIS INTO stereo_disparity_fast cause formatting in that file messed up

import numpy as np
from scipy.ndimage.filters import *

def stereo_disparity_fast(Il, Ir, bbox, maxd):
    """
    Fast stereo correspondence algorithm.

    This function computes a stereo disparity image from left stereo 
    image Il and right stereo image Ir. Only disparity values within
    the bounding box region (inclusive) are evaluated.

    Parameters:
    -----------
    Il    - Left stereo image, m x n pixel np.array, greyscale.
    Ir    - Right stereo image, m x n pixel np.array, greyscale.
    bbox  - 2x2 np.array, bounding box, relative to left image, from top left
            corner to bottom right corner (inclusive). (x, y) in columns.
    maxd  - Integer, maximum disparity value; disparities must be within zero
            to maxd inclusive (i.e., don't search beyond maxd).

    Returns:
    --------
    Id  - Disparity image (map) as np.array, same size as Il.
    """
    # Hints:
    #
    #  - Loop over each image row, computing the local similarity measure, then
    #    aggregate. At the border, you may replicate edge pixels, or just avoid
    #    using values outside of the image.
    #
    #  - You may hard-code any parameters you require in this function.
    #
    #  - Use whatever window size you think might be suitable.
    #
    #  - Optimize for runtime AND for clarity.

    #--- FILL ME IN ---
    print("I AM HERE")
    quit()
    # Define some params: window size (nxn pixel value)
    window = 5
    half_window = window // 2

    # pad both images by half window size (puts zeros around the image)
    Il_padded = np.pad(Il, half_window, mode='constant') 
    Ir_padded = np.pad(Ir, half_window, mode='constant')

    # Get dimensions that we have to search from bbox
    x_min, y_min = bbox[0]
    x_max, y_max = bbox[1]
    # Adjust box dimensions to account for padding
    x_min_padded = x_min + half_window
    x_max_padded = x_max + half_window
    y_min_padded = y_min + half_window
    y_max_padded = y_max + half_window

    # loop through each pixel in the bounding box: Everything in the double loop has padding accounted
    for y in range(y_min_padded, y_max_padded + 1):
        for x in range(x_min_padded, x_max_padded + 1):
            # get the window around the pixel: padding should guarantee we never go out of bounds
            left_win = Il_padded[y - half_window : y + half_window + 1, x - half_window : x + half_window + 1]

            # initialize the best disparity and the best score
            best_sad = np.inf
            best_d = 0

            # Look for best match within d <= maxd (d = x_left - x_right), x,y = x_left, y_left
            for d in range(maxd + 1):
                # get x_right and corresponding window in right img
                x_right = x - d
                # Check if x_right is within bounds:
                #   1. x_right outside on the left: x_right - half_window < 0
                #   2. x_right outside on the right: x_right + half_window >= Ir_padded.shape[1]
                if x_right - half_window < 0 and x_right + half_window >= Ir_padded.shape[1]:
                    continue # x_right is out of bounds, go next d
                right_win = Ir_padded[y - half_window : y + half_window + 1, x_right - half_window : x_right + half_window + 1]

                # compute SAD score
                sad = np.sum(np.abs(left_win - right_win))

                # if current SAD score is better (less) than best_sad, update best disparity
                if sad < best_sad:
                    best_sad = sad
                    best_d = d

            # Get "original coordinates" (i.e. w/o pad)
            x_orig = x - half_window
            y_orig = y - half_window

            # set disparity value in disparity map to best disparity
            Id[y_orig, x_orig] = best_d
    #------------------

    correct = isinstance(Id, np.ndarray) and Id.shape == Il.shape

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return Id