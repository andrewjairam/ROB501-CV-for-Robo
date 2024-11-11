# COPY PASTE THIS INTO stereo_dispary_best.py (formatting messed up in that file o)

import numpy as np
from scipy.ndimage.filters import *

def stereo_disparity_best(Il, Ir, bbox, maxd):
    """
    Best stereo correspondence algorithm.

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

    -------------------------------------------------------------------------
    Algorithm implemented: Semi global matching (SGM) by Hirschmuller, Citation:
    Hirschmüller, H. (2005). Stereo Processing by Semi-Global Matching and Mutual Information. IEEE CVPR, 2005. doi:10.1109/CVPR.2005.56

    Idea: aggregate matching cost from multiple directions and minimize the cost to find the disparity map. This consists of the following key steps:
    1. Create a "cost volume" by splitting each pixel in the right image into a 3D entry, where the third dimension is each disparity level
    2. Compute matching costs: for each disparity level in the range [0, maxd], compute the matching cost for each pixel in the right image, store
       in the cost volume
       - can do this by "rolling" the right image by some disparity, and computing absolute differences 
    3. Aggregate costs: for each pixel in the cost volume, compute the aggregate cost by summing the matching costs from multiple directions:
        - i.e. loop through a number of directions (up, down, left, right, diagonals) and sum the matching costs.
        - This step enforces a smoothness constraint on the disparity map. It is also recommended to include a small penalty, P1, for small disparity
          changes and a larger penalty, P2, for larger disparity changes
        - the aggregate cost can be computed in a number of directions, commonly:
            - 4 (up, down, left, right)
            - 8 (4 + diagonals)
    4. For each pixel, select disparity with lowest aggregated cost: which represents the best match in the left image
    5. Return the disparity map (Id) using these disparities

    This method is more computationally expensive than local methods, but less expensive than some of the global methods discussed in the Scharstein paper
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

    # 0. Initialize parameters: Can tune this stuff
    height, width = Il.shape # m = height, n = width
    # Directions: 0 if "off", "1" if on. ex: (-1,0) means left, (-1,1) means top left, etc.
    directions = [(0, -1), (0, 1), (-1, 0), (1, 0)] # 4 directions
    # P1 usually ~5-10, P2 usually ~120-150\
    P1 = 10
    P2 = 150
    # Initialize Id
    Id = np.empty(Il.shape)


    # 1. Create a "cost volume" by splitting each pixel in the right image into a 3D entry, where the third dimension is each disparity level
    cost_volume = np.full((height, width, maxd + 1), np.inf) # (3d in shape of (height, width, maxd + 1)... initialize values high since we want to return the lowest cost)

    # 2. Compute matching costs: for each disparity level in the range [0, maxd], compute the matching cost for each pixel in the right image, store in the cost volume
    for d in range(maxd + 1):
        # Shift all pixels in right image to the left by disparity level d (i.e. roll by -d)
        Ir_leftshifted = np.roll(Ir, -d, axis=1)
        # Compute matching cost for each pixel in the right image
        matching_cost = np.abs(Il - Ir_leftshifted)
        # Set costs of pixels that rolled to the other side of the image to infinity
        # WHY? roll pushes pixels left (by d), piels pushed off the boundary are rolled over to the other side of img. we dont care about these, so set inf
        matching_cost[:, :d] = np.inf
        # Store in cost volume
        cost_volume[:, :, d] = matching_cost

    # 3. Aggregate costs: for each pixel in the cost volume, compute the aggregate cost by summing the matching costs from multiple directions
    # Initialize the aggregated cost volume: same size as cost volume
    aggregated_cost_volume = np.zeros((cost_volume.shape, np.inf))

    for dx, dy in directions:
        path_cost = np.copy(cost_volume)

        for y in range(height)[::1 if dy >= 0 else -1]:
            for x in range(width)[::1 if dx >= 0 else -1]:
                if x - dx < 0 or x - dx >= width or y - dy < 0 or y - dy >= height:
                    continue

                for d in range(maxd + 1):
                    prev_costs = path_cost[y - dy, x - dx]

                    # Calculate penalties
                    cost_same = prev_costs[d]
                    cost_d_minus = prev_costs[d - 1] + P1 if d > 0 else np.inf
                    cost_d_plus = prev_costs[d + 1] + P1 if d < maxd else np.inf
                    cost_different = np.min(prev_costs) + P2

                    # Aggregate costs with penalties
                    path_cost[y, x, d] += min(cost_same, cost_d_minus, cost_d_plus, cost_different) - np.min(prev_costs)

        aggregated_cost_volume += path_cost  # Accumulate costs for each direction
    Id = np.argmin(aggregated_cost_volume, axis=2)  # Get disparity map by selecting disparity with lowest aggregated cost

    #------------------

    correct = isinstance(Id, np.ndarray) and Id.shape == Il.shape

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return Id
