import numpy as np
from scipy.ndimage import median_filter

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
        - for our purposes, only need to consider pixels in bbox, so make the cost volume the same size as the bounding box (w/ 3rd dimension as disparity levels) 
    2. Compute matching costs: for each disparity level in the range [0, maxd], compute the matching cost for each pixel in the right image (in bbox), store
       in the cost volume
       - Use absolute differences as the matching cost
    3. Aggregate costs: for each pixel in the cost volume, compute the aggregate cost by summing the matching costs from multiple directions:
        - i.e. loop through a number of directions (up, down, left, right, diagonals) and sum in an initialized array the matching costs in any direction: 
        using the matching costs in step 2.
        - in the each iteration of the loop, loop through each disparity level and aggregate the costs in the specified direction
        - To see where the pixel should move, we consider the cost of staying the same, moving one level left or right, and moving to a different pixel altogether
            - penalize moving within one disparity level with a small penalty, P1,
            - and penalize moving to another pixel alltogether by a larger penalty, P2.  
            - This step enforces a smoothness constraint on the disparity map, and tells us where the pixel should move to minimize the cost, i.e. get best match
        - the aggregate cost can be computed in a number of directions, commonly:
            - 4 (up, down, left, right)
            - 8 (4 + diagonals)
        - regardless, we then have to normalize the aggregated cost by dividing by the number of directions, to get an averaged result.
    4. For each pixel, select disparity with lowest aggregated cost: which represents the best match in the left image
        - we can optionally smooth the output by applying a median filter to the disparity map: median filters filter out POINTWISE noise, which happens
          to a disparity map, for example, in the fast case if small windows are used. This helps get higher accuracy performance while removing noise normally
          experienced in these disparity maps.
    5. Return the disparity map (Id) using these disparities

    The advantage of this "directional" method is that a smoother disparity map can be produced: in stereo_disparity_fast, we only search one direction
    (along epipolar lines), we get more information.
    This method is more computationally expensive than local methods, but less expensive than some of the global methods discussed in the Scharstein paper,
    which is why I picked it to balance runtime performance and accuracy.
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
    # 0. Preliminaries: Define some hypparameters. NOTE: all tunables should be here
    # Extract bounding box coordinates
    x_min, x_max = bbox[0]
    y_min, y_max = bbox[1]
    # Bounding box dimensions, to create cost volume of bbox size
    height_bbox = y_max - y_min + 1
    width_bbox = x_max - x_min + 1
    # Directions: up, down, left, right, diagonals. Start with principal 4, add diagonals if poor performance
    # NOTE: (dx, dy), +1 in dx is right, -1 is left, +1 in dy is down, etc. 
    # 8/4 too slow for autolab: try decreasing directions?
    directions = [(0, 1)] # 2 directions: didn't work: trying one only, which means this is basically a more complex version of stereo_disparity_fast: where
                           # we only search in one direction, but we still aggregate costs from multiple directions which helps us get smoother maps
    # directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # 4 directions, diagonals: (-1, -1), (-1, 1), (1, -1), (1, 1)
    # Penalties: P1 usually ~5-10, P2 usually ~120-150, tunable based on performance
    P1 = 10
    P2 = 130
    # Filter size for median filtering
    filter_size = 6
    
    # Initialize disparity map as before
    Id = np.zeros(Il.shape, dtype=np.float32)

    # 1: Create a "cost volume", ONLY including pixels in bbox (use height_bbox, width_bbox)
    cost_volume = np.full((height_bbox, width_bbox, maxd + 1), np.inf)
    # Fill cost volume: loop over disparity levels
    for d in range(maxd + 1):
        # Shift the right image by d to the left
        Ir_leftshifted = np.zeros(Ir.shape, dtype=np.float32)
        if d > 0:
            Ir_leftshifted[:, d:] = Ir[:, :-d] # This makes [:, :d] INVALID since we dont assign values to it. 
        else:
            # Bugfix: Should need this otherwise we'd just get 0 everywhere
            Ir_leftshifted = Ir
        # 2: fill in matching costs as the absolute difference between left image and leftshifted right image (by d)
        matching_costs = np.abs(Il - Ir_leftshifted)
        # [:, :d] INVALID: so set to infinity, i.e. will never "win" the lowest cost battle
        matching_costs[:, :d] = np.inf
        # Fill in cost volume at this disparity level, indexing at bbox dims since thats all we care about
        cost_volume[:, :, d] = matching_costs[y_min:y_max + 1, x_min:x_max + 1]

    # 3: Aggregate costs within the bounding box for each direction: initialize to 0 this time, since we now sum costs for each direction
    # Initialize aggregated costs, same size as cost volume
    aggregated_costs = np.zeros((height_bbox, width_bbox, maxd + 1))
    # Loop through directions:
    for dx, dy in directions:
        # Need to copy cost volume or it gets modified, and we cant run other directions
        path_cost = np.copy(cost_volume)
        # Loop over the bounding box region in the specified direction: if we have a -1 anywhere, we need to reverse the range list, make custom lists
        # if these conditions happen
        if dy >= 0:
            range_y = range(y_min, y_max + 1)
        else:
            # Go backwards
            range_y = range(y_max, y_min - 1, -1)
        if dx >= 0:
            range_x = range(x_min, x_max + 1)
        else:
            # Go backwards
            range_x = range(x_max, x_min - 1, -1)
        
        # Loop through pixels in the bounding box, this considers 
        for y in range_y:
            for x in range_x:
                # Boundary conditions if we are outside of the bbox: we don't care about anything outside the bbox becuase we only have costs inside
                # Can maybe just change the ranges to avoid this if starved for runtime
                if x - dx < x_min or x - dx >= x_max or y - dy < y_min or y - dy >= y_max:
                    continue
                # Loop through disparity levels
                for d in range(maxd + 1):
                    # Get the surrounding costs around the pixel: 
                    # But, in cost, index 0 corresponds to topleft of bbox, so we have to account for the difference 
                    y_idx = y - y_min - dy
                    x_idx = x - x_min - dx
                    surrounding_costs = path_cost[y_idx, x_idx]

                    # Add penalties to surrounding costs: want to see if we can get a better cost by moving in a different direction
                    # Stay the same: no penalty
                    cost_same = surrounding_costs[d]
                    # Move one level left or right: add small penalty
                    if d > 0:
                        cost_d_left = surrounding_costs[d - 1] + P1
                    else:
                        # Invalid disparity level, set cost to inf so it never wins
                        cost_d_left = np.inf
                    if d < maxd:
                        cost_d_right = surrounding_costs[d + 1] + P1
                    else:
                        # Invalid disparity level, set cost to inf so it never wins
                        cost_d_right = np.inf
                    # Move to a different pixel altogether: add larger penalty
                    cost_different = np.min(surrounding_costs) + P2
                    # Aggregate costs with penalties: take the minimum of the penalties and subtract the minimum of the surrounding costs
                    path_cost[y - y_min, x - x_min, d] += min(cost_same, cost_d_left, cost_d_right, cost_different) - np.min(surrounding_costs)
        # Add minimum cost best paths to aggregated costs, for each direction
        aggregated_costs += path_cost

    # 4: Normalize the aggregated cost and select disparity with minimum cost
    aggregated_costs /= len(directions)
    # Select along axis=2, which is the disparity level for each pixel in bbox, overwrite pixels in bbox only
    Id[y_min:y_max + 1, x_min:x_max + 1] = np.argmin(aggregated_costs, axis=2)

    # Enhance results: apply median filtering over bbox to smooth the output in the bounding box region
    Id[y_min:y_max + 1, x_min:x_max + 1] = median_filter(Id[y_min:y_max + 1, x_min:x_max + 1], size=filter_size)

    # 5: Return the disparity map
    # Done below
    #------------------

    correct = isinstance(Id, np.ndarray) and Id.shape == Il.shape

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return Id
