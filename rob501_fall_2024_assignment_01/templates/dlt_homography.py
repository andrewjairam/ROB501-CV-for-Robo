import numpy as np
from numpy.linalg import inv, norm
from scipy.linalg import null_space

def dlt_homography(I1pts, I2pts):
    """
    Find perspective Homography between two images.
    
    Given 4 points from 2 separate images, compute the perspective homography
    (warp) between these points using the DLT algorithm.

    Parameters:
    ----------- 
    I1pts  - 2x4 np.array of points from Image 1 (each column is x, y).
    I2pts  - 2x4 np.array of points from Image 2 (in 1-to-1 correspondence).

    Returns:
    --------
    H  - 3x3 np.array of perspective homography (matrix map) between image coordinates.
    A  - 8x9 np.array of DLT matrix used to determine homography.
    """
    #--- FILL ME IN ---
    # Goal: Find H satisfying I2 = H * I1, by solving Ah = 0, h = [h11, h12, h13, h21, h22, h23, h31, h32, h33]^T.
    # 1. Construct A matrix: using I2 = H * I1, we add 2 arrays to A for each point correspondence (4 points in total, iterate 4 times):
    #    [x1, y1, 1, 0, 0, 0, -x2*x1, -x2*y1, -x2], [0, 0, 0, -x1, -y1, -1, y2*x1, y2*y1, y2], for x1 = I1[0, i], y1 = I1[1, i], x2 = I2[0, i], y2 = I2[1, i], i = 0, 1, 2, 3.
    A = []
    for i in range(4):
        x1, y1 = I1pts[0, i], I1pts[1, i]
        x2, y2 = I2pts[0, i], I2pts[1, i]
        A.append([x1, y1, 1, 0, 0, 0, -x2*x1, -x2*y1, -x2])
        A.append([0, 0, 0, -x1, -y1, -1, y2*x1, y2*y1, y2])
    A = np.array(A)
    # 2. Solve Ah = 0 using scipy nullspace
    h = null_space(A)[:, 0] # take first col vector if there are multiple solutions
    # 3. Reshape h into 3x3 matrix H
    H = np.array(h).reshape(3, 3)
    # 4. Normalize H by h_33 = h[-1] or H[-1, -1]
    H = H / h[-1] 
    #------------------

    return H, A

# if __name__ == "__main__":
#     dlt_homography(np.array([[5, 220, 220, 5], [1, 1, 411, 411]]), np.array([[375, 420, 420, 450], [20, 20, 300, 290]]))