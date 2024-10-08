import numpy as np
from numpy.linalg import inv, lstsq
from scipy.linalg import null_space
from scipy.ndimage.filters import *
from matplotlib.path import Path

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

def saddle_point(I):
    """
    Locate saddle point in an image patch.

    The function identifies the subpixel centre of a cross-junction in the
    image patch I, by fitting a hyperbolic paraboloid to the patch, and then 
    finding the critical point of that paraboloid.

    Note that the location of 'pt' is relative to (-0.5, -0.5) at the upper
    left corner of the patch, i.e., the pixels are treated as covering an 
    area of one unit square.

    Parameters:
    -----------
    I  - Single-band (greyscale) image patch as np.array (e.g., uint8, float).

    Returns:
    --------
    pt  - 2x1 np.array (float64), subpixel location of saddle point in I (x, y).
    """
    #--- FILL ME IN ---
    pt = None
    # GOAL: Solve linear least squares given equation 4 in Lucchese
    # Equivalently: solve p = (A^T A)^-1 A^T psi_s where the psi_s quantity are elements of I flattened, A is the matrix where each row is [x^2, y^2, xy, x, y, 1]
    # p = [a, b, c, d, e, f]^T are the parameters alpha, beta, gamma, ... in the paper. Then use parameter vector p to get coords of saddle pt.
    # NOTE: have to index into I with [y, x] because I is a numpy array and the first index is the row index (y) and the second index is the column index (x)

    # 1. Create the matrix A and intensity vector psi_s
    A = []
    psi_s = [] # Assumes I is smoothed

    for y in range(I.shape[0]):
        for x in range(I.shape[1]):
            A.append([x**2, x*y, y**2, x, y, 1])
            psi_s.append(I[y, x]) # Couldve also done psi_s = I.flatten()?

    A = np.array(A)
    psi_s = np.array(psi_s).reshape(-1, 1)
    # 2. Compute least squares solution for parameters p
    p, _, _, _ = lstsq(A, psi_s, rcond=None)
    a, b, c, d, e, f = p.flatten()

    # 3. Compute saddle point coordinates using p using equation in paper
    M = np.array([[2*a, b], [b, 2*c]])
    v = np.array([d, e]).reshape(-1, 1)
    pt = -1*inv(M).dot(v)
    
    #------------------

    correct = isinstance(pt, np.ndarray) and \
        pt.dtype == np.float64 and pt.shape == (2, 1)

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return pt

def cross_junctions(I, bpoly, Wpts):
    #--- FILL ME IN ---

    # We have location of junctions in world frame if the checkerboard was perfectly flat -> want junctions in image frame
    # The only points we have in image frame are the corners of the bounding box -> if we can estimate bbox in world frame, can use homography to transform junction coords

    # 1. Estimate bounding box in world frame: the "corner junctions" are one square plus some small border in the world frame:
    # also we know that the world frame side length of a square is 63.5mm
    side_len = 0.0635 # = abs(Wpts[0,0] - Wpts[0,1]) if autolab tests on some other img
    # Find min in x and y, max in x and y, and subtract side_len from min x and y and add side_len to max x and y. min/max should be first/last elems of Wpts
    Wxmin, Wxmax = abs(np.min(Wpts[0])) - side_len, abs(np.max(Wpts[0])) + side_len
    Wymin, Wymax = abs(np.min(Wpts[1])) - side_len, abs(np.max(Wpts[1])) + side_len
    # Need to take into account that small border around the checkerboard -> manual tuning. Vertical borders here are x, looks thicker than horizontal borders
    # borders just have to be much less than 0.0625 (say less than a third of 0.0625), horiz border less than vertical. If autolab fails, remove hardcoding by replacing w/ ratio
    vertical_border = 0.02 # Cur vert ratio: 0.32 (0.02/0.0625), horiz ratio: 0.16 (0.01/0.0625)
    horizontal_border = 0.01 # horiz ratio: 0.16 (0.01/0.0625). We would set horizontal_border = 0.16 * side_len
    Wxmin, Wxmax = Wxmin - vertical_border, Wxmax + vertical_border
    Wymin, Wymax = Wymin - horizontal_border, Wymax + horizontal_border
    # Now can make this into a 2x4, looking at bpoly, order is upper left, upper right, lower right, lower left
    Wbbox = np.array([[Wxmin, Wxmax, Wxmax, Wxmin], [Wymin, Wymin, Wymax, Wymax]])
    # 2. Compute homography from world to image frame: want to map Wbbox to bpoly, so Wbbox is I1pts and bpoly is I2pts
    H, _ = dlt_homography(Wbbox, bpoly) # H brings us from Wpts to points on img
    # 3. Map Wpts to image frame using H: NOTE that Wpts in z are all 0, need pts as x, y, 1
    Wpts = np.vstack((Wpts[:2], np.ones(Wpts.shape[1]))) # Wpts.shape = 3 x n (n = 48 for 48 junctions), H*Wpts gives us 3 x n then also
    Ipts = np.dot(H, Wpts).T # Ipts.shape = n x 3: transposing gets each elem as x,y,z coordinates
    # Normalize Ipts to get x, y coordinates
    Ipts = Ipts / Ipts[:, 2].reshape(-1, 1) # Have to do this reshape to do the division element-wise
    # Round x and y to get pixel coordinates since in refining, need rounded x and y's. take Ipts[:, :2] to discard z coords
    Ipts = np.round(Ipts[:, :2]).astype(np.float64) # Need float64 since this is desired output type given
    # INTERMEDIATE STEP: return Ipts, plot them on img, see if junctions are generally in right place
    # return Ipts # On First image, already looks good: Can pass to saddle to be sure, with small window size
    # 4. Refine junctions using saddle point alg, need to define patch size (as discussed, should be small)
    patch_size = 10
    for i in range(Ipts.shape[0]):
        # Extract patch around corner: convert to int since we need to index into I, already rounded though
        x, y = Ipts[i].astype(int)
        # convert 
        # # DEBUG STEP: visualize patch
        # patchbbox = np.array([[x-patch_size, x+patch_size, x+patch_size, x-patch_size], [y-patch_size, y-patch_size, y+patch_size, y+patch_size]])
        # return Ipts, patchbbox # Already pretty accurate on 1st image, can tune if autolab fails
        patch = I[y-patch_size:y+patch_size, x-patch_size:x+patch_size]
        # Refine saddle point
        pt = saddle_point(patch)
        # But pt is relative to top left corner of patch, our x/y coords are in the middle of patch
        pt_x = pt[0] - patch_size
        pt_y = pt[1] - patch_size
        Ipts[i, 0] = x + pt_x
        Ipts[i, 1] = y + pt_y
    # 5. Return Ipts, output wants 2xn np array
    Ipts = Ipts.T # Transpose to get 2 x n
    #------------------

    correct = isinstance(Ipts, np.ndarray) and \
        Ipts.dtype == np.float64 and Ipts.shape[0] == 2

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return Ipts