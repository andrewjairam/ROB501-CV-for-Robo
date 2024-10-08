import numpy as np
from numpy.linalg import inv, lstsq
from scipy.linalg import null_space
from scipy.ndimage.filters import *
from matplotlib.path import Path

# You may add support functions here, if desired.
def harris(I, alpha=0.04, window=1, threshold=1e-2):
    '''
    Detect corners in an image using the Harris corner detector, for a grayscale image according to:
    det(A) - alpha*trace(A)^2
    Parameters:
    I - Single-band (greyscale) image as np.array (e.g., uint8, float): Same as in cross_junctions
    alpha - float: Harris detectpr parameter, usually chosen around 0.04-0.06
    window - int: Size of gaussian standard distribution in gaussian smoothing: larger window means more smoothing, meaning more corners 
                   are detected, but smaller details can be missed
    threshold - float: Reject corners with Harris response less than this value
    '''
    # 1. Compute image gradients, using scipy.ndimage.filters.sobel, and define the four elements of matrix A
    Ix = sobel(I, axis=1)
    Iy = sobel(I, axis=0) # NOTE: as usual, axis 0 is the y axis and axis 1 is the x axis for matplotlib images
    Ix_sq = Ix**2
    Iy_sq = Iy**2
    IxIy = Ix*Iy
    # 2. Apply gaussian smoothing according to chosen window
    Ix_sq = gaussian_filter(Ix_sq, window)
    Iy_sq = gaussian_filter(Iy_sq, window)
    IxIy = gaussian_filter(IxIy, window)
    # 3. Find Harris detector values for each pixel, compiled in matrix H
    det_A = Ix_sq*Iy_sq - IxIy**2
    trace_A = Ix_sq + Iy_sq
    H = det_A - alpha*(trace_A**2)
    # 4a. normalize H by the max value of H
    H = H/H.max()
    # 4b. Threshold H so that corners are 1 and other pixels are 0, according to chosen threshold
    corners = (H > threshold).astype(int)
    # 5. Return matrix where all detected corners are 1 and all other pixels are 0, corners is same shape as I
    return corners, H
def filter_bbox(corners, H, bbox):
    '''
    Filter out corners that are not in the bounding box
    Parameters:
    corners - np.array: Matrix where all detected corners are 1 and all other pixels are 0
    H - np.array: Harris detector values for each pixel
    bbox - np.array: 2x4 np.array, bounding polygon (clockwise from upper left)
    '''
    # 1. Create Path object from bbox
    bbox_path = Path(bbox.T)
    # 2. Iterate over all corners, and set to 0 any corner that is not in the bbox
    for y in range(corners.shape[0]):
        for x in range(corners.shape[1]):
            if not bbox_path.contains_point([x, y]):
                corners[y, x] = 0
                H[y, x] = 0
    return corners, H
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

def refine_saddle_point(I, pt, patch_size=10):
    '''
    Parameters:
    ------------
    I = Single-band (greyscale) image as np.array (e.g., uint8, float)
    pt = 2x1 np.array (float64), subpixel location of saddle point in I (x,y), found by harris corner detection
    patch_size = int, size of patch around saddle point to look in order to refine saddle point
    '''
    pass
def cross_junctions(I, bpoly, Wpts):
    """
    Find cross-junctions in image with subpixel accuracy.

    The function locates a series of cross-junction points on a planar 
    calibration target, where the target is bounded in the image by the 
    specified quadrilateral. The number of cross-junctions identified 
    should be equal to the number of world points.

    Note also that the world and image points must be in *correspondence*,
    that is, the first world point should map to the first image point, etc.

    Parameters:
    -----------
    I      - Single-band (greyscale) image as np.array (e.g., uint8, float).
    bpoly  - 2x4 np.array, bounding polygon (clockwise from upper left).
    Wpts   - 3xn np.array of world points (in 3D, on calibration target).

    Returns:
    --------
    Ipts  - 2xn np.array of cross-junctions (x, y), relative to the upper
            left corner of the target. The array must contain float64 values.
    """
    #--- FILL ME IN ---
    # Sample steps
    # 1. Extract region of interest
    # 2. detect initial cross junction: importing sciy.ndimage.filters indicates we should implement harris corner detection
    # 3. refine cross junction using saddle point algorithm (from part 1)
    # 4. Map cross junction to world points (using homography)
    # 5. Return cross junctions in world coordinates
    # Code goes here...

    # Find harris corner responses

    Ipts = None
    # Find harris corner responses of I
    # print(Wpts)
    # quit()
    corners, H = harris(I,0.04, 10, 0.98)
    #print(H.max())
    # Discard corner responses outside of bbox
    corners, H = filter_bbox(corners, H, bpoly)
    # For each response, refine the saddle point
    # FROM COPILOT: refines saddle point for each corner
    # patch_size = 10
    # corners_not_updated = 0
    # for y in range(corners.shape[0]):
    #     for x in range(corners.shape[1]):
    #         if corners[y, x] == 1:
    #             # Extract patch around corner
    #             patch = I[y-patch_size:y+patch_size, x-patch_size:x+patch_size]
    #             # Refine saddle point
    #             pt = saddle_point(patch)
    #             # Update corners
    #             try:
    #                 corners[y, x] = 0
    #                 corners[y + int(pt[1]), x + int(pt[0])] = 1
    #             except:
    #                 print("corner not updated")
    #                 corners_not_updated += 1

    # ANDREW: Try to see impact of patch first
    patch_size = 10
    for y in range(corners.shape[0]):
        for x in range(corners.shape[1]):
            if corners[y, x] == 1:
                patch_coords = np.array([x, y])
                # patch = I[y-patch_size:y+patch_size, x-patch_size:x+patch_size]
                # patch_coords = np.array([[y-patch_size, y-patch_size, y+patch_size, y+patch_size], \
                #                         [x-patch_size, x+patch_size, x-patch_size, x+patch_size]
                #                         ])
    return corners, patch_coords
    print(Wpts.shape)
    quit()

    #------------------

    correct = isinstance(Ipts, np.ndarray) and \
        Ipts.dtype == np.float64 and Ipts.shape[0] == 2

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return Ipts