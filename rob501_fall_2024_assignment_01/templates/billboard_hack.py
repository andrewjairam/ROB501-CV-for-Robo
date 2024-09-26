# Billboard hack script file.
import numpy as np
from matplotlib.path import Path
from imageio import imread, imwrite

from dlt_homography import dlt_homography
from bilinear_interp import bilinear_interp
from histogram_eq import histogram_eq

from matplotlib import pyplot as plt
def billboard_hack():
    """
    Hack and replace the billboard!

    Parameters:
    ----------- 

    Returns:
    --------
    Ihack  - Hacked RGB intensity image, 8-bit np.array (i.e., uint8).
    """
    # Bounding box in Y & D Square image - use if you find useful. (NOTE: THIS IS with flipped coordinates (y,x) instead of (x,y))
    bbox = np.array([[404, 490, 404, 490], [38,  38, 354, 354]])

    # Point correspondences.
    Iyd_pts = np.array([[416, 485, 488, 410], [40,  61, 353, 349]])
    Ist_pts = np.array([[2, 218, 218, 2], [2, 2, 409, 409]])

    Iyd = imread('../images/yonge_dundas_square.jpg')
    Ist = imread('../images/uoft_soldiers_tower_light.png')

    Ihack = np.asarray(Iyd)
    Ist = np.asarray(Ist)

    #--- FILL ME IN ---
    # Let's do the histogram equalization first.
    Ist_eq = histogram_eq(Ist)
    # Compute the perspective homography we need: 
    # Since we want to warp pixels in Iyd (bounded in the parallelogram path defined by Iyd_pts), use Iyd as source
    H, _ = dlt_homography(Iyd_pts, Ist_pts) 

    # Main 'for' loop to do the warp and insertion - 
    # this could be vectorized to be faster if needed!
    
    # You may wish to make use of the contains_points() method
    # available in the matplotlib.path.Path class!

    # 1. Define path bb_path that contains the billboard in Iyd: Iyd_pts are (roughly) the 4 corners of the billboard
    bb_path = Path(Iyd_pts.T) # Have to transpose to make a N x 2

    # 2. Loop: Bounding box defines the generic (rectangular) area where the billboard is. 
    # The actual billboard has a skew: not fully rectangular, have to use a parallelogram path.
    # Iterate through the bounding box coordinates, and whenever the pixel is in the billboard parallelogram path, perform homography

    for i in range(404, 490+1): # Hard coded x min/max based on bbox
        for j in range(38, 354+1): # Hard coded y min/max based on bbox
            if bb_path.contains_point([i, j]):
                # Define point in homogenous coordinates
                dst_pxl = np.array([i, j, 1])
                # Compute homography with dst_pxl to get src_pxl: corresponding pixel coordinates in Ist
                src_pxl = np.dot(H, dst_pxl)
                # Include normalization to convert back to cartesian coords
                src_pxl = src_pxl[:-1] / src_pxl[-1]
                # Reshape to a np 2x1 so format works for bilinear interpolation
                src_pxl = src_pxl.reshape(2,1)
                # Bilinear interpolate to get intensity value in Ist_eq
                # Remember that have to use flipped coordinates (y,x) instead of (x,y)) due to imread coordinates
                Ihack[j, i, :] = bilinear_interp(Ist_eq, src_pxl)
    # Visualize the result, if desired...
    # plt.imshow(Iyd)
    # plt.show()
    # 3. Write img: EXCLUDE THIS: SEEMS LIKE SUBMISSION DOESNT WANT IT
    # imwrite(Ihack, 'billboard_hacked.png')
    #------------------
    return Ihack