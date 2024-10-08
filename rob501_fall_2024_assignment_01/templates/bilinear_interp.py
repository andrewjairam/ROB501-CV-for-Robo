import numpy as np
from numpy.linalg import inv

def bilinear_interp(I, pt):
    """
    Performs bilinear interpolation for a given image point.

    Given the (x, y) location of a point in an input image, use the surrounding
    four pixels to conmpute the bilinearly-interpolated output pixel intensity.

    Note that images are (usually) integer-valued functions (in 2D), therefore
    the intensity value you return must be an integer (use round()).

    This function is for a *single* image band only - for RGB images, you will 
    need to call the function once for each colour channel.

    Parameters:
    -----------
    I   - Single-band (greyscale) intensity image, 8-bit np.array (i.e., uint8).
    pt  - 2x1 np.array of point in input image (x, y), with subpixel precision.

    Returns:
    --------
    b  - Interpolated brightness or intensity value (whole number >= 0).
    """
    #--- FILL ME IN ---
    # NOTE: imread gives coordinates (x,y) as (row, column) = (y, x), so we need to reverse the order of the coordinates (I[y, x])
    if pt.shape != (2, 1):
        raise ValueError('Point size is incorrect.')
    # 1. Use np.floor and np.ceil to get the nearest surrounding pixels
    x, y = pt[0, 0], pt[1, 0]
    x1, y1 = int(np.floor(x)), int(np.floor(y))
    x2, y2 = int(np.ceil(x)), int(np.ceil(y))
    # 2. Edge cases: x = x1 or x2, y = y1 or y2
    # 2a: if x == x1 or x2 and y == y1 or y2, return the pixel value at (x, y) (since we are at a corner)
    if x == x1 and y == y1: # top left
        return int(I[y1, x1])
    elif x == x2 and y == y1: # top right
        return int(I[y2, x1])
    elif x == x1 and y == y2: # bottom left
        return int(I[y1, x2])
    elif x == x2 and y == y2: # bottom right
        return int(I[y2, x2])
    # 2b: Vertical edges: if x == x1 or x2, interpolate between y1 and y2
    if x == x1 or x == x2:
        x = int(x)
        b = ((y2 - y) * I[y1, x] + (y - y1) * I[y2, x]) / (y2 - y1)
        return int(round(b))
    # 2c: Horizontal edges: if y == y1 or y2, interpolate between x1 and x2
    if y == y1 or y == y2:
        y = int(y)
        b = ((x2 - x) * I[y, x1] + (x - x1) * I[y, x2]) / (x2 - x1)
        return int(round(b))
    # 3. Interpolate x with y = y1
    I_x_y1 = ((x2 - x) * I[y1, x1] + (x - x1) * I[y1, x2]) / (x2 - x1)
    # 4. Interpolate x with y = y2
    I_x_y2 = ((x2 - x) * I[y2, x1] + (x - x1) * I[y2, x2]) / (x2 - x1)
    # 5. Interpolate between the two results using y, round the result since we want a pixel value
    b = ((y2 - y) * I_x_y1 + (y - y1) * I_x_y2) / (y2 - y1)

    #------------------

    return int(round(b))