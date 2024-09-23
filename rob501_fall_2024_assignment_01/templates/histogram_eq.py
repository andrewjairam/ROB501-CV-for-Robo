import numpy as np

def histogram_eq(I):
    """
    Histogram equalization for greyscale image.

    Perform histogram equalization on the 8-bit greyscale intensity image I
    to produce a contrast-enhanced image J. Full details of the algorithm are
    provided in the Szeliski text.

    Parameters:
    -----------
    I  - Single-band (greyscale) intensity image, 8-bit np.array (i.e., uint8).

    Returns:
    --------
    J  - Contrast-enhanced greyscale intensity image, 8-bit np.array (i.e., uint8).
    """
    #--- FILL ME IN ---

    # Verify I is grayscale.
    if I.dtype != np.uint8:
        raise ValueError('Incorrect image format!')
    # Goal: find J = the cumulative distribution function of the histogram of I, mapped to every pixel in I.
    # 1. Compute histogram of I with 256 bins (since grayscale pixel values can be 0-255)
    I_flattened = I.flatten()
    I_hist, bins = np.histogram(I_flattened, bins=256, range=[0, 256]) # Flatten I to 1D array so that each pixel gets included
    # 2. compute the CDF using the histogram
    I_cdf = I_hist.cumsum()
    # 3. Normalize the CDF 
    I_cdf_norm = I_cdf / len(I_flattened)
    # 4. Multiply Normalized CDF by 255 to get the new pixels
    I_cdf_norm = I_cdf_norm * 255
    # 5. Map new pixels to J: can use np.interp to map each pixel in I_flattened to the new pixel value using I_cdf_norm
    J_flattened = np.interp(I_flattened, bins[:-1], I_cdf_norm) # 1 less bin: # bins = # edges + 1
    # 6. Reshape J to the original shape of I
    J = J_flattened.reshape(I.shape)
    #------------------
    return J
