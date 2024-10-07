import numpy as np
from numpy.linalg import inv, lstsq

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