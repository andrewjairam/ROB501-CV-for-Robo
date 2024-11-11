import numpy as np

def ibvs_jacobian(K, pt, z):
    """
    Determine the Jacobian for IBVS.

    The function computes the Jacobian for image-based visual servoing,
    given the camera matrix K, an image plane point, and the estimated
    depth of the point. The x and y focal lengths in K are guaranteed 
    to be identical.

    Parameters:
    -----------
    K  - 3x3 np.array, camera intrinsic calibration matrix.
    pt - 2x1 np.array, image plane point. 
    z  - Scalar depth value (estimated).

    Returns:
    --------
    J  - 2x6 np.array, Jacobian. The matrix must contain float64 values.
    """

    #--- FILL ME IN ---
    # Copy from Corke text
    J = None
    # 1. Get normalized image coordinates
    x = pt[0][0] / z
    y = pt[1][0] / z
    # 2. get u_bar, v_bar. Assume f = fx = fy
    f = K[0][0] # fx = f / rho_u, fy = f / rho_v
    # u_bar = f/rho_u * x = fx * x, v_bar = f/rho_v * y = fy * y
    u_bar = f * x
    v_bar = f * y
    # 3. Compute the Jacobian using 15.9, f' = f/rho = fx = fy
    J = [[-f/z, 0, u_bar/z, u_bar*v_bar/f, -(f**2 + u_bar**2)/f, v_bar], \
         [0, -f/z, v_bar/z, (f**2 + v_bar**2)/f, -u_bar*v_bar/f, -u_bar]]
    J = np.array(J, dtype=np.float64)

    #------------------

    correct = isinstance(J, np.ndarray) and \
        J.dtype == np.float64 and J.shape == (2, 6)

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return J