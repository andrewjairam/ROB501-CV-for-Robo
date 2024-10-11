import numpy as np
from numpy.linalg import inv, norm
from find_jacobian import find_jacobian
from dcm_from_rpy import dcm_from_rpy
from rpy_from_dcm import rpy_from_dcm


#----- Functions Go Below -----

def epose_from_hpose(T):
    """Covert 4x4 homogeneous pose matrix to x, y, z, roll, pitch, yaw."""
    E = np.zeros((6, 1))
    E[0:3] = np.reshape(T[0:3, 3], (3, 1))
    E[3:6] = rpy_from_dcm(T[0:3, 0:3])
  
    return E

def hpose_from_epose(E):
    """Covert x, y, z, roll, pitch, yaw to 4x4 homogeneous pose matrix."""
    T = np.zeros((4, 4))
    T[0:3, 0:3] = dcm_from_rpy(E[3:6])
    T[0:3, 3] = np.reshape(E[0:3], (3,))
    T[3, 3] = 1
  
    return T

def pose_estimate_nls(K, Twc_guess, Ipts, Wpts):
    """
    Estimate camera pose from 2D-3D correspondences via NLS.

    The function performs a nonlinear least squares optimization procedure 
    to determine the best estimate of the camera pose in the calibration
    target frame, given 2D-3D point correspondences.

    Parameters:
    -----------
    K          - 3x3 camera intrinsic calibration matrix.
    Twc_guess  - 4x4 homogenous pose matrix, initial guess for camera pose.
    Ipts       - 2xn array of cross-junction points (with subpixel accuracy).
    Wpts       - 3xn array of world points (one-to-one correspondence with Ipts).

    Returns:
    --------
    Twc  - 4x4 np.array (float64), pose matrix, camera pose in target frame.
    """
    maxIters = 250                          # Set maximum iterations.

    tp = Ipts.shape[1]                      # Num points.
    ps = np.reshape(Ipts, (2*tp, 1), 'F')   # Stacked vector of observations.

    dY = np.zeros((2*tp, 1))                # Residuals.
    J  = np.zeros((2*tp, 6))                # Jacobian.

    #--- FILL ME IN ---

    # Some hints on structure are included below...

    # 1. Convert initial guess to parameter vector (6 x 1).
    # 1a. want 6x1 parameter vector [x, y, z, roll, pitch, yaw] from Twc_guess: can use epose_from_hpose
    params = epose_from_hpose(Twc_guess)
    iter = 1 # Track num iterations
    # 1.b: initialize params_prev: Track best estimate in params_prev
    params_prev = np.zeros((6, 1))

    # 2. Main loop - continue until convergence or maxIters.
    while True:
        # 3. Save previous best pose estimate.
        params_prev = params.copy()

        # 4. Project each landmark into image, given current pose estimate.
        for i in np.arange(tp):
            # 4.a: Get rotation and translation matrix/vector from current params.
            Rwc = dcm_from_rpy(params[3:6]) # params[3:6] are roll, pitch, yaw
            twc = params[:3].reshape(-1, 1) # just take x, y, z
            # 4b. Compute pose guess: 
            # Twc in world frame, so x_guess = K * Rwc.T * (Wpt - twc), Wpt is Wpts[:, i]: the ith world point
            Wpt = Wpts[:, i].reshape(-1, 1)
            x_guess = K @ Rwc.T @ (Wpt - twc)
            # 4c. Normalize x_guess to get projected point and discard z component to get 2x1 pixel coords
            x_guess = x_guess / x_guess[2, 0]
            x_guess = x_guess[:2, 0].reshape(-1, 1) 
            # 4d. Compute e(x): the difference between the projected point and the observed point.
            # It's defined that we wanna use dY as the variable from above, set the two error terms as dY[i] and dY[i+1]
            x_obs = ps[2*i:2*i+2, 0].reshape(-1, 1)
            dY[2*i:2*i+2] = x_guess - x_obs
            # 4e. Compute Jacobian for this 3x1 Wpt
            J[2*i:2*i+2, :] = find_jacobian(K, hpose_from_epose(params), Wpt).reshape(2, 6)


        # 5. Solve system of normal equations for this iteration.
        # 5a. compute delta_x = -inv(J^T J) J^T dY
        delta_x = -inv(J.T @ J) @ J.T @ dY
        # 5b. Update params = params + delta_x
        params = params + delta_x

        # 6. Check - converged?
        diff = norm(params - params_prev)

        if norm(diff) < 1e-12:
            print("Covergence required %d iters." % iter)
            break
        elif iter == maxIters:
            print("Failed to converge after %d iters." % iter)
            break
        
        iter += 1

    # 7. Compute and return homogeneous pose matrix Twc.
    # Take final params and call hpose_from_epose
    Twc = hpose_from_epose(params)
    #------------------

    correct = isinstance(Twc, np.ndarray) and \
        Twc.dtype == np.float64 and \
        Twc.shape == (4, 4) and Twc[3, 3] == 1.0000

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return Twc