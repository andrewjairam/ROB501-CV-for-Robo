import numpy as np
from numpy.linalg import inv
from ibvs_jacobian import ibvs_jacobian

def ibvs_depth_finder(K, pts_obs, pts_prev, v_cam):
    """
    Compute estimated 

    The function computes the Jacobian for image-based visual servoing,
    given the camera matrix K, an image plane point, and the estimated
    depth of the point. The x and y focal lengths in K are guaranteed 
    to be identical.

    Parameters:
    -----------
    K        - 3x3 np.array, camera intrinsic calibration matrix.
    pts_obs  - 2xn np.array, observed (current) image plane points.
    pts_prev - 2xn np.array, observed (previous) image plane points.
    v_cam    - 6x1 np.array, camera velocity (last commmanded).

    Returns:
    --------
    zs_est - nx0 np.array, updated, estimated depth values for each point.
    """
    n = pts_obs.shape[1]
    J = np.zeros((2*n, 6))
    zs_est = np.zeros(n)

    #--- FILL ME IN ---
    for i in range(n):
        # 1. get Jt and Jw: call ibvs_jacobian w/ z = 1, split 2x6 into 2 2x3's
        pt_obs = pts_obs[:, i].reshape(2, 1)
        pt_prev = pts_prev[:, i].reshape(2, 1)
        J_i = ibvs_jacobian(K, pt_obs, 1)
        J_t = J_i[:, :3]
        J_w = J_i[:, 3:]
        # 2. Split v_cam into v and w
        v = v_cam[:3].reshape(3, 1)
        w = v_cam[3:].reshape(3, 1)
        # 3. Get u_dot, v_dot: how? no clear way... try to use approximation: p_obs - p_prev? would assume that timesteps are ~ 1
        u_dot = pt_obs[0][0] - pt_prev[0][0]
        v_dot = pt_obs[1][0] - pt_prev[1][0]
        vel = np.array([[u_dot], [v_dot]]).reshape(2, 1)
        # Form Ax = b, x = 1/z
        A = J_t @ v # 2x6 * 6x1 = 2x1
        b = vel - J_w @ w # 2x1 - 2x3 * 3x1 = 2x1
        # Solve for x using Linear LSQ
        x = inv(A.T @ A) @ A.T @ b
        # 4. add to zs_est, z = 1/x
        zs_est[i] = 1 / x[0][0]

    #------------------

    correct = isinstance(zs_est, np.ndarray) and \
        zs_est.dtype == np.float64 and zs_est.shape == (n,)

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return zs_est