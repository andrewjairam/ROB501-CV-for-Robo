import numpy as np
from numpy.linalg import inv

def rpy_from_dcm(R):
    """
    Roll, pitch, yaw Euler angles from rotation matrix.

    The function computes roll, pitch and yaw angles from the
    rotation matrix R. The pitch angle p is constrained to the range
    (-pi/2, pi/2].  The returned angles are in radians.

    Inputs:
    -------
    R  - 3x3 orthonormal rotation matrix.

    Returns:
    --------
    rpy  - 3x1 np.array of roll, pitch, yaw Euler angles.
    """
    rpy = np.zeros((3, 1))

    # Roll.
    rpy[0] = np.arctan2(R[2, 1], R[2, 2])

    # Pitch.
    sp = -R[2, 0]
    cp = np.sqrt(R[0, 0]*R[0, 0] + R[1, 0]*R[1, 0])

    if np.abs(cp) > 1e-15:
      rpy[1] = np.arctan2(sp, cp)
    else:
      # Gimbal lock...
      rpy[1] = np.pi/2
  
      if sp < 0:
        rpy[1] = -rpy[1]

    # Yaw.
    rpy[2] = np.arctan2(R[1, 0], R[0, 0])

    return rpy

import numpy as np

def dcm_from_rpy(rpy):
    """
    Rotation matrix from roll, pitch, yaw Euler angles.

    The function produces a 3x3 orthonormal rotation matrix R
    from the vector rpy containing roll angle r, pitch angle p, and yaw angle
    y.  All angles are specified in radians.  We use the aerospace convention
    here (see descriptions below).  Note that roll, pitch and yaw angles are
    also often denoted by phi, theta, and psi (respectively).

    The angles are applied in the following order:

     1.  Yaw   -> by angle 'y' in the local (body-attached) frame.
     2.  Pitch -> by angle 'p' in the local frame.
     3.  Roll  -> by angle 'r' in the local frame.  

    Note that this is exactly equivalent to the following fixed-axis
    sequence:

     1.  Roll  -> by angle 'r' in the fixed frame.
     2.  Pitch -> by angle 'p' in the fixed frame.
     3.  Yaw   -> by angle 'y' in the fixed frame.

    Parameters:
    -----------
    rpy  - 3x1 np.array of roll, pitch, yaw Euler angles.

    Returns:
    --------
    R  - 3x3 np.array, orthonormal rotation matrix.
    """
    cr = np.cos(rpy[0]).item()
    sr = np.sin(rpy[0]).item()
    cp = np.cos(rpy[1]).item()
    sp = np.sin(rpy[1]).item()
    cy = np.cos(rpy[2]).item()
    sy = np.sin(rpy[2]).item()

    return np.array([[cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
                     [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
                     [  -sp,            cp*sr,            cp*cr]])

def find_jacobian(K, Twc, Wpt):
    """
    Determine the Jacobian for NLS camera pose optimization.

    The function computes the Jacobian of an image plane point with respect
    to the current camera pose estimate, given a landmark point. The 
    projection model is the simple pinhole model.

    Note that the homogeneous transformation matrix provided defines the
    transformation from the *camera frame* to the *world frame* (to 
    project into the image, you would need to invert this matrix).

    Parameters:
    -----------
    K    - 3x3 np.array, camera intrinsic calibration matrix.
    Twc  - 4x4 np.array, homogenous pose matrix, current guess for camera pose. 
    Wpt  - 3x1 world point on calibration target (one of n).

    Returns:
    --------
    J  - 2x6 np.array, Jacobian matrix (columns are tx, ty, tz, r, p, q).
         The Jacobian must contain float64 values.
    """
    #--- FILL ME IN ---
 
    # The simple pinhole model is given by the equation:
    # x = K * [R | t] * Wpt where x is the image plane point, K is the camera intrinsic matrix, R is the rotation matrix, t is the translation vector, and Wpt is the world point.
    # Want wrt 6 variables: translation variables tx, ty, tz, and rotation variables r, p, y

    # 1. Compute point p in camera frame = pc = Rp + t
    # NOTE: Twc is from camera to world, we need world to camera: i.e. we have p = Rwc * pc + twc. Need to rearrange to get pc = Rwc^T * (p - twc) 
    
    Rwc = Twc[:3, :3]
    twc = Twc[:3, 3].reshape(-1, 1)
    pc = np.dot(Rwc.T, Wpt - twc) # 3x1 in camera frame: pc = (Xc, Yc, Zc)
    Xc, Yc, Zc = pc[0, 0], pc[1, 0], pc[2, 0]
    # 2. Define empty Jacobian matrix 2x6
    J = np.zeros((2, 6)) # 2x6, rows are u, v, columns are x, y, z, roll, pitch, yaw

    # 3. Compute jacobians wrt translation first: camera coordinates = [u,v]. Can get expressions for these by differentiating the projection equations wrt tx, ty, tz
    # have to use chain rule and stuff - implicit definitions above each line of code
    # First define fx, fy for clarity
    fx, fy = K[0, 0], K[1, 1]
    # du/dx = fx/Zc * (-r11 + Xc * r13/Zc) NOTE: r11 here is Rwc[0, 0], r13 is Rwc[0, 2]
    J[0, 0] = fx/Zc * (-Rwc[0, 0] + Xc * Rwc[0, 2]/Zc)
    # du/dy = fx/Zc * (-r21 + Xc * r23/Zc)
    J[0, 1] = fx/Zc * (-Rwc[1, 0] + Xc * Rwc[1, 2]/Zc)
    # du/dz = fx/Zc * (-r31 + Xc * r33/Zc)
    J[0, 2] = fx/Zc * (-Rwc[2, 0] + Xc * Rwc[2, 2]/Zc)
    # dv/dx = fy/Zc * (-r12 + Yc * r13/Zc)
    J[1, 0] = fy/Zc * (-Rwc[0, 1] + Yc * Rwc[0, 2]/Zc)
    # dv/dy = fy/Zc * (-r22 + Yc * r23/Zc)
    J[1, 1] = fy/Zc * (-Rwc[1, 1] + Yc * Rwc[1, 2]/Zc)
    # dv/dz = fy/Zc * (-r32 + Yc * r33/Zc)
    J[1, 2] = fy/Zc * (-Rwc[2, 1] + Yc * Rwc[2, 2]/Zc)

    # 4. Compute Jacobians wrt rotation: have to make skew symmetric matrices
    # Following Assignment Doc: R = C_roll * c_pitch * C_yaw: goal then is to separate R into these 3 matrices, can do the skew stuff
    # du/droll = fx/Zc * (dXc/droll - Xc/Zc * dZc/droll): 

    # 4.a: Extract roll, pitch, yaw from Rwc (actually want Rcw tho? -> transpose later)
    roll, pitch, yaw = rpy_from_dcm(Rwc)
    roll, pitch, yaw = roll[0], pitch[0], yaw[0] # numpy floats
    # 4.b: Get Elementary Rotation Matrices C_roll, C_pitch, C_yaw corresponding to roll, pitch, yaw.
    C_roll = dcm_from_rpy(np.array([roll, 0, 0]))
    C_pitch = dcm_from_rpy(np.array([0, pitch, 0]))
    C_yaw = dcm_from_rpy(np.array([0, 0, yaw]))
    # 4.c: Compute skew symmetric matrices for each of these
    S_roll = np.array([[0, 0, 0], 
                       [0, 0, -1], 
                       [0, 1, 0]])
    S_pitch = np.array([[0, 0, 1], 
                        [0, 0, 0], 
                        [-1, 0, 0]])
    S_yaw = np.array([[0, -1, 0], 
                      [1, 0, 0], 
                      [0, 0, 0]])
    # 4.d: Calculate dCu/droll, dCu/dpitch, dCu/dyaw
    dCu_roll = C_yaw @ C_pitch @ S_roll @ C_roll
    dCp_pitch = C_yaw @ S_pitch @ C_pitch @ C_roll
    dCy_yaw = S_yaw @ C_yaw @ C_pitch @ C_roll

    # Transpose these matrices sicne they are in the world frame: need to put in camera frame, same with Wpts (transform to camera)
    dCu_roll = dCu_roll.T @ (Wpt - twc) # CANT use pc here since a rotation to camera frame was already applied, still want transformation
    dCp_pitch = dCp_pitch.T @ (Wpt - twc)
    dCy_yaw = dCy_yaw.T @ (Wpt - twc)
    # du/droll = fx/Zc * (dXc/droll - Xc/Zc * dZc/droll)
    J[0, 3] = fx/Zc * (dCu_roll[0] - Xc/Zc * dCu_roll[2])
    # du/dpitch = fx/Zc * (dXc/dpitch - Xc/Zc * dZc/dpitch)
    J[0, 4] = fx/Zc * (dCp_pitch[0] - Xc/Zc * dCp_pitch[2])
    # du/dyaw = fx/Zc * (dXc/dyaw - Xc/Zc * dZc/dyaw)
    J[0, 5] = fx/Zc * (dCy_yaw[0] - Xc/Zc * dCy_yaw[2])
    # dv/droll = fy/Zc * (dYc/droll - Yc/Zc * dZc/droll)
    J[1, 3] = fy/Zc * (dCu_roll[1] - Yc/Zc * dCu_roll[2])
    # dv/dpitch = fy/Zc * (dYc/dpitch - Yc/Zc * dZc/dpitch)
    J[1, 4] = fy/Zc * (dCp_pitch[1] - Yc/Zc * dCp_pitch[2])
    # dv/dyaw = fy/Zc * (dYc/dyaw - Yc/Zc * dZc/dyaw)
    J[1, 5] = fy/Zc * (dCy_yaw[1] - Yc/Zc * dCy_yaw[2])
    print(J)
    quit()
    #------------------
    J = None
    correct = isinstance(J, np.ndarray) and \
        J.dtype == np.float64 and J.shape == (2, 6)

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return J