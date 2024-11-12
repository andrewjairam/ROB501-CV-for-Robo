import numpy as np
from ibvs_controller import ibvs_controller
from ibvs_simulation import ibvs_simulation
from dcm_from_rpy import dcm_from_rpy
import time
import matplotlib.pyplot as plt
# Camera intrinsics matrix - known.
K = np.array([[500.0, 0, 400.0], 
              [0, 500.0, 300.0], 
              [0,     0,     1]])

# Target points (in target/object frame).
pts = np.array([[-0.75,  0.75, -0.75,  0.75],
                [-0.50, -0.50,  0.50,  0.50],
                [ 0.00,  0.00,  0.00,  0.00]])

# Camera poses, last and first.
C_last = np.eye(3)
t_last = np.array([[ 0.0, 0.0, -4.0]]).T
#C_init = dcm_from_rpy([np.pi/10, -np.pi/8, -np.pi/8])
C_init = dcm_from_rpy([np.pi/10, -np.pi/6, np.pi/4])
t_init = np.array([[-3, 0.3, -5.0]]).T

Twc_last = np.eye(4)
Twc_last[0:3, :] = np.hstack((C_last, t_last))
Twc_init = np.eye(4)
Twc_init[0:3, :] = np.hstack((C_init, t_init))

gain = 1.75 # 0.1

# Sanity check the controller output if desired.
# ...
gains = [0.1, 0.5, 1, 1.25, 1.5, 1.75, 2]
# Run simulation - estimate depths.
for gain in gains:
    for use_zest in [True, False]:
        start_time = time.time()
        ibvs_simulation(Twc_init, Twc_last, pts, K, gain, use_zest)
        end_time = time.time()     # Record the end time
        elapsed_time = end_time - start_time
        print(f"Time taken to run the function with gain {gain} and use_zest {use_zest}: {elapsed_time:.4f} seconds")
        plt.savefig(f'results_for_writeup/part_03_ibvs_simulation_gain_{gain}_use_z_est_{use_zest}.png')
# start_time = time.time()
# ibvs_simulation(Twc_init, Twc_last, pts, K, gain, True)
# end_time = time.time()     # Record the end time
# elapsed_time = end_time - start_time
# print(f"Time taken to run the function: {elapsed_time:.4f} seconds")
# plt.savefig('part_03_ibvs_simulation.png')