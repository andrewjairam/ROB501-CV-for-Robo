import numpy as np
import matplotlib.pyplot as plt
from imageio import imread
from cross_junctions import cross_junctions
from scipy.ndimage.filters import *
from matplotlib.path import Path
# from cross_junctions_test import cross_junctions_test
# Load the world points.
Wpts = np.load('../data/world_pts.npy')

# Load the example target image.
I = imread("../images/target_01.png")

# Load the bounding polygon.
bpoly = np.load('../data/bounds_01.npy')

# Load the reference solution and compute yours.
Ipts_ref = np.load('../data/cross_junctions_01.npy')
Ipts = cross_junctions(I, bpoly, Wpts)


# # ANDREW TEST 1
# # TRY SMOOTHING
# #I = gaussian_filter(I, 1)

# #corners, patch_coords = cross_junctions(I, bpoly, Wpts)
# pts = cross_junctions_test(I, bpoly, Wpts)
# #quit()
# # BELOW: FROM BELOW
# # Plot the points to check!
# plt.imshow(I, cmap = 'gray')
# # ANDREW TEST 2
# bpoly = np.append(bpoly, bpoly[:, None, 0], axis = 1) # Close the polygon.
# plt.plot(bpoly[0, :], bpoly[1, :], '-', c = 'b', linewidth = 3)
# plt.plot(bpoly[0, 0], bpoly[1, 0], 'x', c = 'b', markersize = 9)
# plt.text(bpoly[0, 0] - 40, bpoly[1, 0] - 10, "Upper Left", c = 'b')

# # ANDREW TEST 3: PLOT HARRIS CORNERS
# # plt.scatter(np.argwhere(corners)[:, 1], np.argwhere(corners)[:, 0], color='r', s=5, label='Corners')
# # plt.title('Detected Corners')

# # ANDREW TEST 4: PLOT PATCHES
# #bpatch = np.append(patch_coords, patch_coords[:, None, 0], axis = 1) # Close the polygon.
# #plt.plot(bpatch[0, :], bpatch[1, :], '-', c = 'g', linewidth = 3)
# #plt.plot(bpatch[0, 0], bpatch[1, 0], 'x', c = 'g', markersize = 9)
# #plt.plot(patch_coords[0], patch_coords[1], 'x', c = 'g', markersize = 9)
# # TEST 4.5: PLOT PATCH BBOX
# # bpatchbbox = np.append(patchbbox, patchbbox[:, None, 0], axis = 1) # Close the polygon.
# # plt.plot(bpatchbbox[0, :], bpatchbbox[1, :], '-', c = 'g', linewidth = 3)
# # plt.plot(bpatchbbox[0, 0], bpatchbbox[1, 0], 'x', c = 'g', markersize = 9)
# # ANDREW TEST 5: PLOT CROSS JUNCTIONS FROM HOMO

# plt.scatter(pts[0, :], pts[1, :], c='r', marker='o', s=10, label='Cross Junctions')
# plt.legend()
# plt.show()
# quit()


plt.plot(Ipts_ref[0, :], Ipts_ref[1, :], 'o', c = 'r', markersize = 8)
for i in range(0, Ipts_ref.shape[1]):
    plt.text(Ipts_ref[0, i] - 10, Ipts_ref[1, i] - 10, str(i + 1), c = 'r')

bpoly = np.append(bpoly, bpoly[:, None, 0], axis = 1) # Close the polygon.
plt.plot(bpoly[0, :], bpoly[1, :], '-', c = 'b', linewidth = 3)
plt.plot(bpoly[0, 0], bpoly[1, 0], 'x', c = 'b', markersize = 9)
plt.text(bpoly[0, 0] - 40, bpoly[1, 0] - 10, "Upper Left", c = 'b')

plt.plot(Ipts[0, :], Ipts[1, :], 'o', c = 'g',)
plt.show()
