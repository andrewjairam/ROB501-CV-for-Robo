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
    # Bounding box in Y & D Square image - use if you find useful.
    #bbox = np.array([[404, 490, 404, 490], [38,  38, 354, 354]])
    #vertices = [(404, 354),  (490, 354), (490, 38), (404, 38)] # Bottom left, # Bottom right, # Top left, # Top right
    vertices = [(354, 404),  (354, 490), (38, 490), (38, 404)] # Bottom left, # Bottom right, # Top left, # Top right, in y, x
    #bbox = np.array([[38,  38, 354, 354], [404, 490, 404, 490]])
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
    # Compute the perspective homography we need: want to warp Ist_eq = I1, Iyd = I2 s.t. Iyd = H * Ist_eq
    H, _ = dlt_homography(Ist_pts, Iyd_pts)
    print(H)
    quit()

    # Main 'for' loop to do the warp and insertion - 
    # this could be vectorized to be faster if needed!
    
    # You may wish to make use of the contains_points() method
    # available in the matplotlib.path.Path class!

    # 1. Define path bbox_path that contains the billboard in Iyd
    bbox_path = Path(vertices)
 
    #bbox_path = Path(np.array([[bbox[0, i], bbox[1, i]] for i in range(bbox.shape[1])]))
    # 2. Define the inverse homography H_inv
    H_inv = np.linalg.inv(H)
    # 3. Iterate over all pixels in Iyd, and find the corresponding pixel in Ist_eq using bilinear interpolation
    for i in range(Iyd.shape[0]):
        for j in range(Iyd.shape[1]):
            # Check if the pixel is in the bbox:
            if bbox_path.contains_point([i, j]):
                Iyd_pxl = np.array([i, j, 1])
                src_pxl = np.dot(H_inv, Iyd_pxl)
                # Normalize src_pixel to [x, y, 1]
                src_pxl = src_pxl / src_pxl[-1] 
                print(f"src_pxl: {src_pxl}")


                # OLD code
                # Iyd[i, j, :] = np.array([255, 0, 0])
                # # Compute the corresponding pixel in Ist_eq using H_inv and bilinear interpolation, for each RGB channel
                # pt = np.array([[i, j, 1]]).T
                # pt = np.dot(H_inv, pt)#[:2]
                # pt = pt / pt[-1] # Maybe include normalization to increase accuracy of getting correct pixel
                # pt = pt[:2] # Ignore last entry
                # # Clip the pixel to be within the image: deals with negative values, and values greater than the image size
                # if pt[0] > 0:
                #     print("POSITIVE HIT")
                # #print(f"i: {i}, j: {j}, pt: {pt}")
                # pt[0] = np.clip(pt[0], 0, Ist.shape[0] - 1)
                # pt[1] = np.clip(pt[1], 0, Ist.shape[1] - 1)
                # #print(f"i: {i}, j: {j}, pt: {pt}")
                # b = bilinear_interp(Ist_eq, pt) # Gets corresponding grayscale pixel in Ist_eq
                # # Replace RGB pixel with b
                # Iyd[i, j, :] = b
    quit()
    Ihack = Iyd

    plt.imshow(Ihack)
    plt.show()
    imwrite(Ihack, 'billboard_hacked.png')












    #             # pt = np.array([[i], [j]])
    #             # pt = np.dot(H_inv, pt)
    #             # pt = pt / pt[-1]
    #             # b = bilinear_interp(Ist_eq, pt)
    #             # # If the pixel is in the bbox, replace the pixel in Iyd with the pixel in Ist_eq
    #             # Ihack[i, j, :] = b
    #         # # 3a. Compute the corresponding pixel in Ist_eq using H_inv and bilinear interpolation
    #         # pt = np.array([[i], [j]])
    #         # pt = np.dot(H_inv, pt)
    #         # pt = pt / pt[-1]
    #         # b = bilinear_interp(Ist_eq, pt)
    #         # # 3b. If the pixel is in the bbox, replace the pixel in Iyd with the pixel in Ist_eq
    #         # if bbox_path.contains_point([i, j]):
    #         #     Ihack[i, j, :] = b
    # # Find indices of pixels in Iyd that are in bbox, using Path.contains_points

    # print(np.array([[bbox[0, i], bbox[1, i]] for i in range(bbox.shape[1])]))

    # bbox_path = Path(np.array([[bbox[0, i], bbox[1, i]] for i in range(bbox.shape[1])]))
    # print(bbox_path.contains_points(np.array([[0, 1]])))
    # quit()
    # inside = path.contains_points(np.array([[Iyd_pts[0, i], Iyd_pts[1, i]] for i in range(4)]))

    #------------------

    # Visualize the result, if desired...
    # plt.imshow(Ihack)
    # plt.show()
    # imwrite(Ihack, 'billboard_hacked.png');
    return Ihack

if __name__ == "__main__":
    billboard_hack()