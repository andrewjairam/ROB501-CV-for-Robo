import cv2
import numpy as np
from matplotlib.path import Path
from imageio import imread, imwrite

from dlt_homography import dlt_homography
from bilinear_interp import bilinear_interp
from histogram_eq import histogram_eq

from matplotlib import pyplot as plt





if __name__ == "__main__":
    # Iyd_pts = np.array([[416, 485, 488, 410], [40,  61, 353, 349]])
    # Ist_pts = np.array([[2, 218, 218, 2], [2, 2, 409, 409]])

    # Define the points
    Iyd_pts = np.array([[416, 485, 488, 410], [40,  61, 353, 349]], dtype=np.float32)
    Ist_pts = np.array([[2, 218, 218, 2], [2, 2, 409, 409]], dtype=np.float32)

    Iyd = imread('../images/yonge_dundas_square.jpg')
    Ist = imread('../images/uoft_soldiers_tower_light.png')

    Ist_eq = histogram_eq(Ist)

    H, status = cv2.findHomography(Ist_pts.T, Iyd_pts.T)
    print(H)
    vertices = [(354, 404),  (354, 490), (38, 490), (38, 404)] # Bottom left, # Bottom right, # Top left, # Top right, in y, x
    bbox_path = Path(vertices)
    H_inv = np.linalg.inv(H)
    for x in range(Iyd.shape[0]):
        for y in range(Iyd.shape[1]):
            # Check if the pixel is in the bbox:
            if bbox_path.contains_point([x, y]):
            #if cv2.pointPolygonTest(Iyd_pts.T, (x, y), False) > 0:
                # Find the corresponding pixel in Ist_eq using bilinear interpolation
            
                src = np.dot(H_inv, np.array([x, y, 1]))

                x_prime, y_prime = src[0] / src[2], src[1] / src[2]
                if x_prime > 0:
                    print("POSITIVE ENTRY FOR X REACHED")
                    #Ihack[x, y] = bilinear_interp(Ist_eq, x_prime, y_prime)