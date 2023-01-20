import numpy as np
import cv2 as cv
from utils import *
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

uvs1, frame1  = get_points_mouse('stereo_calibration/Images1.jpeg')
uvs2, frame2 = get_points_mouse('stereo_calibration/Images2.jpeg')

print(uvs1)
print('--------------')
print(uvs2)

# Show the two images with their 2D points
show_scatter_2D(frame1, frame2, uvs1, uvs2)

""" The next step is to obtain the projection matrices. This is done simply by multiplying the camera matrix by the rotation and translation matrix. """

# Load the result of the calibration
mtx1, mtx2, R, T = np.load('stereo_calibration/camera_parameters/stereo_params.npy', allow_pickle=True)

# Calculate the projection martrix
P1, P2 = get_projection_matrix(mtx1, mtx2, R, T)

# transforme the 2D points in the images to 3D points in the world
p3ds = transforme_to_3D(P1, P2, uvs1, uvs2)

# Show the 3D points
show_scatter_3D(p3ds)

