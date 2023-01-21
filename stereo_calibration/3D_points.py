import numpy as np
import cv2 as cv
from utils import *
import matplotlib
matplotlib.use("TkAgg")
from mpl_toolkits.mplot3d import Axes3D

uvs1 = np.array([[408, 358],
       [681, 336],
       [694, 514],
       [410, 536]])

uvs2 = np.array([[654, 354],
       [924, 332],
       [948, 503],
       [666, 523]])

# uvs1, frame1  = get_points_mouse('stereo_calibration/Images1.jpeg')
# uvs2, frame2 = get_points_mouse('stereo_calibration/Images2.jpeg')

# print(uvs1)
# print('--------------')
# print(uvs2)

# # Show the two images with their 2D points
# show_scatter_2D(frame1, frame2, uvs1, uvs2)

""" The next step is to obtain the projection matrices. This is done simply by multiplying the camera matrix by the rotation and translation matrix. """

# Load the result of the calibration

mtx1 = np.array([[936.33696472,   0.        , 614.70026792],
       [  0.        , 940.48690607, 324.3100667 ],
       [  0.        ,   0.        ,   1.        ]])

mtx2 = np.array([[1.40338287e+03, 0.00000000e+00, 8.23978654e+02],
       [0.00000000e+00, 1.36075117e+03, 1.81336100e+02],
       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

R = np.array([[ 0.9889639 ,  0.14572227,  0.02674749],
       [-0.1342602 ,  0.80513959,  0.57768888],
       [ 0.06264667, -0.57490457,  0.81581869]])

T = np.array([[-30.3484974 ],
       [-14.70076729],
       [ 25.45162356]])

# dict_ = np.load('stereo_calibration/camera_parameters/stereo_params.npy', allow_pickle=True)
# mtx1, mtx2, R, T = dict_['mtx1'], dict_['mtx2'], dict_['R'], dict_['T']

# Calculate the projection martrix
P1, P2 = get_projection_matrix(mtx1, mtx2, R, T)

# transforme the 2D points in the images to 3D points in the world
p3ds = transforme_to_3D(P1, P2, uvs1, uvs2)

# Show the 3D points
show_scatter_3D(p3ds)

print('OK')