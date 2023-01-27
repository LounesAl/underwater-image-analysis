import os 
import sys
import logging
from utils import (parse_calibration_settings_file, start_message, DLT)
import cv2
import glob
import numpy as np
import imutils
from pathlib import Path

# Current path 
CURRENT_PATH = os.path.dirname(os.path.abspath('__file__'))


# Retrieve the calibration parameters 
calibration_settings = parse_calibration_settings_file(os.path.join(CURRENT_PATH, 'calibration_settings.yaml'))


# Calibrate single camera to obtain camera intrinsic parameters from saved frames.
def calibrate_camera_for_intrinsic_parameters(path_image):
    
    #NOTE: images_prefix contains camera name: "frames/camera0*".
    images_names = glob.glob(path_image + '/*.JPG')
    print(images_names)
    #read all frames
    images = [cv2.imread(imname, 1) for imname in images_names]

    #criteria used by checkerboard pattern detector.
    #Change this if the code can't find the checkerboard. 
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)

    rows = calibration_settings['checkerboard_rows']
    columns = calibration_settings['checkerboard_columns']
    world_scaling = calibration_settings['checkerboard_box_size_scale'] #this will change to user defined length scale

    #coordinates of squares in the checkerboard world space
    objp = np.zeros((rows*columns,3), np.float32)
    objp[:,:2] = np.mgrid[0:rows,0:columns].T.reshape(-1,2)
    objp = world_scaling* objp

    #frame dimensions. Frames should be the same size.
    width = images[0].shape[1]
    height = images[0].shape[0]

    #Pixel coordinates of checkerboards
    imgpoints = [] # 2d points in image plane.

    #coordinates of the checkerboard in checkerboard world space.
    objpoints = [] # 3d point in real world space


    for i, frame in enumerate(images):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #find the checkerboard
        ret, corners = cv2.findChessboardCorners(gray, (rows, columns), None)

        if ret == True:

            #Convolution size used to improve corner detection. Don't make this too large.
            conv_size = (11, 11)

            #opencv2 can attempt to improve the checkerboard coordinates
            corners = cv2.cornerSubPix(gray, corners, conv_size, (-1, -1), criteria)
            cv2.drawChessboardCorners(frame, (rows,columns), corners, ret)
            cv2.putText(frame, 'If detected points are poor, press "s" to skip this sample', (25, 25), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1)

            cv2.imshow('img', frame)
            k = cv2.waitKey(0)

            if k & 0xFF == ord('s'):
                logging.info(f'skipping')
                continue

            objpoints.append(objp)
            imgpoints.append(corners)


    cv2.destroyAllWindows()
    ret, cmtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (width, height), None, None)
    logging.info(f'rmse: {ret}')
    logging.info(f'camera matrix:\n {cmtx}')
    logging.info(f'distortion coeffs: {dist}')

    return cmtx, dist

#save camera intrinsic parameters to file
def save_camera_intrinsics(camera_matrix, distortion_coefs, camera_name):

    #create folder if it does not exist
    if not os.path.exists(os.path.join(CURRENT_PATH, 'camera_parameters')):
        os.mkdir(os.path.join(CURRENT_PATH, 'camera_parameters'))

    out_filename = os.path.join(CURRENT_PATH, 'camera_parameters', camera_name + '_intrinsics.dat')
    outf = open(out_filename, 'w')

    outf.write('intrinsic:\n')
    for l in camera_matrix:
        for en in l:
            outf.write(str(en) + ' ')
        outf.write('\n')

    outf.write('distortion:\n')
    for en in distortion_coefs[0]:
        outf.write(str(en) + ' ')
    outf.write('\n')



#open paired calibration frames and stereo calibrate for cam0 to cam1 coorindate transformations
def stereo_calibrate(mtx0, dist0, mtx1, dist1, frames_prefix_c0, frames_prefix_c1):
    #read the synched frames
    c0_images_names = sorted(glob.glob(frames_prefix_c0))
    c1_images_names = sorted(glob.glob(frames_prefix_c1))

    #open images
    c0_images = [cv2.imread(imname, 1) for imname in c0_images_names]
    c1_images = [cv2.imread(imname, 1) for imname in c1_images_names]

    #change this if stereo calibration not good.
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)

    #calibration pattern settings
    rows = calibration_settings['checkerboard_rows']
    columns = calibration_settings['checkerboard_columns']
    world_scaling = calibration_settings['checkerboard_box_size_scale']

    #coordinates of squares in the checkerboard world space
    objp = np.zeros((rows*columns,3), np.float32)
    objp[:,:2] = np.mgrid[0:rows,0:columns].T.reshape(-1,2)
    objp = world_scaling* objp

    #frame dimensions. Frames should be the same size.
    width = c0_images[0].shape[1]
    height = c0_images[0].shape[0]

    #Pixel coordinates of checkerboards
    imgpoints_left = [] # 2d points in image plane.
    imgpoints_right = []

    #coordinates of the checkerboard in checkerboard world space.
    objpoints = [] # 3d point in real world space

    for frame0, frame1 in zip(c0_images, c1_images):
        gray1 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        c_ret1, corners1 = cv2.findChessboardCorners(gray1, (rows, columns), None)
        c_ret2, corners2 = cv2.findChessboardCorners(gray2, (rows, columns), None)

        if c_ret1 == True and c_ret2 == True:

            corners1 = cv2.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
            corners2 = cv2.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)

            p0_c1 = corners1[0,0].astype(np.int32)
            p0_c2 = corners2[0,0].astype(np.int32)

            cv2.putText(frame0, 'O', (p0_c1[0], p0_c1[1]), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1)
            cv2.drawChessboardCorners(frame0, (rows,columns), corners1, c_ret1)
            cv2.imshow('img', frame0)

            cv2.putText(frame1, 'O', (p0_c2[0], p0_c2[1]), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1)
            cv2.drawChessboardCorners(frame1, (rows,columns), corners2, c_ret2)
            cv2.imshow('img2', frame1)
            k = cv2.waitKey(0)

            if k & 0xFF == ord('s'):
                logging.info('skipping')
                continue

            objpoints.append(objp)
            imgpoints_left.append(corners1)
            imgpoints_right.append(corners2)

    stereocalibration_flags = cv2.CALIB_FIX_INTRINSIC
    ret, CM1, dist0, CM2, dist1, R, T, E, F = cv2.stereoCalibrate(objpoints, imgpoints_left, imgpoints_right, mtx0, dist0,
                                                                 mtx1, dist1, (width, height), criteria = criteria, flags = stereocalibration_flags)

    logging.info(f'rmse: {ret}')
    cv2.destroyAllWindows()
    return R, T

#Converts Rotation matrix R and Translation vector T into a homogeneous representation matrix
def _make_homogeneous_rep_matrix(R, t):
    P = np.zeros((4,4))
    P[:3,:3] = R
    P[:3, 3] = t.reshape(3)
    P[3,3] = 1
 
    return P
# Turn camera calibration data into projection matrix
def get_projection_matrix(cmtx, R, T):
    P = cmtx @ _make_homogeneous_rep_matrix(R, T)[:3,:]
    return P

# After calibrating, we can see shifted coordinate axes in the video feeds directly
def check_calibration(camera0_name, camera0_data, camera1_name, camera1_data, _zshift = 50.):
    
    cmtx0 = np.array(camera0_data[0])
    dist0 = np.array(camera0_data[1])
    R0 = np.array(camera0_data[2])
    T0 = np.array(camera0_data[3])
    cmtx1 = np.array(camera1_data[0])
    dist1 = np.array(camera1_data[1])
    R1 = np.array(camera1_data[2])
    T1 = np.array(camera1_data[3])

    P0 = get_projection_matrix(cmtx0, R0, T0)
    P1 = get_projection_matrix(cmtx1, R1, T1)

    #define coordinate axes in 3D space. These are just the usual coorindate vectors
    coordinate_points = np.array([[0.,0.,0.],
                                  [1.,0.,0.],
                                  [0.,1.,0.],
                                  [0.,0.,1.]])
    z_shift = np.array([0.,0.,_zshift]).reshape((1, 3))
    #increase the size of the coorindate axes and shift in the z direction
    draw_axes_points = 5 * coordinate_points + z_shift

    #project 3D points to each camera view manually. This can also be done using cv2.projectPoints()
    #Note that this uses homogenous coordinate formulation
    pixel_points_camera0 = []
    pixel_points_camera1 = []
    for _p in draw_axes_points:
        X = np.array([_p[0], _p[1], _p[2], 1.])
        
        #project to camera0
        uv = P0 @ X
        uv = np.array([uv[0], uv[1]])/uv[2]
        pixel_points_camera0.append(uv)

        #project to camera1
        uv = P1 @ X
        uv = np.array([uv[0], uv[1]])/uv[2]
        pixel_points_camera1.append(uv)

    #these contain the pixel coorindates in each camera view as: (pxl_x, pxl_y)
    pixel_points_camera0 = np.array(pixel_points_camera0)
    pixel_points_camera1 = np.array(pixel_points_camera1)

    

    #set camera resolutions
    width = calibration_settings['frame_width']
    height = calibration_settings['frame_height']
    
    while True:

        img_resp = requests.get(calibration_settings[camera0_name])
        img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
        frame0 = cv2.imdecode(img_arr, -1)
        frame0 = imutils.resize(frame0, width=width, height=height)

        img_resp = requests.get(calibration_settings[camera1_name])
        img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
        frame1 = cv2.imdecode(img_arr, -1)
        frame1 = imutils.resize(frame1, width=width, height=height)


        #follow RGB colors to indicate XYZ axes respectively
        colors = [(0,0,255), (0,255,0), (255,0,0)]
        #draw projections to camera0
        origin = tuple(pixel_points_camera0[0].astype(np.int32))
        for col, _p in zip(colors, pixel_points_camera0[1:]):
            _p = tuple(_p.astype(np.int32))
            cv2.line(frame0, origin, _p, col, 2)
        
        #draw projections to camera1
        origin = tuple(pixel_points_camera1[0].astype(np.int32))
        for col, _p in zip(colors, pixel_points_camera1[1:]):
            _p = tuple(_p.astype(np.int32))
            cv2.line(frame1, origin, _p, col, 2)

        cv2.imshow('frame0', frame0)
        cv2.imshow('frame1', frame1)

        k = cv2.waitKey(1)
        if k == 27: break

    cv2.destroyAllWindows()

def get_world_space_origin(cmtx, dist, img_path):

    frame = cv2.imread(img_path, 1)

    #calibration pattern settings
    rows = calibration_settings['checkerboard_rows']
    columns = calibration_settings['checkerboard_columns']
    world_scaling = calibration_settings['checkerboard_box_size_scale']

    #coordinates of squares in the checkerboard world space
    objp = np.zeros((rows*columns,3), np.float32)
    objp[:,:2] = np.mgrid[0:rows,0:columns].T.reshape(-1,2)
    objp = world_scaling* objp

    gray = cv2.cv2tColor(frame, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (rows, columns), None)

    cv2.drawChessboardCorners(frame, (rows,columns), corners, ret)
    cv2.putText(frame, "If you don't see detected points, try with a different image", (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1)
    cv2.imshow('img', frame)
    cv2.waitKey(0)

    ret, rvec, tvec = cv2.solvePnP(objp, corners, cmtx, dist)
    R, _  = cv2.Rodrigues(rvec) #rvec is Rotation matrix in Rodrigues vector form

    return R, tvec

def get_cam1_to_world_transforms(cmtx0, dist0, R_W0, T_W0, 
                                 cmtx1, dist1, R_01, T_01,
                                 image_path0,
                                 image_path1):

    frame0 = cv2.imread(image_path0, 1)
    frame1 = cv2.imread(image_path1, 1)

    unitv_points = 5 * np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1]], dtype = 'float32').reshape((4,1,3))
    #axes colors are RGB format to indicate XYZ axes.
    colors = [(0,0,255), (0,255,0), (255,0,0)]

    #project origin points to frame 0
    points, _ = cv2.projectPoints(unitv_points, R_W0, T_W0, cmtx0, dist0)
    points = points.reshape((4,2)).astype(np.int32)
    origin = tuple(points[0])
    for col, _p in zip(colors, points[1:]):
        _p = tuple(_p.astype(np.int32))
        cv2.line(frame0, origin, _p, col, 2)


    #project origin points to frame1
    R_W1 = R_01 @ R_W0
    T_W1 = R_01 @ T_W0 + T_01
    points, _ = cv2.projectPoints(unitv_points, R_W1, T_W1, cmtx1, dist1)
    points = points.reshape((4,2)).astype(np.int32)
    origin = tuple(points[0])
    for col, _p in zip(colors, points[1:]):
        _p = tuple(_p.astype(np.int32))
        cv2.line(frame1, origin, _p, col, 2)

    cv2.imshow('frame0', frame0)
    cv2.imshow('frame1', frame1)
    cv2.waitKey(0)

    return R_W1, T_W1


def save_extrinsic_calibration_parameters(R0, T0, R1, T1, prefix = ''):
    
    #create folder if it does not exist
    if not os.path.exists('camera_parameters'):
        os.mkdir('camera_parameters')

    camera0_rot_trans_filename = os.path.join('camera_parameters', prefix + 'camera0_rot_trans.dat')
    outf = open(camera0_rot_trans_filename, 'w')

    outf.write('R:\n')
    for l in R0:
        for en in l:
            outf.write(str(en) + ' ')
        outf.write('\n')

    outf.write('T:\n')
    for l in T0:
        for en in l:
            outf.write(str(en) + ' ')
        outf.write('\n')
    outf.close()

    #R1 and T1 are just stereo calibration returned values
    camera1_rot_trans_filename = os.path.join('camera_parameters', prefix + 'camera1_rot_trans.dat')
    outf = open(camera1_rot_trans_filename, 'w')

    outf.write('R:\n')
    for l in R1:
        for en in l:
            outf.write(str(en) + ' ')
        outf.write('\n')

    outf.write('T:\n')
    for l in T1:
        for en in l:
            outf.write(str(en) + ' ')
        outf.write('\n')
    outf.close()

    return R0, T0, R1, T1

if __name__ == '__main__':

    if len(sys.argv) != 2:
        logging.info(f'Call with settings filename: "python3 calibrate.py calibration_settings.yaml"')
        sys.exit()
    
    # Open and parse the settings file
    parse_calibration_settings_file(sys.argv[1])


    """Step2. Obtain camera intrinsic matrices and save them"""
    # camera0 intrinsics
    path_image = os.path.join(CURRENT_PATH, 'camera0')
    cmtx0, dist0 = calibrate_camera_for_intrinsic_parameters(path_image) 
    save_camera_intrinsics(cmtx0, dist0, 'camera0') #this will write cmtx and dist to disk
    # camera1 intrinsics
    path_image = os.path.join(CURRENT_PATH, 'camera1')
    cmtx1, dist1 = calibrate_camera_for_intrinsic_parameters(path_image)
    save_camera_intrinsics(cmtx1, dist1, 'camera1') #this will write cmtx and dist to disk


    """Step4. Use paired calibration pattern frames to obtain camera0 to camera1 rotation and translation"""
    frames_prefix_c0 = os.path.join(CURRENT_PATH, 'camera0')
    frames_prefix_c1 = os.path.join(CURRENT_PATH, 'camera1')
    R, T = stereo_calibrate(cmtx0, dist0, cmtx1, dist1, frames_prefix_c0, frames_prefix_c1)


    """Step5. Save calibration data where camera0 defines the world space origin."""
    #camera0 rotation and translation is identity matrix and zeros vector
    R0 = np.eye(3, dtype=np.float32)
    T0 = np.array([0., 0., 0.]).reshape((3, 1))

    save_extrinsic_calibration_parameters(R0, T0, R, T) #this will write R and T to disk
    R1 = R; T1 = T #to avoid confusion, camera1 R and T are labeled R1 and T1
    #check your calibration makes sense
    camera0_data = [cmtx0, dist0, R0, T0]
    camera1_data = [cmtx1, dist1, R1, T1]
    check_calibration('camera0', camera0_data, 'camera1', camera1_data, _zshift = 60.)


    """Optional. Define a different origin point and save the calibration data"""
    # #get the world to camera0 rotation and translation
    # R_W0, T_W0 = get_world_space_origin(cmtx0, dist0, os.path.join('frames_pair', 'camera0_4.png'))
    # #get rotation and translation from world directly to camera1
    # R_W1, T_W1 = get_cam1_to_world_transforms(cmtx0, dist0, R_W0, T_W0,
    #                                           cmtx1, dist1, R1, T1,
    #                                           os.path.join('frames_pair', 'camera0_4.png'),
    #                                           os.path.join('frames_pair', 'camera1_4.png'),)

    # #save rotation and translation parameters to disk
    # save_extrinsic_calibration_parameters(R_W0, T_W0, R_W1, T_W1, prefix = 'world_to_') #this will write R and T to disk
