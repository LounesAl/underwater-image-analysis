import os 
import sys
import logging
from utils.calibration import parse_calibration_settings_file
import cv2
import glob
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # UNDERWATER-IMAGE-ANALYSIS root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

# Retrieve the calibration parameters 
calibration_settings = parse_calibration_settings_file(os.path.join(ROOT, 'stereo_calibration','calibration_settings.yaml'))


# Calibrate single camera to obtain camera intrinsic parameters from saved frames.
def calibrate_camera_for_intrinsic_parameters(path_image):
    
    #NOTE: images_prefix contains camera name: "frames/camera0*".
    images_names = glob.glob(path_image + '/*.JPG')
    
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


    for i, frame in tqdm(enumerate(images), total=len(images), desc = f"Calibration Of The {Path(path_image).name.capitalize()}"):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #find the checkerboard
        ret, corners = cv2.findChessboardCorners(gray, (rows, columns), None)

        if ret == True:

            #Convolution size used to improve corner detection. Don't make this too large.
            conv_size = (11, 11)

            #opencv2 can attempt to improve the checkerboard coordinates
            corners = cv2.cornerSubPix(gray, corners, conv_size, (-1, -1), criteria)
        
            objpoints.append(objp)
            imgpoints.append(corners)


    ret, cmtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (width, height), None, None)
    logging.info(f'rmse: {ret}')
    logging.info(f'camera matrix:\n {cmtx}')
    logging.info(f'distortion coeffs: {dist}')

    return cmtx, dist, ret

#save camera intrinsic parameters to file
def save_camera_intrinsics(camera_matrix, distortion_coefs, camera_name):

    #create folder if it does not exist
    if not os.path.exists(os.path.join('.', 'camera_parameters')):
        os.mkdir(os.path.join('.', 'camera_parameters'))

    out_filename = os.path.join('.', 'camera_parameters', camera_name + '_intrinsics.dat')
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
    c0_images_names = sorted(glob.glob(frames_prefix_c0 + '/*.JPG'))
    c1_images_names = sorted(glob.glob(frames_prefix_c1 + '/*.JPG'))

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

    for frame0, frame1 in tqdm(zip(c0_images, c1_images), total=len(c0_images), desc = f'Stereo Calibration'):
        gray1 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        c_ret1, corners1 = cv2.findChessboardCorners(gray1, (rows, columns), None)
        c_ret2, corners2 = cv2.findChessboardCorners(gray2, (rows, columns), None)

        if c_ret1 == True and c_ret2 == True:

            corners1 = cv2.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
            corners2 = cv2.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)

            p0_c1 = corners1[0,0].astype(np.int32)
            p0_c2 = corners2[0,0].astype(np.int32)

            objpoints.append(objp)
            imgpoints_left.append(corners1)
            imgpoints_right.append(corners2)

    stereocalibration_flags = cv2.CALIB_FIX_INTRINSIC
    ret, CM1, dist0, CM2, dist1, R, T, E, F = cv2.stereoCalibrate(objpoints, imgpoints_left, imgpoints_right, mtx0, dist0,
                                                                 mtx1, dist1, (width, height), criteria = criteria, flags = stereocalibration_flags)

    logging.info(f'rmse: {ret}')
    return R, T



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

def main(path_folder_cam1, path_folder_cam2):
    """Step1. Obtain camera intrinsic matrices and save them"""
    # # camera0 intrinsics
    cmtx0, dist0, ret0 = calibrate_camera_for_intrinsic_parameters(path_folder_cam1) 
    save_camera_intrinsics(cmtx0, dist0, 'camera0') #this will write cmtx and dist to disk
    # camera1 intrinsics
    path_image = os.path.join('.', 'camera1')
    cmtx1, dist1, ret1 = calibrate_camera_for_intrinsic_parameters(path_folder_cam2)
    save_camera_intrinsics(cmtx1, dist1, 'camera1') #this will write cmtx and dist to disk
    
    #create folder if it does not exist
    if not os.path.exists(os.path.join('.' , 'camera_parameters')):
        os.mkdir(os.path.join('.' , 'camera_parameters'))
        
    # Open a file to save the dictionary contane cmtx, dist and ret to disk in pkl file
    with open(os.path.join('.', 'camera_parameters', 'mono_params.pkl'), 'wb') as f:
        # Save the dictionary to the file
        pickle.dump({'cmtx0': cmtx0, 'cmtx1': cmtx1, 'dist0': dist0, 'dist1': dist1, 'ret0': ret0, 'ret1': ret1}, f)

    """Step2. Use paired calibration pattern frames to obtain camera0 to camera1 rotation and translation"""
    R, T = stereo_calibrate(cmtx0, dist0, cmtx1, dist1, path_folder_cam1, path_folder_cam2)

    """Step3. Save calibration data where camera0 defines the world space origin."""
    #camera0 rotation and translation is identity matrix and zeros vector
    R0 = np.eye(3, dtype=np.float32)
    T0 = np.array([0., 0., 0.]).reshape((3, 1))

    save_extrinsic_calibration_parameters(R0, T0, R, T) #this will write R and T to disk
    R1 = R; T1 = T #to avoid confusion, camera1 R and T are labeled R1 and T1
    
    # Open a file to save the dictionary contane cmtx, dist and ret to disk in pkl file
    with open(os.path.join('.', 'camera_parameters', 'stereo_params.pkl'), 'wb') as f:
        # Save the dictionary to the file
        pickle.dump({'cmtx0': cmtx0, 'cmtx1': cmtx1, 'R': R, 'T': T}, f)
    

if __name__ == '__main__':
    main('stereo_calibration_/camera0', 'stereo_calibration_/camera1')
    
        
        
        