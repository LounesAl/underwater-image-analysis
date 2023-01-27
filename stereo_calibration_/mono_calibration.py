import pickle
import cv2
import numpy as np
import os 
import logging
import sys
import glob
from utils import (parse_calibration_settings_file)
from pathlib import Path



# Current path 
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # UNDERWATER-IMAGE-ANALYSIS root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


# Retrieve the calibration parameters 
calibration_settings = parse_calibration_settings_file(str(ROOT / 'calibration_settings.yaml'))

# Create logging
logging.basicConfig(level=logging.DEBUG, 
                    filename=os.path.join('./logFile.log'),
                    filemode='w',
                    format='%(asctime)s - %(levelname)s - %(message)s')



# Calibrate single camera to obtain camera intrinsic parameters from saved frames.
def calibrate_camera_for_intrinsic_parameters(images_prefix):
    
    #NOTE: images_prefix contains camera name: "frames/camera0*".
    images_names = glob.glob(images_prefix)

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

    return cmtx, dist, ret


if __name__ == '__main__':

    # camera0 intrinsics
    images_prefix = str(ROOT / 'camera0*')
    
    cmtx0, dist0, ret0 = calibrate_camera_for_intrinsic_parameters(images_prefix) 
    # save_camera_intrinsics(cmtx0, dist0, 'camera0') #this will write cmtx and dist to disk
    # camera1 intrinsics
    images_prefix = str(ROOT / 'camera1*')
    cmtx1, dist1, ret1 = calibrate_camera_for_intrinsic_parameters(images_prefix)
    # save_camera_intrinsics(cmtx1, dist1, 'camera1') #this will write cmtx and dist to disk

    #create folder if it does not exist
    if not os.path.exists(str(ROOT / 'camera_parameters')):
        os.mkdir(str(ROOT / 'camera_parameters'))
        
    # Open a file to save the dictionary contane cmtx, dist and ret to disk in pkl file
    with open(str(ROOT / 'camera_parameters/mono_params.pkl'), 'wb') as f:
        # Save the dictionary to the file
        pickle.dump({'cmtx0': cmtx0, 'cmtx1': cmtx1, 'dist0': dist0, 'dist1': dist1, 'ret0': ret0, 'ret1': ret1}, f)



