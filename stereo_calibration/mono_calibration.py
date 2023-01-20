import cv2
import imutils
import numpy as np
import os 
import time
import logging
import sys
import requests
import glob
from utils import (parse_calibration_settings_file, start_message)




# Current path 
CURRENT_PATH = os.path.dirname(os.path.abspath('__file__'))

# Retrieve the calibration parameters 
calibration_settings = parse_calibration_settings_file(r'./calibration_settings.yaml')

# Create logging
logging.basicConfig(level=logging.DEBUG, 
                    filename=os.path.join('./logFile.log'),
                    filemode='w',
                    format='%(asctime)s - %(levelname)s - %(message)s')



# Open camera stream and save frames
def save_frames_single_camera(camera_name):

    # create frames directory
    if not os.path.exists(os.path.join(CURRENT_PATH, f'frames_{camera_name}')):
        os.mkdir(os.path.join(CURRENT_PATH, f'frames_{camera_name}'))
    
    # get settings
    camera_device_id = calibration_settings[camera_name]
    width = calibration_settings['frame_width']
    height = calibration_settings['frame_height']
    number_to_save = calibration_settings['mono_calibration_frames']
    view_resize = calibration_settings['view_resize']
    cooldown_time = calibration_settings['cooldown']
    
    cooldown = cooldown_time
    start = False
    saved_count = 0

    while True:
        
        # Read frames
        try:
            img_resp = requests.get(camera_device_id)
            img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
            frame = cv2.imdecode(img_arr, -1)
            frame = imutils.resize(frame, width=width, height=height)
        except Exception as e:
            #if no video data is received, can't calibrate the camera, so exit.
            logging.error(f"No video data received from camera. Exiting...")
            sys.exit()

        frame_small = cv2.resize(frame, None, fx = 1/view_resize, fy=1/view_resize)

        if not start:
            frame_small = start_message(frame_small, "Press SPACEBAR to start collection frames")
        
        if start:
            cooldown -= 1
            cv2.putText(frame_small, "Cooldown: " + str(cooldown), (8,25), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 0)
            cv2.putText(frame_small, "Num frames: " + str(saved_count), (8,65), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 0)
            
            #save the frame when cooldown reaches 0.
            if cooldown <= 0:
                id_no = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(time.time()))
                savename = os.path.join(CURRENT_PATH, f'frames_{camera_name}', str(camera_name) + '_' + id_no + '_' + str(saved_count) + '.png')
                cv2.imwrite(savename, frame)
                saved_count += 1
                cooldown = cooldown_time

        cv2.imshow('frame_small', frame_small)
        k = cv2.waitKey(1)
        
        if k == 27:
            # if ESC is pressed at any time, the program will exit.
            sys.exit()

        if k == 32:
            # Press spacebar to start data collection
            start = True

        # break out of the loop when enough number of frames have been saved
        if saved_count == number_to_save: break

    cv2.destroyAllWindows()


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

    """Step1. Save calibration frames for single cameras"""
    save_frames_single_camera('camera0') #save frames for camera0
    save_frames_single_camera('camera1') #save frames for camera1


    """Step2. Obtain camera intrinsic matrices and save them"""
    # camera0 intrinsics
    images_prefix = os.path.join('frames_camera0', 'camera0*')
    cmtx0, dist0, ret0 = calibrate_camera_for_intrinsic_parameters(images_prefix) 
    # save_camera_intrinsics(cmtx0, dist0, 'camera0') #this will write cmtx and dist to disk
    # camera1 intrinsics
    images_prefix = os.path.join('frames_camera1', 'camera1*')
    cmtx1, dist1, ret1 = calibrate_camera_for_intrinsic_parameters(images_prefix)
    # save_camera_intrinsics(cmtx1, dist1, 'camera1') #this will write cmtx and dist to disk

    #create folder if it does not exist
    if not os.path.exists(os.path.join(CURRENT_PATH, 'camera_parameters')):
        os.mkdir(os.path.join(CURRENT_PATH, 'camera_parameters'))
        
    # this will write cmtx, dist and ret to disk in numpy file 
    np.save(os.path.join(CURRENT_PATH, 'camera_parameters', 'mono_params.npy'), [cmtx0, cmtx1, dist0, dist1, ret0, ret1])


