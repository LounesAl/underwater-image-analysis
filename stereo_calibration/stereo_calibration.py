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



#open both cameras and take calibration frames
def save_frames_two_cams(camera0_name, camera1_name):

    #create frames directory
    if not os.path.exists('frames_pair'):
        os.mkdir('frames_pair')

    # settings for taking data
    view_resize = calibration_settings['view_resize']
    cooldown_time = calibration_settings['cooldown']    
    number_to_save = calibration_settings['stereo_calibration_frames']

    # set camera resolutions
    width = calibration_settings['frame_width']
    height = calibration_settings['frame_height']

    cooldown = cooldown_time
    start = False
    saved_count = 0
    while True:

        img_resp = requests.get(calibration_settings[camera0_name])
        img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
        frame0 = cv2.imdecode(img_arr, -1)
        frame0 = imutils.resize(frame0, width=width, height=height)

        img_resp = requests.get(calibration_settings[camera1_name])
        img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
        frame1 = cv2.imdecode(img_arr, -1)
        frame1 = imutils.resize(frame1, width=width, height=height)


        frame0_small = cv2.resize(frame0, None, fx=1./view_resize, fy=1./view_resize)
        frame1_small = cv2.resize(frame1, None, fx=1./view_resize, fy=1./view_resize)

        if not start:
            cv2.putText(frame0_small, "Make sure both cameras can see the calibration pattern well", (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1)
            cv2.putText(frame0_small, "Press SPACEBAR to start collection frames", (50,100), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1)
        
        if start:
            cooldown -= 1
            cv2.putText(frame0_small, "Cooldown: " + str(cooldown), (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 1)
            cv2.putText(frame0_small, "Num frames: " + str(saved_count), (50,100), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 1)
            
            cv2.putText(frame1_small, "Cooldown: " + str(cooldown), (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 1)
            cv2.putText(frame1_small, "Num frames: " + str(saved_count), (50,100), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 1)

            #save the frame when cooldown reaches 0.
            if cooldown <= 0:
                savename = os.path.join(CURRENT_PATH, 'frames_pair', camera0_name + '_' + str(saved_count) + '.png')
                cv2.imwrite(savename, frame0)

                savename = os.path.join(CURRENT_PATH, 'frames_pair', camera1_name + '_' + str(saved_count) + '.png')
                cv2.imwrite(savename, frame1)

                saved_count += 1
                cooldown = cooldown_time

        cv2.imshow('frame0_small', frame0_small)
        cv2.imshow('frame1_small', frame1_small)
        k = cv2.waitKey(1)
        
        if k == 27:
            #if ESC is pressed at any time, the program will exit.
            sys.exit()

        if k == 32:
            #Press spacebar to start data collection
            start = True

        #break out of the loop when enough number of frames have been saved
        if saved_count == number_to_save: break

    cv2.destroyAllWindows()



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


if __name__ == '__main__':

    cmtx0, cmtx1, dist0, dist1, _, _= np.load(os.path.join(CURRENT_PATH, 'camera_parameters', 'mono_params.npy'), allow_pickle=True)
    """Step3. Save calibration frames for both cameras simultaneously"""
    save_frames_two_cams('camera0', 'camera1') #save simultaneous frames


    """Step4. Use paired calibration pattern frames to obtain camera0 to camera1 rotation and translation"""
    frames_prefix_c0 = os.path.join('frames_pair', 'camera0*')
    frames_prefix_c1 = os.path.join('frames_pair', 'camera1*')
    R, T = stereo_calibrate(cmtx0, dist0, cmtx1, dist1, frames_prefix_c0, frames_prefix_c1)


    """Step5. Save calibration data where camera0 defines the world space origin."""
    #camera0 rotation and translation is identity matrix and zeros vector
    R0 = np.eye(3, dtype=np.float32)
    T0 = np.array([0., 0., 0.]).reshape((3, 1))

    # this will write cmtx, dist and ret to disk in numpy file 
    np.save(os.path.join(CURRENT_PATH, 'camera_parameters', 'stereo_params.npy'), [cmtx0, cmtx1, R, T])


    # save_extrinsic_calibration_parameters(R0, T0, R, T) #this will write R and T to disk
    # R1 = R; T1 = T #to avoid confusion, camera1 R and T are labeled R1 and T1
    # #check your calibration makes sense
    # camera0_data = [cmtx0, dist0, R0, T0]
    # camera1_data = [cmtx1, dist1, R1, T1]
    # check_calibration('camera0', camera0_data, 'camera1', camera1_data, _zshift = 60.)

