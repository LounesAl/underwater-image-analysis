import cv2
import requests
import numpy as np
import logging

while True:
        
    # Read frames
    try:
        img_resp = requests.get(camera_device_id)
        img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
        frame = cv2.imdecode(img_arr, -1)

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
