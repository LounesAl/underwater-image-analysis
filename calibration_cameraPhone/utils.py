import os
import yaml
import logging
import sys
import numpy as np
from scipy import linalg
import cv2

def start_message(image, text):

    # Définir les propriétés du texte
    font = cv2.FONT_HERSHEY_COMPLEX
    text_size = cv2.getTextSize(text, font, 1, 2)[0]
    text_x = (image.shape[1] - text_size[0]) // 2
    text_y = (image.shape[0] + text_size[1]) // 2

    # Définir les propriétés de l'ombre du texte
    shadow_color = (0, 0, 0)
    shadow_x_offset = 2
    shadow_y_offset = 2

    # Ajouter l'ombre du texte à l'image
    cv2.putText(image, text, (text_x+shadow_x_offset, text_y+shadow_y_offset), font, 1, shadow_color, 2)

    # Ajouter le texte à l'image
    cv2.putText(image, text, (text_x, text_y), font, 1, (255, 255, 255), 3)
    return image


# Open and load the calibration_settings.yaml file
def parse_calibration_settings_file(filename):
    if not os.path.exists(filename):
        logging.error(f'File does not exist: {filename}')
        sys.exit()
    
    logging.info(f'Using for calibration settings: {filename}')

    with open(filename) as f:
        calibration_settings = yaml.safe_load(f)

    # rudimentray check to make sure correct file was loaded
    if 'camera0' not in calibration_settings.keys():
        logging.error(f'camera0 key was not found in the settings file. Check if correct {filename} file was passed')
        sys.exit()
        
    return calibration_settings

#Given Projection matrices P1 and P2, and pixel coordinates point1 and point2, return triangulated 3D point.
def DLT(P1, P2, point1, point2):

    A = [point1[1]*P1[2,:] - P1[1,:],
         P1[0,:] - point1[0]*P1[2,:],
         point2[1]*P2[2,:] - P2[1,:],
         P2[0,:] - point2[0]*P2[2,:]
        ]
    A = np.array(A).reshape((4,4))

    B = A.transpose() @ A
    U, s, Vh = linalg.svd(B, full_matrices = False)

    #print('Triangulated point: ')
    #print(Vh[3,0:3]/Vh[3,3])
    return Vh[3,0:3]/Vh[3,3]

