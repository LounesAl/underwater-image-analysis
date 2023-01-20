import os
import yaml
import logging
import sys
import numpy as np
from scipy import linalg
import cv2
import matplotlib.pyplot as plt

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

def transforme_to_3D(P1, P2, uvs1, uvs2):
        p3ds = []
        for uv1, uv2 in zip(uvs1, uvs2):
                _p3d = DLT(P1, P2, uv1, uv2)
        p3ds.append(_p3d)
        p3ds = np.array(p3ds)
        return p3ds
    
    
def get_projection_matrix(mtx1, mtx2, R, T):
        # RT matrix for C1 is identity.
        RT1 = np.concatenate([np.eye(3), [[0],[0],[0]]], axis = -1)
        P1 = mtx1 @ RT1 #projection matrix for C1
        # RT matrix for C2 is the R and T obtained from stereo calibration.
        RT2 = np.concatenate([R, T], axis = -1)
        P2 = mtx2 @ RT2 #projection matrix for C2
        return  P1, P2  

def show_scatter_2D(frame1, frame2, uvs1, uvs2):
        plt.imshow(frame1[:,:,[2,1,0]])
        plt.scatter(uvs1[:,0], uvs1[:,1])
        plt.show()
        plt.imshow(frame2[:,:,[2,1,0]])
        plt.scatter(uvs2[:,0], uvs2[:,1])
        plt.show()
        
def show_scatter_3D(p3ds):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim3d(-15, 5)
        ax.set_ylim3d(-10, 10)
        ax.set_zlim3d(10, 30)
        connections = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,8], [1,9], [2,8], [5,9], [8,9], [0, 10], [0, 11]]
        for _c in connections:
                print(p3ds[_c[0]])
                print(p3ds[_c[1]])
                ax.plot(xs = [p3ds[_c[0],0], p3ds[_c[1],0]], ys = [p3ds[_c[0],1], p3ds[_c[1],1]], zs = [p3ds[_c[0],2], p3ds[_c[1],2]], c = 'red')
        plt.show()
        
def get_points_mouse(img_path):
        points = []
        def on_mouse(event, x, y, flags, param):
                if event == cv2.EVENT_LBUTTONDOWN:
                        points.append((x, y))
        img = cv2.imread(img_path)
        cv2.namedWindow("image")
        cv2.setMouseCallback("image", on_mouse)
        while True:
                cv2.imshow("image", img)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        cv2.destroyAllWindows()
        return np.array(points), img