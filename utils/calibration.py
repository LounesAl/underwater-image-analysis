import os
import yaml
import logging
import sys
import numpy as np
from scipy import linalg
import pickle
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
    
    return Vh[3,0:3]/Vh[3,3]

def transforme_to_3D(P1, P2, uvs1, uvs2):
        p3dss = []
        # pour chaque especes dans l'image. len(p3dss) sera egale à 2 si y'a 2 gibbula
        for uv1, uv2 in zip(uvs1, uvs2):
            p3ds = []
            # pour pairs de points (un point dans chaque image). boucler 4 fois. len(p3ds) = 4
            for u1, u2 in zip(uv1, uv2):
                p3d = DLT(P1, P2, u1, u2)
                p3ds.append(p3d)
            p3ds = np.array(p3ds)
            p3dss.append(p3ds)
        # au total on renvoie 8 points, dans le cas où on a 2 especes detectées
        return p3dss
    
    
def get_projection_matrix(mtx1, mtx2, R, T):
        # RT matrix for C1 is identity.
        RT1 = np.concatenate([np.eye(3), [[0],[0],[0]]], axis = -1)
        P1 = mtx1 @ RT1 #projection matrix for C1
        # RT matrix for C2 is the R and T obtained from stereo calibration.
        RT2 = np.concatenate([R, T], axis = -1)
        P2 = mtx2 @ RT2 #projection matrix for C2
        return  P1, P2  

def show_scatter_2D(frame1, 
                    frame2, 
                    uvs1, 
                    uvs2
                    ):
        plt.imshow(frame1[:,:,[2,1,0]])
        plt.scatter(uvs1[:,0], uvs1[:,1])
        plt.show()
        plt.imshow(frame2[:,:,[2,1,0]])
        plt.scatter(uvs2[:,0], uvs2[:,1])
        plt.show()
        
def get_3D_distances(p3dss, 
                     connections = [[0,2], [1,3]]
                     ):
    distances = []
    for p3ds in p3dss:
        dists = []
        for _c in connections:
            point1 = p3ds[_c[0]]
            point2 = p3ds[_c[1]]
            dist = np.linalg.norm(point2 - point1) # Euclidian default
            dists.append(dist)
        distances.append(dists)
            
    return distances, connections
    
def show_scatter_3D(p3dss, 
                    connections = [[0,2], [1,3]], 
                    linewidth=2
                    ):
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d', proj_type = 'ortho')
    
    for p3ds in p3dss:
        for _c in connections:
            point1 = p3ds[_c[0]]
            point2 = p3ds[_c[1]]
            distance = np.linalg.norm(point2 - point1) # Euclidian default
            print("Distance entre les points {} et {}: {:.2f} cm".format(_c[0],_c[1], distance))
            mid_point = (point1 + point2)/2
            ax.text(mid_point[0], mid_point[1], mid_point[2], "Distance: {:.2f} cm".format(distance))
            ax.view_init(elev=20, azim=30)
            ax.plot(xs = [point1[0], point2[0]], ys = [point1[1], point2[1]], zs = [point1[2], point2[2]], c = 'red', linewidth=linewidth)#, linestyle='dotted')
            vectors = point2 - point1
            ax.quiver(point1[0], point1[1], point1[2], vectors[0], vectors[1], vectors[2], color='red', arrow_length_ratio=0.1)
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
    
def load_calibration(path_calib):
    # Open the file containing the saved dictionary
    with open(path_calib, "rb") as f:
        # Load the dictionary from the file
        loaded_data = pickle.load(f)
    mtx1, mtx2, R, T = loaded_data['cmtx0'], loaded_data['cmtx1'], loaded_data['R'], loaded_data['T']
    return mtx1, mtx2, R, T