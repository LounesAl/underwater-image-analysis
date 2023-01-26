from utils.segmentation import *
from utils.calibration import *

visualize = False

im2 = 'data/imgs_c2/image1832_jpg.rf.3f76332c090e81175da88367876dfcd8.jpg'
im1 = 'data/imgs_c2/image1842_jpg.rf.fc8794fa0e45edf088fbd294b4b66188.jpg'

im2 = cv2.imread(im2)
im1 = cv2.imread(im1)

weights = 'models/model_final.pth'

calib_cam = 'stereo_calibration/camera_parameters/stereo_params.pkl'

mtx1, mtx2, R, T = load_calibration(calib_cam)

# Calculate the projection martrix
P1, P2 = get_projection_matrix(mtx1, mtx2, R, T)

# Inferred with the images of each camera
output1 = inference(weights, im1, show=False)
output2 = inference(weights, im2, show=False)

# voir les classes predites
## le resultat est de la forme : tensor([0, 1, 1, 2, 3, 3]), cela veut dire :
## une espces PFE, deux actinia fermées, une ouverte et deux gibbula
classes1 = output1["instances"].pred_classes
classes2 = output1["instances"].pred_classes

print(classes1)
print(classes1)

# --------------------------------------------#

# maintenant il faut trouver comment traiter que 
# les class en commune entre l'image 1 et 2

# --------------------------------------------#

# Get segmentation points " A optimiser "
uvs1, seg1, boxes1 = get_segment_points(output1)
uvs2, seg2, boxes2 = get_segment_points(output2)

# transforme the 2D points in the images to 3D points in the exit()world
# Il faut avoir le meme nombre de pairs de points dans les deux images
if len(uvs1) == len(uvs2) and uvs1!=None:
    p3dss = transforme_to_3D(P1, P2, uvs1, uvs2)
else:
    pass #contninu
    
######################
## len(p3dss) = 2   ## => car il y'a deux espces detectées
## p3dss[0].shape = ## 
## (4, 3)           ## => 4 points (largeur longuer), 3 c'est x,y,z
######################

if visualize:
    # visualize the 3D points
    show_scatter_3D(p3dss)
        
distances, connections = get_3D_distances(p3dss, connections = [[0,2], [1,3]])

class_dict = {
    "0" : "PFE",
    "1" : "Actinia fermee",
    "2" : "Actinia ouverte",
    "3" : "Gibbula"
}

im1_seg = dist_on_img(uvs1, boxes1, im1, distances, classes1, class_dict, show=True)
im2_seg = dist_on_img(uvs2, boxes2, im2, distances, classes2, class_dict, show=True)

# on a 4 distances ici ... 2 espces, chaque espces a deux distances (longeur et largeur)
# distances = [[36.848607476959444, 32.03362311274617], [29.84703379255751, 45.008806942883936]]