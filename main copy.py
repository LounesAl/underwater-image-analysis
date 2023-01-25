from utils.segmentation import *
from utils.calibration import *

visualize = True

im2 = 'data/imgs_c1/gibbula_07_16-1097-_jpg.rf.632cd8777f2b205c4fcd37545105890d.jpg'
im1 = 'data/imgs_c1/gibbula_07_16-1123-_jpg.rf.7b95db6ec76cb17ba97041006273c7c9.jpg'

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

# Get segmentation points " A optimiser "
uvs1, seg1, white_img1, im1_seg = get_segment_points(output1, im1, show=visualize)
uvs2, seg2, white_img2, im2_seg = get_segment_points(output2, im2, show=visualize)

# transforme the 2D points in the images to 3D points in the exit()world
# Il faut avoir le meme nombre de pairs de points dans les deux images
if len(uvs1) == len(uvs2):
    p3dss = transforme_to_3D(P1, P2, uvs1, uvs2)
else:
    pass
    
######################
## len(p3dss) = 2   ## => car il y'a deux espces detectées
## p3dss[0].shape = ## 
## (4, 3)           ## => 4 points (largeur longuer), 3 c'est x,y,z
######################

if visualize:
    # visualize the 3D points
    show_scatter_3D(p3dss)
        
distances, connections = get_3D_distances(p3dss, connections = [[0,2], [1,3]])

print(distances)

# on a 4 distances ici ... 2 espces, chaque espces a deux distances (longeur et largeur)
# distances = [[36.848607476959444, 32.03362311274617], [29.84703379255751, 45.008806942883936]]