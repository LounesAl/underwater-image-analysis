import os
from pathlib import Path
import sys
import glob

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # UNDERWATER-IMAGE-ANALYSIS root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from utils.segmentation import *
from utils.calibration import *
# from utils.dataloaders import (IMG_FORMATS, VID_FORMATS, check_file, increment_path, select_device, check_imshow, LoadImages, LoadScreenshots, LoadStreams)

visualize = True

im2 = 'data/imgs_c1/gibbula_07_16-1097-_jpg.rf.632cd8777f2b205c4fcd37545105890d.jpg'
im1 = 'data/imgs_c2/image1838_jpg.rf.ba882ef8aecfce01c9175ae9a4b25c24.jpg'

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

print("OK")

# # transforme the 2D points in the images to 3D points in the exit()world
# p3ds = transforme_to_3D(P1, P2, uvs1, uvs2)

# if visualize:
#     # visualize the 3D points
#     show_scatter_3D(p3ds)
        
# distances, connections = get_3D_distances(p3ds, connections = [[0,2], [1,3]])