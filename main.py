"""Dans un nouveau fichier : 
1) Acquisition (d'image, video ou GIF) avec les deux cameras en stereo:
            -> Lecture des images en temps rell avec opencv ou autres.
            -> Distance entre les deux cameras (paramètre à optimisé).
            -> Espèce sous marine (sous aquarium) ex: GIBBULA, TIMHARINES.
          
2) Récupérer les coords  pixels de longueur et largeur de chaque espèces pour les deux caméras: 
            -> Inférence des espèces avec notre modèle detectron 2 avec les deux caméras stereo
            -> Fonctions supplémentaires pour l'extraction des coords

3) Conversion 2D -> 3D:"""


import logging
logging.info(f"model initialisation in progress ...")

import glob
from utils.segmentation import *
from utils.calibration import *
import PIL

logging.info(f"load camera params in progress ...")
mtx1, mtx2, R, T = load_calibration('stereo_calibration/camera_parameters/stereo_params.pkl')
# Calculate the projection martrix
P1, P2 = get_projection_matrix(mtx1, mtx2, R, T)

path_model = './models/model_final.pth'

images_c1 = glob.glob('dataset_download/test/imgs_c1/*.{jpg,jpeg,png,gif}', recursive=True)
images_c2 = glob.glob('dataset_download/test/imgs_c2/*.{jpg,jpeg,png,gif}', recursive=True)

show = True

for img_c1, img_c2 in zip(images_c1, images_c2):

    # Inferred with the images of each camera
    output1 = inference(path_model, img_c1, show=show)
    output2 = inference(path_model, img_c2, show=show)

    # Get segmentation points " A optimiser "
    uvs1, seg1 = get_segment_points(output1, 0, show=show)
    uvs2, seg2 = get_segment_points(output2, 0, show=show)
    
    # transforme the 2D points in the images to 3D points in the exit()world
    p3ds = transforme_to_3D(P1, P2, uvs1, uvs2)
    
    if show:
        # Show the 3D points
        show_scatter_3D(p3ds)
    
    distances, connections = get_3D_distances(p3ds, connections = [[0,2], [1,3]])
        
        

    




















