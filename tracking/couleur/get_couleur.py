#code pour suivre la couleur d'un objet : 
#bibliotheque : 

from segmentation import *
from glob import glob
from webcolors import rgb_to_hex
from webcolors import (
    CSS3_HEX_TO_NAMES,
    hex_to_rgb,
)
from scipy.spatial import KDTree

import cv2
import numpy as np
import os 

def convert_rgb_to_names(rgb_tuple):
    

    css3_db = CSS3_HEX_TO_NAMES
    names = []
    rgb_values = []
    for color_hex, color_name in css3_db.items():
        names.append(color_name)
        rgb_values.append(hex_to_rgb(color_hex))
    
    kdt_db = KDTree(rgb_values)
    distance, index = kdt_db.query(rgb_tuple)

    return f' {names[index]}'

# def get_nbr_image (chemin) : 
#     files = os.listdir(chemin)
#     num_files = len(files)/2
    
#     return num_files



def get_couleur(chemin_img, num_espece, duree_entre_frames, seuil_couleur) : 

    num_files = len(chemin_img) 
    weights = 'models/model_final.pth'
    # get config
    predictor, cfg = init_config(weights, SCORE_THRESH_TEST = 0.8)

    moy_tot_couleur = []                        #vecteur de la couleur moyenne pour toute le images 
    for i in range(num_files) : 
        #chargement de l'image :
        image = cv2.imread(chemin_img[i])
        #recuperer tous les pixel de l'espèce souhaitée 
        outputs, im_seg1 = inference(predictor, cfg,  image.copy())
        mask_seg = outputs["instances"].pred_masks.cpu().numpy()
        intensite = []
        # for mask in mask_seg :
        indices = np.column_stack(np.where(mask_seg[num_espece] == True))       
        # Get the intensity values of the pixels in the mask
        intensite = [image[y, x] for y, x in indices]  #recuperer les intensité des pixel d'interet 
        #vecteur de la couleur moyenne de chaque image
        moy_couleur = np.empty((np.shape(intensite)[1],1))  
        moy_tot_couleur.append(moy_couleur)

    itens_0 = np.take(moy_tot_couleur, 0, axis=1) 
    itens_1 = np.take(moy_tot_couleur, 1, axis=1) 
    itens_2 = np.take(moy_tot_couleur, 2, axis=1) 
    #recuperer la premiere couleur de l'espece :
    rgb = tuple(np.concatenate([itens_0[0],itens_1[0],itens_2[0]]))
    color = convert_rgb_to_names(rgb)
    print (f"la couleur de l'espece au debut est {color}")

    for i in range (0,num_files-1) :
        # seuil de 10% pour chaque image                              
        seuil_0 = (itens_0[i]*seuil_couleur)/100
        seuil_1 = (itens_1[i]*seuil_couleur)/100
        seuil_2 = (itens_2[i]*seuil_couleur)/100
        if (((itens_0[i+1] < seuil_0 + itens_0[i]) and (itens_0[i+1] > seuil_0 - itens_0[i])) or (itens_1[i+1] < seuil_1 + itens_1[i] and itens_1[i+1] > seuil_1 - itens_1[i]) or (itens_2[i+1] < seuil_2 + itens_2[i] and itens_2[i+1] > seuil_2 - itens_2[i]) ) : 
            pass
        else : 
            rgb = tuple(np.concatenate((itens_0[i+1],itens_1[i+1],itens_2[i+1])))
            #conversion en hexadécimal
            color = convert_rgb_to_names(rgb)
            #affichage de la couleur
            print (f"lespece devient {color} à l'instant {i*duree_entre_frames} secondes:") 
            
    color_1 = []
    for i in range (0,num_files) : 
         rgb_1 = tuple(np.concatenate((itens_0[i],itens_1[i],itens_2[i])))
         #conversion en hexadécimal
         color_1.append(convert_rgb_to_names(rgb_1))
         
    return (color_1) 



if __name__ == "__main__":
    path_img = glob('data/outputs/*.jpg')
    get_couleur(path_img, num_espece=0, duree_entre_frames=2, seuil_couleur = 10)




