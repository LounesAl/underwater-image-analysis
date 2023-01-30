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
    
    # a dictionary of all the hex and their respective names in css3
    css3_db = CSS3_HEX_TO_NAMES
    names = []
    rgb_values = []
    for color_hex, color_name in css3_db.items():
        names.append(color_name)
        rgb_values.append(hex_to_rgb(color_hex))
    
    kdt_db = KDTree(rgb_values)
    distance, index = kdt_db.query(rgb_tuple)

    return f' {names[index]}'

def get_nbr_image (chemin) : 
   #path = 'chemin/vers/dossier'
    files = os.listdir(chemin)
    num_files = len(files)/2
    
    return num_files

#C:\Users\Amel\Documents\GitHub\underwater-image-analysis\data\outputs\output_0.jpg

def get_couleur (chemin_npy , chemin_img) : 

    num_files = len(chemin_img)
    moy_tot_couleur = []                        #vecteur de la couleur moyenne pour toute le images 
    for i in range(num_files) : 
        #chargement de l'image :
        image = cv2.imread(chemin_img[i])
        #recuperer tous les pixel de l'espèce souhaitée 

        outputs = np.load(chemin_npy[i], allow_pickle=True)

        print(outputs[0])
        # a tester et voir 
        mask_seg = outputs["instances"].pred_masks.cpu().numpy()
        cnt = get_segmentation(mask_seg)

        taille_cnt = len(cnt) 
        intensite = []

        for j in range (taille_cnt) :        
            intensite.append(image[cnt[j][0],cnt[j][1]])  #recuperer les intensité des pixel d'interet 
    
        moy_couleur = np.empty((np.shape(intensite)[1],1))                   #vecteur de la couleur moyenne de chaque image 

        for j in range(np.shape(intensite)[1]):
            #recuperer les colonnes de chaque intensité (R,G,B)
            exec("colonne_" + str(j) + " = np.take(intensite, j, axis=0)")
            #calculer leurs moyennes
            exec("moy_couleur [j]  = np.mean(colonne_"+str(j)+")")
        
        moy_tot_couleur.append(moy_couleur)

    #une fois la matrice de couleur remplie nous verrons si on a une difference entre les images 
    #for i in range(len(moy_tot_couleur)):
        #recuperer chaque colonne des intensités moyennes 
        #exec("itens_" + str(i) + " = moy_tot_couleur[i]")
        #exec("colonne_" + str(i) + " = tableau[i]")
        #print (itens_1)

    itens_0 = np.take(moy_tot_couleur, 0, axis=1) 
    itens_1 = np.take(moy_tot_couleur, 1, axis=1) 
    itens_2 = np.take(moy_tot_couleur, 2, axis=1) 

    #recuperer la premiere couleur de l'espece :
    rgb = tuple(np.concatenate([itens_0[0],itens_1[0],itens_2[0]]))
    print(rgb)
    color = convert_rgb_to_names(rgb)
    print ("la couleur de l'espece est" , color)

    for i in range (0,num_files-1) : 
        

        s = 10                              #seuil de 10% pour chaque image 
        seuil_0 = (itens_0[i]*s)/100
        seuil_1 = (itens_1[i]*s)/100
        seuil_2 = (itens_2[i]*s)/100

        if (((itens_0[i+1] < seuil_0 + itens_0[i]) and (itens_0[i+1] > seuil_0 - itens_0[i])) or (itens_1[i+1] < seuil_1 + itens_1[i] and itens_1[i+1] > seuil_1 - itens_1[i]) or (itens_2[i+1] < seuil_2 + itens_2[i] and itens_2[i+1] > seuil_2 - itens_2[i]) ) : 
            print("pas de changement de couleur")

        else : 

            rgb = tuple(np.concatenate((itens_0[i+1],itens_1[i+1],itens_2[i+1])))
            #conversion en hexadécimal
            color = convert_rgb_to_names(rgb)
            #affichage de la couleur
            print ("lespece change de couleur vers :" , color) 

        
    color_1 = []

    for i in range (0,num_files) : 

        rgb_1 = tuple(np.concatenate((itens_0[i],itens_1[i],itens_2[i])))
        #conversion en hexadécimal
        color_1.append(convert_rgb_to_names(rgb_1))

    return (color_1) 

        

if __name__ == "__main__":
    path_img = glob('data/outputs/*.jpg')
    path_npy = glob('data/outputs/*.npy')
    num_img = len(path_img)
    color = get_couleur (path_npy , path_img)