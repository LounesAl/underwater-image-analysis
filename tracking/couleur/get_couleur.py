#code pour suivre la couleur d'un objet : 
#bibliotheque : 

from utils.segmentation import *
from webcolors import rgb_to_hex
from webcolors import (
    CSS3_HEX_TO_NAMES,
    hex_to_rgb,
)

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
    files = os.listdir(path)
    num_files = len(files)/2
    
    return num_files


def get_couleur (nbr_image) : 
    #d'abord recuperer toutes les images : 
    #nbr_image = 10 #(recuperer le nbr d'image)

    moy_tot_couleur = []                        #vecteur de la couleur moyenne pour toute le images 
    for i in range(nbr_image) : 
        #chargement de l'image :
        image = cv2.imread("image"+i+".jpg")
        #recuperer tous les pixel de l'espèce souhaitée 
        cnt = np.load('Coords'+i+'.npy')
        taille_cnt = len(cnt) 
        pixel = []
        intensite = []

        for j in range (taille_cnt) : 
            pixel = cnt[i]                              #recuperer les pixel 
            intensite.append(image[pixel[0],pixel[1]])  #recuperer les intensité des pixel d'interet 

        moy_couleur = np.empty((3,1))                   #vecteur de la couleur moyenne de chaque image 
        #print(len(moy_couleur))
        for j in range(3):
        #recuperer les colonnes de chaque intensité (R,G,B)
        exec("colonne_" + str(i) + " = np.take(intensite, i, axis=1)")
        #calculer leurs moyennes
        exec("moy_couleur [i]  = np.mean(colonne_"+str(i)+")")
        
        moy_tot_couleur.append(moy_couleur)

    #une fois la matrice de couleur remplie nous verrons si on a une difference entre les images 
    for i in range(3):
    #recuperer chaque colonne des intensités moyennes 
    exec("itens" + str(i) + " = np.take(moy_tot_couleur, i, axis=1)")

    for i in range (0,nbr_image-1) : 
        #seuil de 10% pour chaque image 
        s = 10
        seuil_0 = (itens0[i]*s)/100
        seuil_1 = (itens1[i]*s)/100
        seuil_2 = (itens2[i]*s)/100

        if ((itens0[i+1] < seuil_0 + itens0[i] and itens0[i+1] > seuil_0 - itens0[i]) or (itens1[i+1] < seuil_1 + itens1[i] and itens1[i+1] > seuil_1 - itens1[i]) or (itens2[i+1] < seuil_2 + itens2[i] and itens2[i+1] > seuil_2 - itens2[i]) ) : 
            print("pas de changement de couleur")

        else : 

            rgb = (itens0[i+1],itens1[i+1],itens2[i+1])

            #conversion en hexadécimal
            color = convert_rgb_to_names(rgb)

            #affichage de la couleur
            print("l'espece change de couleur vers :" color) 


