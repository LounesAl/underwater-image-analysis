#code pour suivre la couleur d'un objet : 
#bibliotheque : 

from utils.segmentation import *
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

def extract_images(video_path, output_folder, n_seconds):
    # Ouvrir la vidéo
    video = cv2.VideoCapture(video_path)
    fps = int(video.get(cv2.CAP_PROP_FPS))
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = fps * n_seconds
    
    # Vérifier si le dossier de sortie existe, sinon le créer
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    #else :
        
    #    for the_file in os.listdir(output_folder):
    #       file_path = os.path.join(output_folder, the_file)
    #        try:
    #            if os.path.isfile(file_path):
    #                os.unlink(file_path)
    #            elif os.path.isdir(file_path):
    #                shutil.rmtree(file_path)
    #        except Exception as e:
    #            print(e)     
    
    # Compter les images extraites
    count = 0
    #i = 0
    # Boucle sur chaque frame de la vidéo
    path_img = []
    while True:
        ret, frame = video.read()
        if ret == False:
            break
            
        if count % frame_interval == 0:
            image_path = os.path.join(output_folder, f"frame_{count}.jpg")
            path_img.append(image_path)
            cv2.imwrite(image_path, frame)
            #print(image_path)
            
        count += 1

    #for i, path in enumerate(path_img):
    #    path_img[i] = path.replace("\\\\", "\\")
        
    #print(fixed_paths)
        #i +=1
    #paths = []
    #for filename in os.listdir(output_folder):
    #    if filename.endswith(".jpg"):
    #        paths.append(os.path.join(output_folder, filename))

    # Fermer la vidéo
    video.release()
    return path_img


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




def track(path_video, N, output_folder, num_espece, self, seuil_couleur = 10) : 
    
    self.progress_bar.setValue(1)
    
    self.text_edit.append("Initialisation de l'algorithme du tracking")
    self.text_edit.append("------------------------------------------")
    self.text_edit.append(f"Traitement de la video {path_video} en cours ...")
    
    chemin_img = extract_images(path_video, output_folder, N)

    num_files = len(chemin_img) 

    weights = 'models/model_final.pth'
    # get config
    self.progress_bar.setValue(5)
    predictor, cfg = init_config(weights, SCORE_THRESH_TEST = 0.8)

    moy_tot_couleur = []                        #vecteur de la couleur moyenne pour toute le images 

    
    class_dict = {
        "0" : "PFE",
        "1" : "Actinia fermee",
        "2" : "Actinia ouverte",
        "3" : "Gibbula"}


    nbr_classe = 4#len(class_dict)                           #recuperer le nbr de classe à detecter 
    #num_files_npy = len(chemin_npy)                             #recuperer le nbr d'image à traiter 
    classe_tot = []
    
    self.progress_bar.setValue(10)
    
    for i in range(num_files) : 
        if (i*100/num_files) > 10 and (i*100/num_files) < 80:
            self.progress_bar.setValue(i*100/num_files)
        #chargement de l'image :
        image = cv2.imread(chemin_img[i])
        #recuperer tous les pixel de l'espèce souhaitée 
        outputs, im_seg1 = inference(predictor, cfg,  image.copy())
        mask_seg = outputs["instances"].pred_masks.cpu().numpy()
        ###################
        #recperer les tensors des classes 
        tensor = outputs["instances"].pred_classes
        ################

        #récupération des intensité
        intensite = []
        # for mask in mask_seg :
        indices = np.column_stack(np.where(mask_seg[num_espece] == True))       
        # Get the intensity values of the pixels in the mask
        intensite = [image[y, x] for y, x in indices]  #recuperer les intensité des pixel d'interet 
        #vecteur de la couleur moyenne de chaque image
        moy_couleur = np.empty((np.shape(intensite)[1],1))  
        moy_tot_couleur.append(moy_couleur)

        ############# 

        #nbr d'espece : 


        nbr_classe_ =  np.zeros(nbr_classe, dtype=int)          #creer le vecteur de repetition 

        for elem in tensor :
            for i in range (0 , nbr_classe) : 
                if (elem == i and i != 0) : 
                    nbr_classe_[i] = nbr_classe_[i] + 1         #incrementer la 

        classe_tot.append( nbr_classe_ )

        # for classe, nom in class_dict.items():
        #     if (int(classe) != 0 ) : 
        #         # print("l'image " , j , ":")
        #         if (nbr_classe_[int(classe)] != 0) : 
        #             self.text_edit.append(f"l'espece {nom} a ete detectee {nbr_classe_[int(classe)]} fois")
        #             print(f"l'espece {nom} a ete detectee {nbr_classe_[int(classe)]} fois")
        #         else : 
        #             self.text_edit.append(f"l'espece {nom} n'a pas ete detectee")
        
        ################
    ##############
    #couleur : 
    itens_0 = np.take(moy_tot_couleur, 0, axis=1) 
    itens_1 = np.take(moy_tot_couleur, 1, axis=1) 
    itens_2 = np.take(moy_tot_couleur, 2, axis=1) 
    #recuperer la premiere couleur de l'espece :
    rgb = tuple(np.concatenate([itens_0[0],itens_1[0],itens_2[0]]))
    color = convert_rgb_to_names(rgb)
    self.text_edit.append(f"la couleur de l'espece choisis au debut est {color}")

    # color_tot = convert_rgb_to_names(rgb)
    temps = []
    
    self.progress_bar.setValue(90)

    for i in range (0,num_files-1) :

        # seuil de 10% par defaut pour chaque image                              
        seuil_0 = (itens_0[i]*seuil_couleur)/100
        seuil_1 = (itens_1[i]*seuil_couleur)/100
        seuil_2 = (itens_2[i]*seuil_couleur)/100
        if (((itens_0[i+1] < seuil_0 + itens_0[i]) and (itens_0[i+1] > seuil_0 - itens_0[i])) or (itens_1[i+1] < seuil_1 + itens_1[i] and itens_1[i+1] > seuil_1 - itens_1[i]) or (itens_2[i+1] < seuil_2 + itens_2[i] and itens_2[i+1] > seuil_2 - itens_2[i]) ) : 
            pass
        else : 
            temps.append(i*N)
            rgb = tuple(np.concatenate((itens_0[i+1],itens_1[i+1],itens_2[i+1])))
            #conversion en hexadécimal
            color = convert_rgb_to_names(rgb)
            # color_tot.append(color)
            #affichage de la couleur
            self.text_edit.append(f"lespece devient {color} à l'instant {i*N} secondes:")
            
    self.progress_bar.setValue(95)
    
    #for i in range (0,num_files) : 
    #     rgb_1 = tuple(np.concatenate((itens_0[i],itens_1[i],itens_2[i])))
         #conversion en hexadécimal
    #     color_1.append(convert_rgb_to_names(rgb_1))


    #####################
    # nbr despece : 
    
    # classe_tot = np.array(classe_tot)

    # classes_tot_ = classe_tot.reshape(num_files, classe_tot)
    # classes_tot_ = classes_tot_.transpose()

    moy_classe = []

    for ligne in classe_tot :
        moy_classe.append(np.mean(ligne, axis=0))  

    for classe, nom in class_dict.items():
            if int(classe) : 
                self.text_edit.append(f"l'espece {nom} a ete detectee en moyenne {moy_classe[int(classe)]} fois")
                
    self.progress_bar.setValue(100)
    #return (color_1) 