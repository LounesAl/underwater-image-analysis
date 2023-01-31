import numpy as np
from segmentation import *
from glob import glob

def get_nbr_espece (chemin_npy, classe_dict) : 

    nbr_classe = len(classe_dict)                           #recuperer le nbr de classe à detecter 
    num_files = len(chemin_npy)                             #recuperer le nbr d'image à traiter 

    #parcourir toutes les images 
    classe_tot = []
    for j in range (num_files) : 
        #recuperer les npy 
        outputs = np.load(chemin_npy[j], allow_pickle=True)
        #recupere les tensors 
        tensor = outputs["instances"].pred_classes

        nbr_classe_ =  np.zeros(nbr_classe, dtype=int)          #creer le vecteur de repetition 

        for elem in tensor :
            for i in range (0 , nbr_classe) : 
                if (elem == i and i != 0) : 
                    nbr_classe_[i] = nbr_classe_[i] + 1         #incrementer la 

        classe_tot.append( nbr_classe_ )

        for classe, nom in classe_dict.items():
            if (int(classe) != 0 ) : 
                print("l'image " , j , ":")
                if (nbr_classe_[int(classe)] != 0) : 
                    print ("l'espece" , nom , "a ete detectee" ,  nbr_classe_[int(classe)] , "fois")
                else : 
                    print ("l'espece" , nom , "n'a pas ete detectee")
        
    
    vecteur = classe_tot.reshape(num_files, classe_tot)
    vecteur = vecteur.transpose()
    moy = []
    for ligne in vecteur:
        moy.append(np.mean(ligne))  

    for classe, nom in classe_dict.items():
            if (int(classe) != 0 ) : 
                 ("l'espece" , nom , "a ete detectee en moyenne" ,  moy[int(classe)] , "fois")
    
    return classe_tot 





if __name__ == "__main__":

    path_npy = glob('data/outputs/*.npy')

    class_dict = {
            "0" : "PFE",
            "1" : "Actinia fermee",
            "2" : "Actinia ouverte",
            "3" : "Gibbula"}

    nbr = get_nbr_espece (path_npy, class_dict)













