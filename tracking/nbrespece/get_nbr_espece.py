import numpy as np
from segmentation import *
from glob import glob

def get_nbr_espece (tensor, classe_dict , chemin_npy , chemin_img) : 

    taille_tensor = len(tensor)                             #recuperer la taille du vecteur qui correspond au nbr d'espece detectée
    nbr_classe = len(classe_dict)                           #recuperer le nbr de classe à detecter 

    num_files = len(chemin_npy)
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

        for classe, nom in classe_dict.items():
            if (int(classe) != 0 ) : 
                print("l'image " , j , ":")
                if (nbr_classe_[int(classe)] != 0) : 
                    print ("l'espece" , nom , "a ete detectee" ,  nbr_classe_[int(classe)] , "fois")
                else : 
                    print ("l'espece" , nom , "n'a pas ete detectee")
        
    return nbr_classe_ 

if __name__ == "__main__":

    tensor = [0, 2, 3, 3 , 3]
    class_dict = {
            "0" : "PFE",
            "1" : "Actinia fermee",
            "2" : "Actinia ouverte",
            "3" : "Gibbula"}

    nbr = get_nbr_espece (tensor, class_dict)
    #print(nbr)






