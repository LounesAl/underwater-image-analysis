
#from glob import glob

import numpy as np
#import os 


## voir les classes predites
    ## le resultat est de la forme : tensor([0, 1, 1, 2, 3, 3]), cela veut dire :
    ## une espces PFE, deux actinia fermées, une ouverte et deux gibbula

#class_dict = {
#            "0" : "PFE",
#            "1" : "Actinia fermee",
#            "2" : "Actinia ouverte",
#            "3" : "Gibbula"
#        }

def get_nbr_espece (tensor, nbr_classe) : 

    taille_tensor = len(tensor)

    nbr_classe_ =  np.empty((nbr_classe,1))

    for i in range (nbr_classe) : 
        #exec("nbr_classe_" + str(i) + " = 0")
        nbr_classe_[i] = 0

    print(nbr_classe_[2])
    print("je suis la")

    for i in range (taille_tensor) : 
        pass



    #return nbr_espece 


if __name__ == "__main__":

    tensor = [0, 1, 1, 2, 3, 3]
    nbr_classe = 3

    get_nbr_espece (tensor, nbr_classe)







