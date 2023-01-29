import numpy as np


## voir les classes predites
## le resultat est de la forme : tensor([0, 1, 1, 2, 3, 3]), cela veut dire :
## une espces PFE, deux actinia fermées, une ouverte et deux gibbula

#class_dict = {
#            "0" : "PFE",
#            "1" : "Actinia fermee",
#            "2" : "Actinia ouverte",
#            "3" : "Gibbula"
#        }

def get_nbr_espece (tensor, classe_dict) : 

    taille_tensor = len(tensor)                             #recuperer la taille du vecteur qui correspond au nbr d'espece detectée
    nbr_classe = len(classe_dict)


    nbr_classe_ =  np.empty((nbr_classe+1,1))

    for i in range (nbr_classe+1) : 
        #exec("nbr_classe_" + str(i) + " = 0")
        nbr_classe_[i] = 0

    for elem in tensor : 
        #for i in range (1 , nbr_classe+1) : 
        for classe, nom in classe_dict.items():
            if (elem == classe) : 
                nbr_classe_[i] = nbr_classe_[i] + 1 

    #for i in range (1 , nbr_classe+1) : 

    #    print ("l'espece" , classe_dict[i] ) #, "a ete detecte" ,  nbr_classe_[i] , "fois")

    for classe, nom in classe_dict.items():
        if (classe != 0 ) : 
            print ("l'espece" , nom , "a ete detecte" ,  nbr_classe_[int(classe)] , "fois")




    #print(nbr_classe_[3])   #enlever le print apres et uriliser un dictionnaire cad retourner le dictionnaire 
    #chaque classe aura son nom plus le nbr de fois qu'elle a été detecté 

    #return nbr_espece 

#ce qui reste a faire associer les classes aux nbr de fois qu'on l'a detecter 
#penser a comment faire un matplotlib 

if __name__ == "__main__":

    tensor = [0, 1, 1, 2, 3, 3 , 3]
    nbr_classe = 3
    class_dict = {
            "0" : "PFE",
            "1" : "Actinia fermee",
            "2" : "Actinia ouverte",
            "3" : "Gibbula"}
    #for cle, valeur in dictionnaire.items():
    #    print (class_dict['i'])
    get_nbr_espece (tensor, class_dict)







