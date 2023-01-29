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

    nbr_classe_ =  np.zeros(nbr_classe, dtype=int)
    print(nbr_classe_)

    for elem in tensor :
        #print("ele",elem) 
        for i in range (0 , nbr_classe) : 
        #for classe, nom in classe_dict.items():
            #print ("cls" , classe , "nom" , nom)
            if (elem == i and i != 0) : 
                #print ("lkd")
                #nbr_classe_[int(classe)] = nbr_classe_[int(classe)] + 1
                nbr_classe_[i] = nbr_classe_[i] + 1
                print(i , "=" , nbr_classe_[i]) 


    for classe, nom in classe_dict.items():
        if (int(classe) != 0 ) : 
            print ("l'espece" , nom , "a ete detecte" ,  nbr_classe_[int(classe)] , "fois")
    
    return nbr_classe_ 




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
    nbr = get_nbr_espece (tensor, class_dict)

    print(nbr)






