from get_couleujr import *
from get_nbr_espece import * 
import matplotlib.pyplot as plt

def affichage_nbr (path_npy, class_dict) : 


    nbr = get_nbr_espece (path_npy, class_dict)

    fig = plt.figure()
    #ax = fig.add_subplot(111)
    plt.plot(len(path_img) , nbr)
    #plt.axis([])
    plt.xlabel('Temps')
    plt.ylabel('Couleur')
    plt.title("Suivi de la couleur")

    plt.show() 

if __name__ == "__main__":
    
    path_npy = glob('data/outputs/*.npy')

    class_dict = {
            "0" : "PFE",
            "1" : "Actinia fermee",
            "2" : "Actinia ouverte",
            "3" : "Gibbula"}

    affichage_nbr (path_npy, class_dict) 








