from get_couleujr import *
from get_nbr_espece import * 
import matplotlib.pyplot as plt

def affichage_couleur (path_npy , path_img) : 

    color = get_couleur (path_npy , path_img)
    nbr_image = len(path_img)

    fig = plt.figure()
    #ax = fig.add_subplot(111)
    plt.plot(len(path_img) , color)
    #plt.axis([])
    plt.xlabel('Temps')
    plt.ylabel('Couleur')
    plt.title("Suivi de la couleur")

    plt.show() 



if __name__ == "__main__":
    
    path_img = glob('data/outputs/*.jpg')
    path_npy = glob('data/outputs/*.npy')
    affichage_couleur (path_npy , path_img)




