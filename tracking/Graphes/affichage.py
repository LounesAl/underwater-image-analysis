from get_couleujr import *
from get_nbr_espece import * 
import matplotlib.pyplot as plt




def affichage_couleur (path_npy , path_img) : 

    color = get_couleur (path_npy , path_img)
    nbr_image = len(path_img)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(color, len(path_img))

































if __name__ == "__main__":




