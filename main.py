




"""Dans un nouveau fichier : 
1) Acquisition (d'image, video ou GIF) avec les deux cameras en stereo:
            -> Lecture des images en temps rell avec opencv ou autres.
            -> Distance entre les deux cameras (paramètre à optimisé).
            -> Espèce sous marine (sous aquarium) ex: GIBBULA, TIMHARINES.
          
2) Récupérer les coords  pixels de longueur et largeur de chaque espèces pour les deux caméras: 
            -> Inférence des espèces avec notre modèle detectron 2 avec les deux caméras stereo
            -> Fonctions supplémentaires pour l'extraction des coords

3) Conversion 2D -> 3D:"""

import glob
import PIL



images_c1 = glob.glob('chemin_du_dossier/*.{jpg,jpeg,png,gif}', recursive=True)
images_c2 = glob.glob('chemin_du_dossier/*.{jpg,jpeg,png,gif}', recursive=True)

for img_c1, img_c2 in zip(images_c1, images_c2):

    # Inferred with the images of each camera




















