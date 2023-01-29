import cv2
from utils.segmentation import *
from utils.segmentation import inference
import random

colors = [ (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255),
           (255, 128, 0), (255, 0, 128), (0, 255, 128), (128, 255, 0), (0, 128, 255), (128, 0, 255),
           (128, 128, 0), (0, 128, 128), (128, 0, 128) ]

# charger l'image
img = cv2.imread("/home/bra/Downloads/underwater-image-analysis/data/imgs_c0/gibbula_07_16-1137-_jpg.rf.14010270c5faa4b12daee4276d06fbcf.jpg")
weights='./models/model_final.pth'
predictor, cfg = init_config(str(weights), SCORE_THRESH_TEST = 0.8)
output, _ = inference(predictor, cfg, img)
classes = output["instances"].pred_masks.cpu().numpy()
uvs1, seg1, boxes1 = get_segment_points(output, img)


for mask in classes:
    mask = mask.astype(np.uint8)    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    assert len(contours) == 1, f'we are supposed to retrieve a single contour that represents a species in what we have {len(contours)} contours.'
    
    # pour chaque contour
    c = contours[0]
    # trouver les moments de l'image
    moments = cv2.moments(c)
    if moments["m00"] != 0:
        # calculer le centre de gravité
        cX = int(moments["m10"] / moments["m00"])
        cY = int(moments["m01"] / moments["m00"])
        
        # Calcule le nombre de points dans le contour
        n_points = c.shape[0]
        # Crée un tableau d'indices tous les 20 pas
        spaced_indices = np.round(np.linspace(0, len(c) - 1, 8)).astype(int)
        # Utilise les indices pour extraire les points du contour
        subset = c[spaced_indices]
        
        # dessiner un cercle autour du centre de gravité
        # cv2.circle(img, (cX, cY), 5, (255, 0, 0), -1)
        # cv2.drawContours(img, [c], -1, (0, 255, 0), 2)
        
        for sub, color in zip(subset, colors):
            
            # color = random.choice(colors)
            cv2.line(img, (cX, cY), tuple(sub[0]), color, 1)
        
        

        

# Trouver les extrémités du contour
# leftmost = tuple(c[c[:,:,0].argmin()][0])
# rightmost = tuple(c[c[:,:,0].argmax()][0])
# topmost = tuple(c[c[:,:,1].argmin()][0])
# bottommost = tuple(c[c[:,:,1].argmax()][0])
            

    
    
    cv2.imshow("mask", img) # *255
    cv2.waitKey(0)

cv2.destroyAllWindows()




# # afficher l'image
# cv2.imshow("image", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



