import cv2
from utils.segmentation import *
from utils.segmentation import inference
import random
import imutils

colors = [ (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255),
           (255, 128, 0), (255, 0, 128), (0, 255, 128), (128, 255, 0), (0, 128, 255), (128, 0, 255),
           (128, 128, 0), (0, 128, 128), (128, 0, 128) ]

weights='./models/model_final.pth'
predictor, cfg = init_config(str(weights), SCORE_THRESH_TEST = 0.5)



# charger l'image
img_cam1 = cv2.imread("./data/imgg1/GOPR1483.JPG")
img_cam2 = cv2.imread("./data/imgg2/GOPR1501.JPG")

img_cam1 = imutils.resize(img_cam1, width=640, height=640)
img_cam2 = imutils.resize(img_cam2, width=640, height=640)


output_cam1, _ = inference(predictor, cfg, img_cam1)
output_cam2, _ = inference(predictor, cfg, img_cam2)

masks_cam1 = output_cam1["instances"].pred_masks.cpu().numpy().astype(np.uint8)
masks_cam2 = output_cam2["instances"].pred_masks.cpu().numpy().astype(np.uint8)

classes_cam1 = output_cam1["instances"].pred_classes
classes_cam2 = output_cam2["instances"].pred_classes

# def center_of_gravity_distance(mask):
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     cnt = contours[0]
#     moments = cv2.moments(cnt)
#     cx = int(moments["m10"] / moments["m00"])
#     cy = int(moments["m01"] / moments["m00"])
#     return np.subtract((np.linalg.norm(((cx, cy)))), ((np.linalg.norm((cx, mask.shape[1]-cy)))))
#     # return np.linalg.norm(np.linalg.norm((cx, cy)), np.linalg.norm((mask.shape[0] // 2, mask.shape[1]//2)))
    
# masks_cam1 = [index for index, _ in sorted(enumerate(masks_cam1), key=center_of_gravity_distance, reverse=True)]
# masks_cam2 = [index for index, _ in sorted(enumerate(masks_cam2), key=center_of_gravity_distance, reverse=True)]

def center_of_gravity_distance(mask_index):
    mask = cv2.UMat(masks_cam1[mask_index])
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(contours, key=cv2.contourArea)
    M = cv2.moments(c)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return cX, cY

masks_cam1 = np.array(masks_cam1)
sorted_indices = np.argsort(np.array([center_of_gravity_distance(i) for i in range(len(masks_cam1))]), axis=0)

print(sorted_indices)



for mask_cam1, mask_cam2, class_cam1, classe_cam2 in zip(masks_cam1, masks_cam2, classes_cam1, classes_cam2):
        
    contours_cam1, _ = cv2.findContours(mask_cam1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_cam2, _ = cv2.findContours(mask_cam2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    assert len(contours_cam1) == 1 and len(contours_cam1) == 1, \
    f'we are supposed to retrieve a single contour that represents a species in what we have {len(contours_cam1) and len(contours_cam2)}  contours.'
    
    # pour chaque contour
    cnt_cam1, cnt_cam2 = contours_cam1[0], contours_cam2[0]
    
    # trouver les moments de l'image
    moments_cam1, moments_cam2 = cv2.moments(cnt_cam1), cv2.moments(cnt_cam2)
    if moments_cam1["m00"] != 0 and moments_cam2["m00"] != 0:
        # calculer le centre de gravité
        cx_cam1 = int(moments_cam1["m10"] / moments_cam1["m00"])
        cy_cam1 = int(moments_cam1["m01"] / moments_cam1["m00"])
        cx_cam2 = int(moments_cam2["m10"] / moments_cam2["m00"])
        cy_cam2 = int(moments_cam2["m01"] / moments_cam2["m00"])
        
        # Calcule le nombre de points dans le contour
        n_points_cam1, n_points_cam2 = cnt_cam1.shape[0], cnt_cam2.shape[0]
        
        # Crée un tableau d'indices tous les 20 pas
        spaced_indices_cam1 = np.round(np.linspace(0, len(cnt_cam1) - 1, 15))[:-2].astype(int)
        spaced_indices_cam2 = np.round(np.linspace(0, len(cnt_cam2) - 1, 15))[:-2].astype(int)
        
        # Utilise les indices pour extraire les points du contour
        subset_cam1 = cnt_cam1[spaced_indices_cam1]
        subset_cam2 = cnt_cam2[spaced_indices_cam2]
        
        # dessiner un cercle autour du centre de gravité
        cv2.circle(img_cam1, (cx_cam1, cy_cam1), 5, (255, 0, 0), -1)
        cv2.circle(img_cam2, (cx_cam2, cy_cam2), 5, (255, 0, 0), -1)
        cv2.drawContours(img_cam1, [cnt_cam1], -1, (0, 255, 0), 1)
        cv2.drawContours(img_cam2, [cnt_cam2], -1, (0, 255, 0), 1)
        
        for sub_cam1, sub_cam2, color in zip(subset_cam1, subset_cam2, colors):
            
            # color = random.choice(colors)
            cv2.line(img_cam1, (cx_cam1, cy_cam1), tuple(sub_cam1[0]), color, 1)
            cv2.line(img_cam2, (cx_cam2, cy_cam2), tuple(sub_cam2[0]), color, 1)
        
    
    cv2.imshow("mask_cam1", img_cam1) # *255
    cv2.imshow("mask_cam2", img_cam2)
    cv2.waitKey(0)

cv2.destroyAllWindows()


# Trouver les extrémités du contour
# leftmost = tuple(c[c[:,:,0].argmin()][0])
# rightmost = tuple(c[c[:,:,0].argmax()][0])
# topmost = tuple(c[c[:,:,1].argmin()][0])
# bottommost = tuple(c[c[:,:,1].argmax()][0])
            

    




