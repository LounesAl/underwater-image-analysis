import cv2
from utils.segmentation import *
from utils.calibration import *
import random
import imutils

colors = [ (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255),
           (255, 128, 0), (255, 0, 128), (0, 255, 128), (128, 255, 0), (0, 128, 255), (128, 0, 255),
           (128, 128, 0), (0, 128, 128), (128, 0, 128) ]

weights='./models/model_final.pth'
predictor, cfg = init_config(str(weights), SCORE_THRESH_TEST = 0.5)
mtx1, mtx2, R, T = load_calibration('./settings/camera_parameters/stereo_params.pkl')
P1, P2 = get_projection_matrix(mtx1, mtx2, R, T)


# charger l'image
img_cam1 = cv2.imread("./data/imm/GOPR1463.JPG")
img_cam2 = cv2.imread("./data/imm/GOPR1463.JPG")

img_cam1 = imutils.resize(img_cam1, width=640, height=640)
img_cam2 = imutils.resize(img_cam2, width=640, height=640)

output_cam1, _ = inference(predictor, cfg, img_cam1)
output_cam2, _ = inference(predictor, cfg, img_cam2)

classes_cam1 = output_cam1["instances"].pred_classes
classes_cam2 = output_cam2["instances"].pred_classes

masks_cam1 = output_cam1["instances"].pred_masks.cpu().numpy().astype(np.uint8)
masks_cam2 = output_cam2["instances"].pred_masks.cpu().numpy().astype(np.uint8)


def center_of_gravity_distance(index_mask):
    _, mask = index_mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    moments = cv2.moments(cnt)
    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])
    return np.subtract((np.linalg.norm(((cx, cy)))), ((np.linalg.norm((cx, mask.shape[1]-cy)))))
    
sorted_args1 = [index for index, _ in sorted(enumerate(masks_cam1), key=center_of_gravity_distance, reverse=True)]
sorted_args2 = [index for index, _ in sorted(enumerate(masks_cam2), key=center_of_gravity_distance, reverse=True)]
    

for arg1, arg2 in zip(sorted_args1, sorted_args2):
        
    contours_cam1, _ = cv2.findContours(masks_cam1[arg1], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_cam2, _ = cv2.findContours(masks_cam2[arg2], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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
        
        p3ds = []
        temp_dist = []
        for i, sub_cam1, sub_cam2, color in zip(range(len(subset_cam1)), subset_cam1, subset_cam2, colors):
            
            # color = random.choice(colors)
            cv2.line(img_cam1, (cx_cam1, cy_cam1), tuple(sub_cam1[0]), color, 1)
            cv2.line(img_cam2, (cx_cam2, cy_cam2), tuple(sub_cam2[0]), color, 1)
            
            # calculer la distance le centre et les autres points autour
            u1 = (cx_cam1, cy_cam1) if i == 0 else tuple(sub_cam1[0])
            u2 = (cx_cam2, cy_cam2) if i == 0 else tuple(sub_cam2[0])
            p3d = DLT(P1, P2, u1, u2)
            p3ds.append(p3d)
            if i!= 0 : temp_dist.append(np.linalg.norm(p3ds[-1] - p3ds[0]))
            
        height = np.max(temp_dist)*2
        width = np.min(temp_dist)*2
        # distances.append([longueur, largeur])
        draw_text(img=img_cam1, text="{} W : {:.1f} cm L : {:.1f} cm".format(classes_cam1[arg1], width, height), pos=tuple(sub_cam1[0]), 
                  font_scale=0.5, font_thickness=1, text_color=(255, 255, 255), text_color_bg=(0, 0, 0))
    
    cv2.imshow("mask_cam1", img_cam1) # *255
    cv2.imshow("mask_cam2", img_cam2)
    cv2.waitKey(0)

cv2.destroyAllWindows()