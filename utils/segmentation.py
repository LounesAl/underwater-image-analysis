import cv2
import numpy as np
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2 import model_zoo
import torch
from utils.calibration import *
import imutils

def visualiser(outputs, cfg, im):
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TEST[0]), scale=1)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    return v.get_image()[:, :, ::-1]
    
def init_config(path_model, SCORE_THRESH_TEST = 0.8):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TEST = (f"PFE_valid",)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4

    # Additional Info when using cuda
    if not torch.cuda.is_available():
        cfg.MODEL.DEVICE = 'cpu'
    
    cfg.MODEL.WEIGHTS = path_model # Let training initialize from model zoo
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = SCORE_THRESH_TEST
    predictor = DefaultPredictor(cfg)
    return predictor, cfg

def inference(predictor, cfg,  im):
    outputs = predictor(im)
    im_seg = visualiser(outputs, cfg, im)
    
    return outputs, im_seg

def get_segmentation(mask_array):
    num_instances = mask_array.shape[0]
    coordinates = []
    for i in range(num_instances):
        indices = np.column_stack(np.where(mask_array[i] == True))
        coordinates.append([(y, x) for x, y in indices])
    return coordinates

def mid_point(box):
    sorted_box = sorted(box, key=lambda x: x[1])
    highest = sorted_box[-1]
    second = sorted_box[-2]
    mid = ((highest[0] + second[0]) / 2, (highest[1] + second[1]) / 2)
    return mid

def draw_text(img, text,
          font=cv2.FONT_HERSHEY_PLAIN,
          pos=(0, 0),
          font_scale=3,
          font_thickness=2,
          text_color=(0, 255, 0),
          text_color_bg=(0, 0, 0)
          ):

    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, pos, (x + text_w, y + text_h), text_color_bg, -1)
    cv2.putText(img, text, (x, int(y + text_h + font_scale - 1)), font, font_scale, text_color, font_thickness)

    return text_size

def extract_caractristics(mask, pts, im):
    img = im.copy()
    
    pts = [tuple(map(int, coord)) for coord in pts]
    (x1, y1), (x2, y2), (x3, y3), (x4, y4) = pts 
    cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 1)
    l1 = extract_desired_color_coordinates(img, (255, 0, 0))
    cv2.line(img, (x3, y3), (x4, y4), (0, 255, 0), 1)
    l2 = extract_desired_color_coordinates(img, (0, 255, 0))
    
    contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    assert len(contours) == 1, f'we are supposed to retrieve a single contour that represents a species in what we have {len(contours)} contours.'
    cv2.drawContours(img, [contours[0]], 0, (0, 0, 255), 1)
    c = extract_desired_color_coordinates(img, (0, 0, 255))
    common_elements1 = l1[(l1[:, None] == c).all(-1).any(1)]
    common_elements2 = l2[(l2[:, None] == c).all(-1).any(1)]
    print('common_elements1 : ', common_elements1)
    print('common_elements2 : ', common_elements2)
    (x1, y1), (x2, y2) = common_elements1
    (x3, y3), (x4, y4) = common_elements2
    return [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]


def dist_on_img(segment_points, boxes, im, distances, classes, class_dict, copy=True):
    # Creer une sauvegarde
    img = im.copy() if copy else im
    
    # for i,box in enumerate(boxes):
    #     class_name = class_dict[str(classes[i].item())]
    #     cv2.drawContours(img, [np.int0(box)], -1, (0, 0, 255), 2)
    #     # mid = mid_point(box)
    #     highest = min(box, key=lambda point: point[1])
    #     draw_text(img=img, text=class_name, pos=(int(highest[0]) + 10, int(highest[1]) - 10), 
    #               font_scale=2, font_thickness=2, text_color=(255, 255, 255), text_color_bg=(0, 0, 0))
    
    # Initialiser un tableau de tableau vide pour stocker les points de segmentation de chaque espece
    for j, segment_point in enumerate(segment_points):
        # dessiner les lines de longuer et de largeur
        for i, n in zip(range(0, len(segment_point), 2), range(len(distances[j]))):
            start = (int(segment_point[i][0]), int(segment_point[i][1]))
            end = (int(segment_point[i+1][0]), int(segment_point[i+1][1]))
            x = start[0] + (end[0] - start[0]) * 0.25
            y = start[1] + (end[1] - start[1]) * 0.25
            # Dessiner le segment sur l'image
            cv2.line(img, start, end, (255, 0, 0), 2)
            draw_text(img=img, text="{:.1f} cm".format(distances[j][n]), pos=(int(x) + 10, int(y) - 10), 
                  font_scale=2, font_thickness=2, text_color=(255, 255, 255), text_color_bg=(0, 0, 0))
    return img

def extract_desired_color_coordinates(img, color):
    # Créer un masque pour isoler les pixels de la couleur cible
    mask = np.all(img == color, axis=-1)
    # Trouver les coordonnées des pixels de la couleur cible
    x, y = np.where(mask)
    return np.column_stack((x, y))

def get_segment_points(outputs, im):
    
    # Calculer le mask et points de segmentation
    mask_seg = outputs["instances"].pred_masks.cpu().numpy()
    # np.save('mask_seg.npy', mask_seg)
    coordonnes = get_segmentation(mask_seg)

    #Convertir les coordonnées en un tableau numpy
    if coordonnes == []:
        return None, None, None
    coords = np.array(coordonnes)
    
    boxes = []
    for i, coord in enumerate(coords):
        rect = cv2.minAreaRect(np.array(coord))
        box = cv2.boxPoints(rect)
        boxes.append(box)

    # Initialiser un tableau de tableau vide pour stocker les points de segmentation de chaque espece
    segment_points = []
    for i, box in enumerate(boxes):
        segment_point = []
        # Récupérer les coordonnées x et y des points de la boite
        x = [p[0] for p in box]
        y = [p[1] for p in box]
        ## Calculer les points médians des segments horizontaux
        segment_point.append((((x[0]+x[1])/2), ((y[0]+y[1])/2)))
        segment_point.append((((x[2]+x[3])/2), ((y[2]+y[3])/2)))
        ## Calculer les points médians des segments verticaux
        segment_point.append((((x[1]+x[2])/2), ((y[1]+y[2])/2))) 
        segment_point.append((((x[3]+x[0])/2), ((y[3]+y[0])/2)))
        # Sauvegarde des points de cette espece
        # segment_point = extract_caractristics(mask_seg[i], segment_point, im)
        # print(segment_point)
        segment_points.append(segment_point)
        
    return segment_points, coords, boxes

def seg_img(self, SCORE_THRESH_TEST = 0.8, show_inf = False, show_3d = False, show_final = True):
    im2 = cv2.imread(self.path1)
    im1 = cv2.imread(self.path2)
    im1 = imutils.resize(im1, width=640, height=640)
    im2 = imutils.resize(im2, width=640, height=640)

    weights = 'models/model_final.pth'

    calib_cam = 'settings/camera_parameters/stereo_params.pkl'

    mtx1, mtx2, R, T = load_calibration(calib_cam)

    # Calculate the projection martrix
    P1, P2 = get_projection_matrix(mtx1, mtx2, R, T)

    # get config
    predictor, cfg = init_config(weights, SCORE_THRESH_TEST)


    # Inferred with the images of each camera
    output1, im_seg1 = inference(predictor, cfg,  im1.copy())
    output2, im_seg2 = inference(predictor, cfg,  im2.copy())
    
    if show_inf:
        cv2.imshow("Image segmentee 1", im_seg1)
        cv2.imshow("Image segmentee 2", im_seg2)
        cv2.waitKey(0)
        cv2.destroyAllWindows() 

    # voir les classes predites
    ## le resultat est de la forme : tensor([0, 1, 1, 2, 3, 3]), cela veut dire :
    ## une espces PFE, deux actinia fermées, une ouverte et deux gibbula
    classes1 = output1["instances"].pred_classes
    classes2 = output2["instances"].pred_classes

    # --------------------------------------------#

    # maintenant il faut trouver comment traiter que 
    # les class en commune entre l'image 1 et 2

    # --------------------------------------------#

    # Get segmentation points " A optimiser "
    uvs1, seg1, boxes1 = get_segment_points(output1, im1)
    uvs2, seg2, boxes2 = get_segment_points(output2, im2)
    
    if uvs1==None:
        self.error.error_msg = f"Aucune detection dans l'image 1 \nVeuillez diminuer le seuil de detection"
        self.error.check_error = True
        return
    if uvs2==None:
        self.error.error_msg = f"Aucune detection dans l'image 2 \nVeuillez diminuer le seuil de detection"
        self.error.check_error = True
        return 

    # transforme the 2D points in the images to 3D points in the exit()world
    # Il faut avoir le meme nombre de pairs de points dans les deux images
    if len(uvs1) == len(uvs2):
        p3dss = transforme_to_3D(P1, P2, uvs1, uvs2)
    else:
        self.error.error_msg = "Nombre de pairs de points dans les deux images est different \nVeuillez changer d'image ou modifier le seuil de detectrion"
        self.error.check_error = True
        return 
        
    ######################
    ## len(p3dss) = 2   ## => car il y'a deux espces detectées
    ## p3dss[0].shape = ## 
    ## (4, 3)           ## => 4 points (largeur longuer), 3 c'est x,y,z
    ######################

    if show_3d:
        # visualize the 3D points
        show_scatter_3D(p3dss)
        
    if show_final:
        distances, connections = get_3D_distances(p3dss, connections = [[0,2], [1,3]])

        class_dict = {
            "0" : "PFE",
            "1" : "Actinia fermee",
            "2" : "Actinia ouverte",
            "3" : "Gibbula"
        }

        im1_dist = dist_on_img(uvs1, boxes1, im_seg1, distances, classes1, class_dict)
        # im2_dist = dist_on_img(uvs2, boxes2, im_seg2, distances, classes2, class_dict)
        
        cv2.imshow("Image 1 segmentee avec distances", im1_dist)
        # cv2.imshow("Image 2 segmentee avec distances", im2_dist)
        cv2.waitKey(0)
        cv2.destroyAllWindows() 

    # on a 4 distances ici ... 2 espces, chaque espces a deux distances (longeur et largeur)
    # distances = [[36.848607476959444, 32.03362311274617], [29.84703379255751, 45.008806942883936]]