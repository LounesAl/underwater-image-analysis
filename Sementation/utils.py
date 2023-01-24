import cv2
import numpy as np
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2 import model_zoo

def visualiser(outputs, cfg, im):
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TEST[0]), scale=1.2)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imshow("Inference", v.get_image()[:, :, ::-1])
    cv2.waitKey(0)

def inference(path_model, path_img, show=True):
    im = cv2.imread(path_img)
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TEST = (f"PFE_valid",)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4
    cfg.MODEL.DEVICE='cpu'
    cfg.MODEL.WEIGHTS = path_model # Let training initialize from model zoo
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8
    predictor = DefaultPredictor(cfg)
    outputs = predictor(im)
    
    if show:
        visualiser(outputs, cfg, im)
    return outputs

def get_segmentation(mask_array):
    num_instances = mask_array.shape[0]
    coordinates = []
    for i in range(num_instances):
        indices = np.column_stack(np.where(mask_array[i] == True))
        coordinates.append([(y, x) for x, y in indices])
    return coordinates

def get_segment_points(outputs, ind_img, show = True):

  # Calculer le mask et points de segmentation
  mask_seg = outputs["instances"].pred_masks.cpu().numpy()
  coord = get_segmentation(mask_seg)

  # Créer une image blanche
  white_image = np.zeros((mask_seg.shape[1],mask_seg.shape[2],3), dtype=np.uint8)

  #Convertir les coordonnées en un tableau numpy
  coords = np.array(coord[ind_img])

  # Dessiner un contour autour des points
  cv2.drawContours(white_image, [coords], -1, (255, 255, 255), 2)

  # Dessiner le rectangle
  rect = cv2.minAreaRect(coords)
  box = cv2.boxPoints(rect)
  cv2.drawContours(white_image, [np.int0(box)], -1, (0, 255, 0), 2)

  # Initialiser un tableau vide pour stocker les points de segmentation
  segment_points = []

  # Récupérer les coordonnées x et y des points de la boite
  x = [p[0] for p in box]
  y = [p[1] for p in box]

  ## Calculer les points médians des segments horizontaux
  segment_points.append((((x[0]+x[1])/2), ((y[0]+y[1])/2)))
  segment_points.append((((x[2]+x[3])/2), ((y[2]+y[3])/2)))
  ## Calculer les points médians des segments verticaux
  segment_points.append((((x[1]+x[2])/2), ((y[1]+y[2])/2))) 
  segment_points.append((((x[3]+x[0])/2), ((y[3]+y[0])/2)))

  # Boucle sur les points de segmentation
  for i in range(0, len(segment_points), 2):
      start = (int(segment_points[i][0]), int(segment_points[i][1]))
      end = (int(segment_points[i+1][0]), int(segment_points[i+1][1]))
      # Dessiner le segment sur l'image
      cv2.line(white_image, start, end, (255, 0, 0), 2)
  if show == True:
    #Afficher l'image
    cv2.imshow("Points segmentation", white_image)
    cv2.waitKey(0)
  return segment_points, coord