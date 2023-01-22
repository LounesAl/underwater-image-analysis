##########################################################################################################################
##                     Demo pour utiliser detectron 2 avec le modele pre entrainée sur les gibule et actinia            ##
##                                           install detectron 2 on linux only CPU                                      ##
## 1/ conda create --name detectron python==3.8                                                                         ##
## 2/ conda activate detectron                                                                                          ##
## 3/ conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cpuonly -c pytorch                           ##
## 4/ python -m pip install detectron2 -f   https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.10/index.html   ##
## 5/ pip install opencv-python                                                                                         ##
##########################################################################################################################

import detectron2

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

# Charger les poids sauvegardés dans un modèle vierge
cfg = get_cfg()
cfg.MODEL.DEVICE='cpu'
# cfg.merge_from_file("path/to/config.yaml")
# cfg.MODEL.WEIGHTS = "path/to/model.pth"
# predictor = DefaultPredictor(cfg)

# # Chargez l'image
# from PIL import Image
# img = Image.open("path/to/image.jpg")