# import cv2
# import os

# # Ouvrez la vidéo
# video = cv2.VideoCapture("actinia.mp4")

# # Récupérer le nombre total de frames
# total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

# # Définir le dossier de sortie
# if not os.path.exists("data/frames"):
#     os.makedirs("data/frames")

# # Boucle sur toutes les images et les enregistrer dans le dossier
# for i in range(total_frames):
#     ret, frame = video.read()
#     cv2.imwrite("data/frames/frame_{}.jpg".format(i), frame)

# # Libérer la vidéo
# video.release()

from utils.segmentation import *
from utils.calibration import *

weights = 'models/model_final.pth'

path = 'data/imgs_c1'

segmentation = []

files = os.listdir(path)

for i in range(0,len(files), 40):
    im1 = cv2.imread(os.path.join(path, files[i]))
    
    print(files[i])
    output1 = inference(weights, im1, show=False)
    np.save("data/outputs/output_{}.npy".format(i), output1)
    cv2.imwrite("data/outputs/output_{}.jpg".format(i), im1)

print('OK')