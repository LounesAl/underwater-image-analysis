import cv2
from utils.segmentation import inference

# charger l'image
img = cv2.imread("/home/bra/Downloads/underwater-image-analysis/data/imgs_c0/gibbula_07_16-1137-_jpg.rf.14010270c5faa4b12daee4276d06fbcf.jpg")
output, _ = inference[img]

# convertir l'image en niveaux de gris
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# trouver les contours dans l'image
contours, _ = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# pour chaque contour
for contour in contours:
    # trouver les moments de l'image
    moments = cv2.moments(contour)
    if moments["m00"] != 0:
        # calculer le centre de gravité
        cX = int(moments["m10"] / moments["m00"])
        cY = int(moments["m01"] / moments["m00"])
        # dessiner un cercle autour du centre de gravité
        cv2.circle(img, (cX, cY), 5, (255, 0, 0), -1)

# afficher l'image
cv2.imshow("image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()



