##########################################################################################################################
##                     Demo pour utiliser detectron 2 avec le modele pre entrainée sur les gibule et actinia            ##
##                                           install detectron 2 on linux only CPU                                      ##
## 1/ conda create --name detectron python==3.8                                                                         ##
## 2/ conda activate detectron                                                                                          ##
## 3/ conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cpuonly -c pytorch                           ##
## 4/ python -m pip install detectron2 -f   https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.10/index.html   ##
## 5/ pip install opencv-python                                                                                         ##
##                                          TO install it on windows or on GPU                                          ##
##                                                  Check version here                                                  ##
## 1/ https://pytorch.org/get-started/previous-versions/                                                                ##
## 2/ https://detectron2.readthedocs.io/en/latest/tutorials/install.html                                                ##
##########################################################################################################################

if __name__ == "__main__":

    print("Initialisation du model en cours ...")
    from utils import *
    path_model = "Sementation/model/model_final.pth"
    path_img = "dataset_download/test/gibbula_07_16-1101-_jpg.rf.5751193b6d97c81c7ed7f10ae7541bba.jpg"
    output = inference(path_model, path_img, show=True)
    points_car, seg = get_segment_points(output, 0, show = True)
    # print(seg)
    # np.save('seg.npy', seg)