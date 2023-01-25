"""Dans un nouveau fichier : 
1) Acquisition (d'image, video ou GIF) avec les deux cameras en stereo:
            -> Lecture des images en temps rell avec opencv ou autres.
            -> Distance entre les deux cameras (paramètre à optimisé).
            -> Espèce sous marine (sous aquarium) ex: GIBBULA, TIMHARINES.
          
2) Récupérer les coords  pixels de longueur et largeur de chaque espèces pour les deux caméras: 
            -> Inférence des espèces avec notre modèle detectron 2 avec les deux caméras stereo
            -> Fonctions supplémentaires pour l'extraction des coords

3) Conversion 2D -> 3D:"""

# Gérer le cas ou 


import logging
logging.info(f"model initialisation in progress ...")


import os
from pathlib import Path
import sys
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # UNDERWATER-IMAGE-ANALYSIS root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from utils.segmentation import *
from utils.calibration import *
from utils.dataloaders import (IMG_FORMATS, VID_FORMATS, check_file, increment_path, select_device, check_imshow, LoadImages, LoadScreenshots, LoadStreams)


def run(
        weights = ROOT / 'models/model_final.pth',                          # model path or triton URL
        save_img = True,                                                    # save inference images
        src1 = ROOT / 'data/imgs_c1',                                       # file/dir/URL/glob/screen/0(webcam)
        src2 = ROOT / 'data/imgs_c1',                                       # file/dir/URL/glob/screen/0(webcam)
        imgsz=(640, 640),                                                   # inference size (height, width)
        calib_cam='stereo_calibration/camera_parameters/stereo_params.pkl', # stereo cameras path parameters 
        conf_thres=0.25,                                                    # confidence threshold
        iou_thres=0.45,                                                     # NMS IOU threshold
        device='',                                                          # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,                                                     # visualize results
        save_txt=False,                                                     # save results to *.txt
        nosave=False,                                                       # do not save images/videos
        classes=None,                                                       # filter by class: --class 0, or --class 0 2 3
        visualize=False,                                                     # visualize features
        update=False,                                                       # update all models
        project=ROOT / 'runs/detect',                                       # save results to project/name
        name='exp',                                                         # save results to project/name
        exist_ok=False,                                                     # existing project/name ok, do not increment
        vid_stride=1,                                                       # video frame-rate stride
):
    
    src1, src2 = str(src1), str(src2)
    is_file = Path(src1).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = src1.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = src1.isnumeric() or src1.endswith('.streams') or (is_url and not is_file)
    screenshot = src1.lower().startswith('screen')
    if is_url and is_file:
        src1 = check_file(src1)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    # model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset_1 = LoadStreams(src1, img_size=imgsz) # stride=stride, auto=pt, vid_stride=vid_stride
        dataset_2 = LoadStreams(src2, img_size=imgsz) # stride=stride, auto=pt, vid_stride=vid_stride
        bs = len(dataset_1)
        
    elif screenshot:
        dataset_1 = LoadScreenshots(src1, img_size=imgsz) # , stride=stride, auto=pt
        dataset_2 = LoadScreenshots(src2, img_size=imgsz) # , stride=stride, auto=pt
    
    else:
        dataset_1 = LoadImages(src1, img_size=imgsz) # , stride=stride, auto=pt, vid_stride=vid_stride
        dataset_2 = LoadImages(src2, img_size=imgsz)
    
    assert len(dataset_1) == len(dataset_2), 'The size of the two datasets must be equal.'
            
    vid_path, vid_writer = [None] * bs, [None] * bs
    
    mtx1, mtx2, R, T = load_calibration(calib_cam)
    # Calculate the projection martrix
    P1, P2 = get_projection_matrix(mtx1, mtx2, R, T)
    
    for i, (path1, im1, im0s1, vid_cap1, s), (path2, im2, im0s2, vid_cap2, s2) in tqdm(zip(range(len(dataset_1)),dataset_1, dataset_2), unit='%', total=len(dataset_1),
                                                                                        bar_format='{percentage:3.0f}%|{bar}|'):
        
        im1 = np.transpose(im1, (1, 2, 0))[:,:,::-1]
        im2 = np.transpose(im2, (1, 2, 0))[:,:,::-1]
        
        # Inferred with the images of each camera
        output1 = inference(str(weights), im1, show=visualize)
        output2 = inference(str(weights), im2, show=visualize)

        # Get segmentation points " A optimiser "
        uvs1, seg1, boxes1 = get_segment_points(output1)
        uvs2, seg2, boxes2 = get_segment_points(output2)
        
        if (uvs1 == None or uvs2 == None) or (len(uvs1) != len(uvs2)):
            continue
                
        # transforme the 2D points in the images to 3D points in the exit()world
        p3ds = transforme_to_3D(P1, P2, uvs1, uvs2)
        
        if visualize:
            # visualize the 3D points
            show_scatter_3D(p3ds)
        
        distances, connections = get_3D_distances(p3ds, connections = [[0,2], [1,3]])
        im1_seg = get_dist_on_img(uvs1, boxes1, im1, distances, connections, show=True)
        im2_seg = get_dist_on_img(uvs2, boxes2, im2, distances, connections, show=True)
        
        # Save results (image with detections)
        if save_img:
            if dataset_1.mode == 'image':
                if not os.path.exists(str(save_dir / 'results')):
                    (save_dir / 'results').mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(Path(save_dir / f'results/img{i}.jpg')), im1_seg)
                
            else:  # 'video' or 'stream'
                if not os.path.exists(str(save_dir / 'results')):
                    (save_dir / 'results').mkdir(parents=True, exist_ok=True)
                
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))
                save_path = Path(save_dir / f'results_c1/img{i}.jpg')
                
                # if vid_path[i] != save_path :  # new video
                #     vid_path[i] = save_path
                #     if isinstance(vid_writer[i], cv2.VideoWriter):
                #         vid_writer[i].release()  # release previous video writer
                #     if vid_cap1 and vid_cap2:  # video
                #         fps = vid_cap.get(cv2.CAP_PROP_FPS)
                #         w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                #         h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        
                #     else:  # stream
                #         fps, w, h = 30, im0.shape[1], im0.shape[0]
                        
                #     save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                #     vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps1, (w1, h1))
                #     vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps1, (w1, h1))
                # vid_writer[i].write(im0)
        
        # if i==3: break
           

if __name__ == '__main__':
    run(src1 = './data/imgs_c1', src2 = './data/imgs_c2')
    
    
    
"""
 * Gérer le cas ou l'on met deux webcam/url
 * S'assurer que les deux sources sont de même type image/dissier/webcam/cameras/url..
 * S'assurer que les deux sources ont le même contenu.
 * Récuperer les information aquise extraite dans un dataframe.
    - nombre de chaque espèce pour chaque image.
    - mosiner les dimensions de chaque espèce (taille longueur).
"""
        
        

    



















