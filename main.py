import logging
logging.info(f"model initialisation in progress ...")


import os
import platform
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
from utils.dataloaders import (IMG_FORMATS, VID_FORMATS, check_file, increment_path, select_device, check_imshow, 
                               Profile, LoadImages, LoadScreenshots, LoadStreams)


def run(
        weights = ROOT / 'models/model_final.pth',                          # model path or triton URL
        save_rest = True,                                                    # save inference images
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
        visualize=False,                                                    # visualize features
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

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset_1 = LoadStreams(src1, img_size=imgsz) 
        dataset_2 = LoadStreams(src2, img_size=imgsz) 
        bs = len(dataset_1)
        
    elif screenshot:
        dataset_1 = LoadScreenshots(src1, img_size=imgsz) # , stride=stride, auto=pt
        dataset_2 = LoadScreenshots(src2, img_size=imgsz) # , stride=stride, auto=pt
    
    else:
        dataset_1 = LoadImages(src1, img_size=imgsz) # , stride=stride, auto=pt, vid_stride=vid_stride
        dataset_2 = LoadImages(src2, img_size=imgsz)
    
    assert len(dataset_1) == len(dataset_2), 'The size of the two datasets must be equal.'
    assert dataset_1.mode == dataset_2.mode, 'Both datasets must have the same mode.'
    
    vid_path, vid_writer = None, None         
    
    mtx1, mtx2, R, T = load_calibration(calib_cam)
    # Calculate the projection martrix
    P1, P2 = get_projection_matrix(mtx1, mtx2, R, T)
    
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    
    i = 0
    predictor, cfg = init_config(str(weights), SCORE_THRESH_TEST = 0.8)
    
    for (path1, im1, im0s1, vid_cap1, s1), (path2, im2, im0s2, vid_cap2, s2) in tqdm(zip(dataset_1, dataset_2)): # , unit='%', total=len(dataset_1), bar_format='{percentage:3.0f}%|{bar}|'
        
        i += 1
        
        im1 = np.transpose(im1, (1, 2, 0))[:,:,::-1]
        im2 = np.transpose(im2, (1, 2, 0))[:,:,::-1]
        
                
        # Inferred with the images of each camera
        output1 = inference(predictor, cfg,  im0s1, show=visualize)
        output2 = inference(predictor, cfg,  im0s2, show=visualize)
        
        # voir les classes predites
        ## le resultat est de la forme : tensor([0, 1, 1, 2, 3, 3]), cela veut dire :
        ## une espces PFE, deux actinia fermées, une ouverte et deux gibbula
        classes1 = output1["instances"].pred_classes
        classes2 = output2["instances"].pred_classes


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
        
        class_dict = {
                        "0" : "PFE",
                        "1" : "Actinia fermee",
                        "2" : "Actinia ouverte",
                        "3" : "Gibbula"
                     }

        
        distances, connections = get_3D_distances(p3ds, connections = [[0,2], [1,3]])
        
        im1_seg = dist_on_img(uvs1, boxes1, im0s1, distances, classes1, class_dict, copy=False, show=visualize)
        im2_seg = dist_on_img(uvs2, boxes2, im0s2, distances, classes2, class_dict, copy=False, show=visualize)
        
         # Stream results
        if view_img:
            if platform.system() == 'Linux' and path1 not in windows:
                windows.append(path1)
                cv2.namedWindow(str(path1), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                cv2.resizeWindow(str(path1), im1_seg.shape[1], im1_seg.shape[0])
            cv2.imshow(str(path1), im1_seg)
            cv2.waitKey(1)  # 1 millisecond
        
        # Save results (image with detections)
        if save_rest:
            save_path = str(save_dir / Path(path1).name)
            if dataset_1.mode == 'image':
                cv2.imwrite(save_path, im1_seg)
            else:  # 'video' or 'stream'
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer
                    if vid_cap1:  # video
                        fps = vid_cap1.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im1_seg.shape[1], im1_seg.shape[0]
                    save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer.write(im1_seg)
           

if __name__ == '__main__':
    run(src1 = './data/video_c1', src2 = './data/video_c2')
    
    
    
"""
 * Gérer le cas ou l'on met deux webcam/url
 * S'assurer que les deux sources sont de même type image/dissier/webcam/cameras/url..
 * S'assurer que les deux sources ont le même contenu.
 * Récuperer les information aquise extraite dans un dataframe.
    - nombre de chaque espèce pour chaque image.
    - mosiner les dimensions de chaque espèce (taille longueur).
 * Ajouter un loggger
 * Enlever la partie webcam et screen
"""
        
        

    



















