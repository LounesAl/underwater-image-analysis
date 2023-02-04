import logging
logging.info(f"model initialisation in progress ...")


import os
import platform
from pathlib import Path
import sys
from tqdm import tqdm
import argparse
import pandas as pd
from copy import deepcopy

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # UNDERWATER-IMAGE-ANALYSIS root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from utils.segmentation import *
from utils.calibration import *
from utils.dataloaders import (IMG_FORMATS, VID_FORMATS, check_file, increment_path, select_device, print_args, Profile, LoadImages)


# progress_bar,                                                               # peogress bar for application
def one_only_camera(progress_bar,
        weights=ROOT / 'models/model_final.pth',                                    # model path or triton URL
        save_rest=True,                                                             # save inference images
        src1=ROOT / 'data/imgs_c1',                                                 # file/dir/URL/glob/screen/0(webcam)
        src2=ROOT / 'data/imgs_c1',                                                 # file/dir/URL/glob/screen/0(webcam)
        imgsz=(640, 640),                                                           # inference size (height, width)
        calib_cam=ROOT / 'settings/camera_parameters/stereo_params.pkl',            # stereo cameras path parameters 
        conf_thres=0.25,                                                            # confidence threshold
        device='',                                                                  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,                                                             # visualize results
        visualize=False,                                                            # visualize features
        project=ROOT / 'runs/detect',                                               # save results to project/name
        name='exp',                                                                 # save results to project/name
        exist_ok=False,                                                             # existing project/name ok, do not increment
        nb_lines=15,                                                                # number of lines/distance between the counter and the center of gravity 
        draw_size=1,                                                                # the width of the markers 
        save_df=True,                                                               # save datafram of statistics
       ):
    
    src1, src2, calib_cam = str(src1), str(src2), str(calib_cam)
    is_file = Path(src1).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = src1.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = src1.isnumeric() or src1.endswith('.streams') or (is_url and not is_file)
    screenshot = src1.lower().startswith('screen')
    if is_url and is_file:
        src1 = check_file(src1)  # download
        
    progress_bar.setValue(10)

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)

    # Dataloader
    if webcam:
        logging.info(f'the url form is not supported')
        sys.exit()
        
    elif screenshot:
        logging.info(f'the screenshot form is not supported')
        sys.exit()
    else:
        dataset = LoadImages(src1, img_size=imgsz)
        
    progress_bar.setValue(20)
    
    assert len(dataset) > 0, 'no file was found .'
    
    vid_path, vid_writer = None, None         
    
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    
    
    predictor, cfg = init_config(str(weights), SCORE_THRESH_TEST = conf_thres)
    
    progress_bar.setValue(40)
    # dataset = tqdm(zip(dataset_1, dataset_2), \
    #                    total=len(dataset_1) if dataset_1.mode=='image' else dataset_1.frames, \
    #                    desc=f'Detection of characteristics ')
    for (path, im, im0s, vid_cap, s)  in dataset:
        # , unit='%', total=len(dataset_1), bar_format='{percentage:3.0f}%|{bar}|'
        
        # progress_bar.setValue(100 * dataset.frame / dataset.frames)
        
        im = np.transpose(im, (1, 2, 0))[:,:,::-1]
                
        # Inferred with the images of each camera
        output, im0s = inference(predictor, cfg,  im0s)
        
        masks = output["instances"].pred_masks.cpu().numpy().astype(np.uint8)

                 
        # s1 += '%gx%g ' % im0s1.shape[2:]
        print(s)
        
        if len(output) > 0:
            
            for mask in masks:
            
                contours_cam1, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours_cam2, _ = cv2.findContours(masks_cam2[arg2], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if not (len(contours_cam1) == len(contours_cam2) == 1):
                    continue
                # assert len(contours_cam1) == 1 and len(contours_cam2) == 1, \
                # f'we are supposed to retrieve a single contour that represents a species in what we have \
                #  {len(contours_cam1)} contours in camera0 and {len(contours_cam2)} in contour1  .'
                
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
                    spaced_indices_cam1 = np.round(np.linspace(0, len(cnt_cam1) - 1, nb_lines))[:-1].astype(int)
                    spaced_indices_cam2 = np.round(np.linspace(0, len(cnt_cam2) - 1, nb_lines))[:-1].astype(int)
                    
                    # Utilise les indices pour extraire les points du contour
                    subset_cam1 = cnt_cam1[spaced_indices_cam1]
                    subset_cam2 = cnt_cam2[spaced_indices_cam2]
                    
                    # dessiner un cercle autour du centre de gravité
                    cv2.drawContours(im0s1, [cnt_cam1], -1, (0, 255, 0), draw_size)
                    
                    p3ds = []
                    temp_dist = []
                    for i, sub_cam1, sub_cam2, color in zip(range(len(subset_cam1)), subset_cam1, subset_cam2, COLORS):
                        
                        cv2.line(im0s1, (cx_cam1, cy_cam1), tuple(sub_cam1[0]), color, draw_size)
                        
                        # calculer la distance le centre et les autres points autour
                        u1 = (cx_cam1, cy_cam1) if i == 0 else tuple(sub_cam1[0])
                        u2 = (cx_cam2, cy_cam2) if i == 0 else tuple(sub_cam2[0])
                        p3d = DLT(P1, P2, u1, u2)
                        p3ds.append(p3d)
                        if i!= 0 : temp_dist.append(np.linalg.norm(p3ds[-1] - p3ds[0]))
                    
                    temp_dist = sorted(temp_dist, reverse=True)
                    height = np.mean(temp_dist[:int(0.7*len(temp_dist))]) * 2 
                    width = np.mean(temp_dist[int(0.7*len(temp_dist)):]) * 2
                    draw_text(img=im0s1, text="{} W : {:.1f} cm L : {:.1f} cm".format(CLASSES_DICT[str(classes1[arg1].item())], width, height), 
                              pos=tuple(sub_cam1[0]), font_scale=1, font_thickness=1, text_color=(255, 255, 255), text_color_bg=(0, 0, 0))
        
         # Stream results
        if view_img:
            if platform.system() == 'Linux' and path not in windows:
                windows.append(path)
                cv2.namedWindow(str(path), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                cv2.resizeWindow(str(path), im0s.shape[1], im0s.shape[0])
            cv2.imshow(str(path), im0s)
            cv2.waitKey(1)  # 1 millisecond
        
        # Save results (image with detections)
        if save_rest:
            save_path = str(save_dir / Path(path).name)
            if dataset.mode == 'image':
                cv2.imwrite(save_path, im0s)
            else:  # 'video' or 'stream'
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0s.shape[1], im0s.shape[0]
                    save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer.write(im0s) 
                  
    progress_bar.setValue(100)         
    

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'models/model_final.pth', help='model path')
    parser.add_argument('--save-rest', action='store_false', help='save results of inference')    
    parser.add_argument('--src1', type=str, default=ROOT / 'data/video_c0', help='file/dir/')
    parser.add_argument('--src2', type=str, default=ROOT / 'data/video_c1', help='file/dir/')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=(640, 640), help='inference size h,w')
    parser.add_argument('--calib-cam', default=ROOT / 'settings/camera_parameters/stereo_params.pkl', help='parameters calibration for cameras')    
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--nb-lines', type=int, default=15, help='number of lines/distance between the counter and the center of gravity')
    parser.add_argument('--draw-size', type=int, default=1, help='the width of the markers')
    parser.add_argument('--save-df', action='store_false', help='save datafram of statistics')  

    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt       

def main():
    opt = parse_opt()
    one_only_camera(**vars(opt))
           

if __name__ == '__main__':
    main()
    
        
        

    



















