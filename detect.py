import logging
logging.info(f"model initialisation in progress ...")


import os
import platform
from pathlib import Path
import sys
from tqdm import tqdm
import argparse

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # UNDERWATER-IMAGE-ANALYSIS root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from utils.segmentation import *
from utils.calibration import *
from utils.dataloaders import (IMG_FORMATS, VID_FORMATS, check_file, increment_path, select_device, print_args, Profile, LoadImages)

def run(
        weights=ROOT / 'models/model_final.pth',                                    # model path or triton URL
        save_rest=True,                                                             # save inference images
        src1=ROOT / 'data/imgs_c1',                                                 # file/dir/URL/glob/screen/0(webcam)
        src2=ROOT / 'data/imgs_c1',                                                 # file/dir/URL/glob/screen/0(webcam)
        imgsz=(640, 640),                                                           # inference size (height, width)
        calib_cam=ROOT / 'settings/camera_parameters/stereo_params.pkl',  # stereo cameras path parameters 
        conf_thres=0.25,                                                            # confidence threshold
        device='',                                                                  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,                                                              # visualize results
        visualize=False,                                                            # visualize features
        project=ROOT / 'runs/detect',                                               # save results to project/name
        name='exp',                                                                 # save results to project/name
        exist_ok=False,                                                             # existing project/name ok, do not increment
):
    
    src1, src2, calib_cam = str(src1), str(src2), str(calib_cam)
    is_file = Path(src1).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = src1.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = src1.isnumeric() or src1.endswith('.streams') or (is_url and not is_file)
    screenshot = src1.lower().startswith('screen')
    if is_url and is_file:
        src1 = check_file(src1)  # download

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
        dataset_1 = LoadImages(src1, img_size=imgsz)
        dataset_2 = LoadImages(src2, img_size=imgsz)
    
    assert dataset_1.mode == dataset_2.mode, 'Both datasets must have the same mode.'
    assert len(dataset_1) == len(dataset_2), 'The size of the two datasets must be equal.'
    
    vid_path, vid_writer = None, None         
    
    mtx1, mtx2, R, T = load_calibration(calib_cam)
    # Calculate the projection martrix
    P1, P2 = get_projection_matrix(mtx1, mtx2, R, T)
    
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    
    i = 0
    predictor, cfg = init_config(str(weights), SCORE_THRESH_TEST = conf_thres)
    
    for (path1, im1, im0s1, vid_cap1, s1), (path2, im2, im0s2, vid_cap2, s2) in tqdm(zip(dataset_1, dataset_2), desc = f'Detection of characteristics '): 
       
        # , unit='%', total=len(dataset_1), bar_format='{percentage:3.0f}%|{bar}|'
        i += 1
        
        im1 = np.transpose(im1, (1, 2, 0))[:,:,::-1]
        im2 = np.transpose(im2, (1, 2, 0))[:,:,::-1]
        
                
        # Inferred with the images of each camera
        output1, _ = inference(predictor, cfg,  im0s1)
        output2, _ = inference(predictor, cfg,  im0s2)
        
        # voir les classes predites
        ## le resultat est de la forme : tensor([0, 1, 1, 2, 3, 3]), cela veut dire :
        ## une espces PFE, deux actinia fermées, une ouverte et deux gibbula
        classes1 = output1["instances"].pred_classes
        classes2 = output2["instances"].pred_classes


        # Get segmentation points " A optimiser "
        uvs1, seg1, boxes1 = get_segment_points(output1, im1)
        uvs2, seg2, boxes2 = get_segment_points(output2, im2)
        
        if (uvs1 == None or uvs2 == None) or (len(uvs1) != len(uvs2)):
            continue
                
        # transforme the 2D points in the images to 3D points in the exit()world
        p3ds = transforme_to_3D(P1, P2, uvs1, uvs2)
        
        
        class_dict = {
                        "0" : "PFE",
                        "1" : "Actinia fermee",
                        "2" : "Actinia ouverte",
                        "3" : "Gibbula"
                     }

        
        distances, connections = get_3D_distances(p3ds, connections = [[0,2], [1,3]])
        
        im1_seg = dist_on_img(uvs1, boxes1, im0s1, distances, classes1, class_dict, copy=False)
        im2_seg = dist_on_img(uvs2, boxes2, im0s2, distances, classes2, class_dict, copy=False)
        
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


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'models/model_final.pth', help='model path')
    parser.add_argument('--save-rest', action='store_false', help='save results of inference')    
    parser.add_argument('--src1', type=str, default=ROOT / 'data/video_c1', help='file/dir/')
    parser.add_argument('--src2', type=str, default=ROOT / 'data/video_c2', help='file/dir/')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=(640, 640), help='inference size h,w')
    parser.add_argument('--calib-cam', default=ROOT / 'settings/camera_parameters/stereo_params.pkl', help='parameters calibration for cameras')    
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt       

def main():
    opt = parse_opt()
    run(**vars(opt))
           

if __name__ == '__main__':
    main()
    
    
    
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
        
        

    



















