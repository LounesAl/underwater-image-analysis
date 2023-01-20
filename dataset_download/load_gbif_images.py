import urllib.request
from tqdm import tqdm
import os
import sys
import argparse
import time
import pandas as pd
import re
import logging

CURRENT_PATH = os.path.dirname(os.path.abspath('__file__'))

logging.basicConfig(level=logging.DEBUG, 
                    filename=os.path.join(CURRENT_PATH, 'logFile.log'),
                    filemode='w',
                    format='%(asctime)s - %(levelname)s - %(message)s')

def main(opt):
    logging.info('Start converting to CSV file ...')
    # assert opt.output_file[-4:] == '.csv', "The output file must be in (.csv) format."
    with open(opt.input_file, 'r', encoding='utf-8') as input_file, \
         open(opt.output_file + '.csv', 'w', encoding='utf-8', newline='') as output_file:
        lignes = input_file.readlines()
        [output_file.write(re.sub(r' ', '-', re.sub(r'\t+', ',', ligne))) for ligne in lignes] 

    links = pd.read_csv(opt.output_file + '.csv', delimiter=",", usecols=['identifier']).iloc[:, 0]
    if opt.img_max > len(links) : opt.img_max = len(links)
    
    try:
         os.makedirs(os.path.join(CURRENT_PATH, opt.images_stacking.upper()))
    except FileExistsError:
        if not opt.continu:
            logging.error(f"You have chosen not to overwrite the file {opt.images_stacking.upper()}.")
            sys.exit()
        
    if opt.rmInputFile:
        os.remove(os.path.join(CURRENT_PATH, opt.input_file))
    
    nm_unloaded_images = 0
    for i, link in tqdm(enumerate(links[:opt.img_max]), total=opt.img_max, unit='%', bar_format='{percentage:3.0f}%|{bar}|'):
        id_no = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(time.time()))
        try:
            urllib.request.urlretrieve(link, os.path.join(opt.images_stacking.upper(), 
                                                          opt.images_stacking + '_' + id_no + "_{:05d}".format(i) + '.' + opt.expansion))
        except Exception as e:
            nm_unloaded_images += 1
            logging.info(f"the image under this link:'{link}' can not be downloaded.")
            continue

    logging.info(f"Number of unloaded images :  {nm_unloaded_images}.")
    logging.info(f"Number of loaded images   : {i+1-nm_unloaded_images}.")


def get_arguments(known=True):
    parser = argparse.ArgumentParser(description='Download via the GBIF platform')
    parser.add_argument('--input_file', type=str, default=os.path.join(CURRENT_PATH, 'multimedia.txt'), help='input (txt) file path')
    parser.add_argument('--output_file', type=str, default='multimedia', help='output (csv) file path')
    parser.add_argument('--images_stacking', type=str, default='GIBBULA', help='images stacking folder path')
    parser.add_argument('--expansion', type=str, default='jpeg', help='image expansion during recording')
    parser.add_argument('--img_max', type=int, default='100000000000', help='number of images to store')

    # Python < 3.9
    parser.add_argument('--rmInputFile', default='0', type = int, help='delete the mutimedia file after downloading')

    # Python < 3.9
    parser.add_argument('--continu', default='1', type = int, help='do not erase the images folder')

    return parser.parse_known_args()[0] if known else parser.parse_args()

            
if __name__ =='__main__':
    opt = get_arguments()
    main(opt)