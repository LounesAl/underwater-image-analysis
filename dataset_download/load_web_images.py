from bs4 import *
import requests
import os
import logging
import sys
from tqdm import tqdm
import argparse


CURRENT_PATH = os.path.dirname(os.path.abspath('__file__'))
 
# CREATE FOLDER
def folder_create(images, opt):
    try:
        # folder creation
        os.mkdir(os.path.join(CURRENT_PATH, opt.output_folder))
 
    # if folder exists with that name, ask another name
    except FileExistsError:
        if not opt.crush:
            logging.exception("Folder Exist with that name!")
            sys.exit()
 
    # image downloading start
    download_images(images, opt.output_folder)
 
 
# DOWNLOAD ALL IMAGES FROM THAT URL
def download_images(images, opt):
   
    # initial count is zero
    count = 0
 
    # print total images found in URL
    print(f"Total {len(images)} Image Found!")
 
    # checking if images is not zero
    if len(images) != 0:
        for i, image in tqdm(enumerate(images), total=len(images), unit='%', bar_format='{percentage:3.0f}%|{bar}|'):
            # From image tag ,Fetch image Source URL
 
                        # 1.data-srcset
                        # 2.data-src
                        # 3.data-fallback-src
                        # 4.src
 
            # Here we will use exception handling
 
            # first we will search for "data-srcset" in img tag
            try:
                # In image tag ,searching for "data-srcset"
                image_link = image["data-srcset"]
                 
            # then we will search for "data-src" in img
            # tag and so on..
            except:
                try:
                    # In image tag ,searching for "data-src"
                    image_link = image["data-src"]
                except:
                    try:
                        # In image tag ,searching for "data-fallback-src"
                        image_link = image["data-fallback-src"]
                    except:
                        try:
                            # In image tag ,searching for "src"
                            image_link = image["src"]
 
                        # if no Source URL found
                        except:
                            pass
 
            # After getting Image Source URL
            # We will try to get the content of image
            try:
                r = requests.get(image_link).content
                try:
 
                    # possibility of decode
                    r = str(r, 'utf-8')
 
                except UnicodeDecodeError:
 
                    # After checking above condition, Image Download start
                    with open(f"{opt.output_folder}/images{i+1}." + opt.expansion, "wb+") as f:
                        f.write(r)
 
                    # counting number of image downloaded
                    count += 1
            except:
                pass
 
        # There might be possible, that all
        # images not download
        # if all images download
        if count == len(images):
            print("All Images Downloaded!")
             
        # if all images not download
        else:
            print(f"Total {count} Images Downloaded Out of {len(images)}")
 
# MAIN FUNCTION START
def main(opt):
   
    # content of URL
    r = requests.get(opt.url)
 
    # Parse HTML Code
    soup = BeautifulSoup(r.text, 'html.parser')
 
    # find all images in URL
    images = soup.findAll('img')
 
    # Call folder create function
    folder_create(images, opt)

# ADD ARGUMENTS 
def get_arguments(known=True):
    parser = argparse.ArgumentParser(description='Download via the GBIF platform')
    parser.add_argument('--url', type=str, default='', help='input (txt) file path')
    parser.add_argument('--output_folder', type=str, default='lulu', help='images stacking folder path')
    parser.add_argument('--expansion', type=str, default='jpg', help='image expansion during recording')

    
    # Python < 3.9
    parser.add_argument('--crush', dest='crush', action='store_true')
    parser.add_argument('--no-crush', dest='crush', action='store_false')
    parser.set_defaults(crush=False)
    return parser.parse_known_args()[0] if known else parser.parse_args()

 

# MAIN PROGRAM
if __name__ =='__main__':
    # take url
    opt = get_arguments()
    
    # CALL MAIN FUNCTION
    main(opt)