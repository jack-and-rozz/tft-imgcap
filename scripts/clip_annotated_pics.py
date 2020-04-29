# coding: utf-8
import argparse, os, sys, glob
sys.path.append(os.getcwd())

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import math

import pandas as pd
import xml.etree.ElementTree as ET
from main import plotImages

def parse_label(label):
    '''
    - format: champion(*star)(:item1, item2)
    '''
    label = label.split(':')
    items = [x.strip() for x in label[1].split(',')] if len(label) > 1 else []
    champion = label[0].split('*')
    star = int(champion[1]) if len(champion) > 1 else 1
    champion = champion[0].strip()
    return champion, star, items

def clip(entire_img, xml):
    # Record:  []
    origin_file = xml.getroot().find('filename').text
    objects = xml.getroot().findall('object')
    for obj in objects:
        label = obj.find('name').text
        bndbox = [int(x.text) for x in obj.find('bndbox')]
        xmin, ymin, xmax, ymax = bndbox
        img = entire_img[ymin:ymax, xmin:xmax]
        champion, star, items = parse_label(label)
        print(champion, star, items)
        exit(1)
        # plotImages([img])
        # # print(label)
        # # print(bndbox)
        # exit(1)
    pass
    

def main(args):
    for xml_path in glob.glob(args.data_dir + '/*.xml'):
        img_path = '.'.join(xml_path.split('.')[:-1]) + '.' + args.rawpics_ext
        if not os.path.exists(img_path):
            continue
        img = np.asarray(Image.open(img_path))
        xml = ET.parse(xml_path)
        clip(img, xml)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='datasets/rawpics')
    parser.add_argument('--save-dir', default='datasets/clipped')
    parser.add_argument('--rawpics-ext', default='jpg')
    args = parser.parse_args()
    main(args)

