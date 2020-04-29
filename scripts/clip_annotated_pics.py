# coding: utf-8
import argparse, os, sys, glob

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import math
import pandas as pd
import xml.etree.ElementTree as ET

def clip(img, xml):
    origin_file = xml.getroot().find('filename').text
    objects = xml.getroot().findall('object')
    for obj in objects:
        label = obj.find('name').text
        bndbox = obj.find('bndbox').getchildren()
        xmin, ymin, xmax, ymax = bndbox
        print(label)
        print(bndbox)
        exit(1)
    pass
    

def main(args):
    for xml_path in glob.glob(args.data_dir + '/*.xml'):
        img_path = '.'.join(xml_path.split('.')[:-1]) + '.' + args.rawpics_ext
        if not os.path.exists(img_path):
            continue
        img = Image.open(img_path)
        xml = ET.parse(xml_path)
        clip(img, xml)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='datasets/rawpics')
    parser.add_argument('--save-dir', default='datasets/clipped')
    parser.add_argument('--rawpics-ext', default='jpg')
    args = parser.parse_args()
    main(args)

