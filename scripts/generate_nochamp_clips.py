# coding: utf-8
import argparse, os, sys, glob
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import numpy as np


sys.path.append(os.getcwd())
from clip_rawpics import clip_myfield, clip_bench

def clip_arena(output_dir, path):
    img = Image.open(path)
    img_width, img_height = img.size

    screenshot_x = 2880
    screenshot_y = 1800
    # screenshot_x = 2876
    # screenshot_y = 1606

    x_left_offset = 0
    x_right_offset = 5
    y_upper_offset = 25
    y_lower_offset = 103
    img = img.crop((x_left_offset, 
                    y_upper_offset, 
                    img_width - x_right_offset, 
                    img_height - y_lower_offset))
    img = img.resize((screenshot_x, screenshot_y))
    # img.show()
    basename_without_ext = os.path.splitext(os.path.basename(path))[0]

    img = np.asarray(img)
    os.makedirs(output_dir + '/' + basename_without_ext, exist_ok=True)
    clip_myfield(img, output_dir + '/' + basename_without_ext)
    clip_bench(img, output_dir + '/' + basename_without_ext)

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    for img_path in glob.glob(args.arenas_dir + '/*.png'):
        print(img_path)
        clip_arena(args.output_dir, img_path)

if __name__ == "__main__":
      parser = argparse.ArgumentParser()
      parser.add_argument('--arenas-dir', default='icons/arenas')
      # parser.add_argument('--output-dir', default='icons/clipped_arenas')
      parser.add_argument('--output-dir', default='datasets/emptys')
      args = parser.parse_args()
      main(args)
