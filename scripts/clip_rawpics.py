# coding: utf-8
import argparse, os, sys, glob

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import math

# https://github.com/tzutalin/labelImg

def plotImages(images, labels=None, save_as=None, x=None, y=None):
    if not (x and y):
        print(math.sqrt(3))
        n = math.sqrt(len(images))
        if n != int(n):
            n = int(n) + 1
        else:
            n = int(n)
        x = n
        y = n
    fig, axes = plt.subplots(y, x)

    axes = axes.flatten()
    for i in range(y*x):
        ax = axes[i]
        if i <= len(images) - 1:
            img = images[i]
            ax.imshow(img)
            if labels is not None:
                ax.set_title(labels[i])
        ax.axis('off')
    plt.tight_layout()
    if save_as:
        plt.savefig(save_as)
    else:
        plt.show()

def clip_bench(img):
    start_x = 175
    y = 275
    w = 50
    h = 70

    images = []
    for i in range(9):
        x = start_x + i * w
        images.append(img[y:y+h , x:x+w])
    plotImages(images, x=3, y=3)
    return images 

def clip_myfield(img):
    images = []

    # 1段目
    start_x = 240
    y = 150
    w = 53
    h = 70
    #images = []
    for i in range(7):
        x = start_x + i * w
        images.append(img[y:y+h , x:x+w])

    # 2段目
    start_x = 215
    y = 195
    w = 53
    h = 70
    for i in range(7):
        x = start_x + i * w
        images.append(img[y:y+h , x:x+w])

    # 3段目
    start_x = 240
    y = 230
    w = 53
    h = 70
    for i in range(7):
        x = start_x + i * w
        images.append(img[y:y+h , x:x+w])

    plotImages(images, x=7, y=3, save_as='clipped_field.png')
    exit(1)

def clip(path):
    img = Image.open(path)
    img = np.asarray(img)
    # clip_bench(img)
    clip_myfield(img)




def main(args):
    for path in glob.glob(args.data_dir + '/*.png'):
        clip(path)

if __name__ == "__main__":
      parser = argparse.ArgumentParser()
      parser.add_argument('--data-dir', default='datasets/rawpics')
      parser.add_argument('--save-dir', default='datasets/clipped')
      args = parser.parse_args()
      main(args)
