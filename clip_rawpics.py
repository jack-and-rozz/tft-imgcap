# coding: utf-8
import argparse, os, sys, glob

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import numpy as np
import math
from tensorflow.keras.models import load_model
import re
from dataset import load_classes_from_definition

from util import get_best_and_final_model_path, dotDict

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

def clip_myfield(img, output_dir):
    #images = []

    height = len(img)
    width = len(img[0])
    ref_image_width = 2876
    ref_image_height = 1606

    ref_row_boundary_top = [440, 550, 670, 800]
    ref_row_boundary_left = [770, 830, 720, 790]
    ref_row_boundary_width = [1200, 1250, 1295, 1330]
    ref_row_boundary_height = [280, 280, 280, 280]
    ref_margin_width = 0
    ref_margin_height = 0

    margin_width = ref_margin_width / ref_image_width * width
    margin_height = ref_margin_height / ref_image_height * height

    for col in range(4):
        start_x = ref_row_boundary_left[col] / ref_image_width * width
        y = ref_row_boundary_top[col] / ref_image_height * height
        w = ref_row_boundary_width[col] / 7 / ref_image_width * width
        h = ref_row_boundary_height[col] / ref_image_height * height
        for i in range(7):
            x = start_x + i * w
            #images.append(img[int(y - margin_height):int(y+h + margin_height) , int(x - margin_width):int(x+w + margin_width)])
            pilImg = Image.fromarray(img[int(y - margin_height):int(y+h + margin_height) , int(x - margin_width):int(x+w + margin_width)])
            pilImg.save(output_dir + '/clipped_field_' + str(i) + '_' + str(col) + '.png')

    #plotImages(images, x=7, y=4, save_as=args.save_dir + '/clipped_field.png')

def clip_bench(img, output_dir):
    height = len(img)
    width = len(img[0])
    ref_image_width = 2876
    ref_image_height = 1606

    ref_boundary_top = 960
    ref_boundary_left = 550
    ref_boundary_width = 1590
    ref_boundary_height = 280
    ref_margin_width = 0
    ref_margin_height = 0

    margin_width = ref_margin_width / ref_image_width * width
    margin_height = ref_margin_height / ref_image_height * height

    start_x = ref_boundary_left / ref_image_width * width
    y = ref_boundary_top / ref_image_height * height
    w = ref_boundary_width / 9 / ref_image_width * width
    h = ref_boundary_height / ref_image_height * height
    for i in range(9):
        x = start_x + i * w
        pilImg = Image.fromarray(img[int(y - margin_height):int(y+h + margin_height) , int(x - margin_width):int(x+w + margin_width)])
        pilImg.save(output_dir + '/clipped_bench_' + str(i) + '.png')

def clip(path):
    basename_without_ext = os.path.splitext(os.path.basename(path))[0]
    output_dir = args.save_dir + "/" + basename_without_ext
    os.makedirs(output_dir, exist_ok=True)

    img = Image.open(path) 
    img.save(output_dir + "/" + os.path.basename(path))
    img = np.asarray(img)

    clip_myfield(img, output_dir)
    clip_bench(img, output_dir)

def generate_estimate_image(movie_dir, model):
    # make estimated champion position
    im = Image.new("RGB", (1500, 450), (128, 128, 128))
    id2class, _ = load_classes_from_definition(["champion"])

    field_left = 60
    field_top = 0

    bench_left = 0
    bench_top = 350

    # estimate field
    for path in glob.glob(movie_dir + "/*"):
        result = re.search(r'clipped_field_([0-9])_([0-9])', path)
        if result == None:
            continue

        img = Image.open(path).convert('RGB')
        img = img.resize((80, 100))

        # 上位３位まで取得
        outputs_all = model.predict(np.asarray([np.asarray(img) / 255]))
        outputs = (-outputs_all).argsort(axis=-1)[:, :3]

        hypotheses = []
        for i in range(3):
            hypotheses.append(id2class["champion"][outputs[0][i]])

        x = int(result.group(1))
        y = int(result.group(2))


        x_offset = 40
        if y % 2 == 0:
            x_offset = 0

        im1 = Image.open(args.icon_dir + '/champions/' + hypotheses[0] + '.png')
        im1 = im1.resize((64, 64))
        im.paste(im1, (x_offset + field_left + x * 80, field_top + y * 80))

        im1 = Image.open(args.icon_dir + '/champions/' + hypotheses[1] + '.png')
        im1 = im1.resize((30, 30))
        im.paste(im1, (x_offset + field_left + x * 80, field_top + y * 80 + 40))
        im1 = Image.open(args.icon_dir + '/champions/' + hypotheses[2] + '.png')
        im1 = im1.resize((20, 20))
        im.paste(im1, (x_offset + field_left + x * 80 + 30, field_top + y * 80 + 50))

        draw = ImageDraw.Draw(im)
        draw.text((x_offset + field_left + x * 80, field_top + y * 80), str(outputs_all[0][outputs][0][0]), (255, 255, 255))

        img_ref = img.resize((60, 75))
        im.paste(img_ref, ( 750 + field_left + x * 80 + x_offset, field_top + y * 80))

    # estimate bench
    for path in glob.glob(movie_dir + "/*"):
        result = re.search(r'clipped_bench_([0-9])', path)
        if result == None:
            continue

        img = Image.open(path).convert('RGB')
        img = img.resize((80, 100))

        # 上位３位まで取得
        outputs_all = model.predict(np.asarray([np.asarray(img) / 255]))
        outputs = (-outputs_all).argsort(axis=-1)[:, :3]

        hypotheses = []
        for i in range(3):
            hypotheses.append(id2class["champion"][outputs[0][i]])

        x = int(result.group(1))

        im1 = Image.open(args.icon_dir + '/champions/' + hypotheses[0] + '.png')
        im1 = im1.resize((64, 64))
        im.paste(im1, (bench_left + x * 80, bench_top))

        im1 = Image.open(args.icon_dir + '/champions/' + hypotheses[1] + '.png')
        im1 = im1.resize((30, 30))
        im.paste(im1, (bench_left + x * 80, bench_top + 40))
        im1 = Image.open(args.icon_dir + '/champions/' + hypotheses[2] + '.png')
        im1 = im1.resize((20, 20))
        im.paste(im1, (bench_left + x * 80 + 30, bench_top + 50))

        draw = ImageDraw.Draw(im)
        draw.text((bench_left + x * 80, bench_top), str(outputs_all[0][outputs][0][0]), (255, 255, 255))

        img_ref = img.resize((60, 75))
        im.paste(img_ref, (730 + bench_left + x * 80, bench_top))

    # ref_image_width = 2876
    # ref_image_height = 1606

    # for path in glob.glob(movie_dir + "/*"):
    #     if not os.path.basename(movie_dir) in os.path.basename(path):
    #         continue
    #     im_source = Image.open(path)
    #     width, height = im_source.size
    #     im_source = im_source.crop((550 / ref_image_width * width, 440 / ref_image_height * height, 2140 / ref_image_width * width, 1240 / ref_image_height * height))
    #     im_source = im_source.resize((700, 400))

    #     im.paste(im_source, (750, 0))

    im.save(movie_dir + "/" + os.path.basename(movie_dir) + '_prediction.png')


def main(args):
    for path in glob.glob(args.data_dir + '/**', recursive=True):
        if os.path.isdir(path):
            continue
        print(path)
        clip(path)

    # estimate image
    best_model_path, _ = get_best_and_final_model_path(args.model_root)
    model = load_model(best_model_path)

    for movie_dir in glob.glob(args.save_dir + '/*' ,recursive=True):
        if not os.path.isdir(movie_dir):
            print(movie_dir)
            continue
        generate_estimate_image(movie_dir, model)


if __name__ == "__main__":
      parser = argparse.ArgumentParser()
      parser.add_argument('model_root')
      parser.add_argument('--icon-dir', default='icons/set3-mid/')
      parser.add_argument('--data-dir', default='datasets/tests/rawpics')
      parser.add_argument('--save-dir', default='datasets/tests/clipped_raw')
      args = parser.parse_args()
      main(args)
