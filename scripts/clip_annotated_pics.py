# coding: utf-8
import argparse, os, sys, glob, random
sys.path.append(os.getcwd())
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import math

import pandas as pd
import xml.etree.ElementTree as ET
from util import plotImages

def parse_label(label):
    '''
    - format: champion(*star)(:item1, item2)
    '''
    label = label.split(':')
    items = [x.strip() for x in label[1].split(',')] if len(label) > 1 else []

    if label[0]:
        champion = label[0].split('*')
        star = int(champion[1]) if len(champion) > 1 else 1
        champion = champion[0].strip()
    else:
        champion = 'items'
        star = 0
    items = sorted(items)
    return champion, star, items

def clip(entire_img, xml, champ_counts):
    # Record:  [target_file, source_file,]
    source_file = xml.getroot().find('filename').text
    objects = xml.getroot().findall('object')
    data = []
    for obj in objects:
        label = obj.find('name').text
        bndbox = [int(x.text) for x in obj.find('bndbox')]
        xmin, ymin, xmax, ymax = bndbox
        img = entire_img[ymin:ymax, xmin:xmax]
        champion, star, items = parse_label(label)
        target_file = "%s.%d.png" % (champion, champ_counts[champion])
        # target_path = os.getcwd() + '/' + args.save_dir + '/' + target_file
        target_path = args.save_dir + '/' + target_file
        champ_counts[champion] += 1

        labels = [champion] + ['*%d' % star] + items
        # l = [target_file, source_file, labels]
        l = [target_file, source_file, champion, '*%d' % star, items]
        data.append(l)

        # if not os.path.exists(target_path):
        Image.fromarray(img).save(target_path)
    return data


def create_dataframe(data):
    # columns = ['clipped', 'original', 'labels']
    columns = ['clipped', 'original', 'champion', 'star', 'item']
    df = pd.DataFrame(data, columns=columns).set_index('clipped')
    return df


def separate_data(data, dev_rate, test_rate):
    n_dev = int(len(data) * dev_rate)
    n_test = int(len(data) * test_rate)
    n_train = len(data) - n_dev - n_test

    all_indices = set(range(len(data)))
    dev_indices = set(random.sample(all_indices, n_dev))
    all_indices -= dev_indices
    test_indices = set(random.sample(all_indices, n_test))
    all_indices -= test_indices
    train_indices = all_indices

    train = [data[idx] for idx in train_indices]
    dev = [data[idx] for idx in dev_indices]
    test = [data[idx] for idx in test_indices]
    return train, dev, test

def main(args):
    os.makedirs(args.save_dir, exist_ok=True)
    champ_counts = defaultdict(int)
    data = []

    xml_paths = glob.glob(args.data_dir + '/*/*.xml')
    
    pbar = tqdm(total=len(xml_paths))
    for xml_path in xml_paths:
        img_path = '.'.join(xml_path.split('.')[:-1]) + '.jpg' 
        if not os.path.exists(img_path):
            img_path = '.'.join(xml_path.split('.')[:-1]) + '.png' 
            if not os.path.exists(img_path):
                sys.stderr.write("%s is not found.\n" % img_path)
                continue
        img = np.asarray(Image.open(img_path))
        xml = ET.parse(xml_path)
        data += clip(img, xml, champ_counts)
        pbar.update(1)

    train, dev, test = separate_data(data, args.dev_rate, args.test_rate)

    df_train = create_dataframe(train)
    df_dev = create_dataframe(dev)
    df_test = create_dataframe(test)

    with open(args.save_dir + '/train.csv', 'w') as f:
        print(df_train.to_csv(), file=f)

    with open(args.save_dir + '/dev.csv', 'w') as f:
        print(df_dev.to_csv(), file=f)

    with open(args.save_dir + '/test.csv', 'w') as f:
        print(df_test.to_csv(), file=f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='datasets/annotated_pics')
    parser.add_argument('--save-dir', default='datasets/clipped')
    parser.add_argument('--dev_rate', type=float, default=0.05)
    parser.add_argument('--test_rate', type=float, default=0.05)
    args = parser.parse_args()
    main(args)

