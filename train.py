# coding: utf-8
import argparse, os, sys, glob, math, subprocess, yaml, random, re
import numpy as np
import tensorflow as tf
import pandas as pd
from collections import Counter, defaultdict
from pprint import pprint
import matplotlib.pyplot as plt

from tensorflow.keras import datasets, layers, models
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler

from test import evaluation, test_batch_size
from dataset import read_data
from model import define_model
from util import plotImages, dotDict, flatten, get_best_and_final_model_path, parse_epoch_and_loss_from_path
from option import get_train_parser

def get_class_weight(file_path, key, class2id):
    valid_classes = set([l.strip() for l in open('classes/%s.txt' % key)])
    df = pd.read_csv(file_path)
    data = df[key].tolist()

    if type(data[0]) == list:
        data = flatten(data)
    
    hist = Counter(data)
    weights = defaultdict(float)
    for k in hist:
        weights[class2id[k]] = 1.0 / hist[k] if k in valid_classes else 0.
    return weights

def save_args(args, argfile='config.yaml'):
    config_path = args.model_root + '/' + argfile
    with open(config_path, 'w') as f:
        f.write(yaml.dump(args.__dict__) + '\n')

def make_model_dirs(args):
    model_root = args.model_root
    os.makedirs(model_root + '/checkpoints', exist_ok=True)
    os.makedirs(model_root + '/evaluations', exist_ok=True)
    save_args(args)


def fix_random_seed():
    # Fix random seeds.
    os.environ['PYTHONHASHSEED'] = '0'
    random.seed(0)
    np.random.seed(0)
    tf.set_random_seed(0)

def fixed_decay_scheduler(init_lr, decay_rate, final_lr=None):
    def schedule(epoch):
        lr = init_lr * (decay_rate ** epoch)
        if final_lr:
            lr = max(lr, final_lr)
        return lr
    return schedule


def save_classes(model_root, class2id):
    with open(model_root + '/classes.txt', 'w') as f:
        for c in class2id:
            print(c, file=f)

def main(args):
    sess = tf.InteractiveSession()
    make_model_dirs(args)
    fix_random_seed()

    train_df =  pd.read_csv(args.data_dir + '/train.csv')
    dev_df =  pd.read_csv(args.data_dir + '/dev.csv')
    test_df =  pd.read_csv(args.data_dir + '/test.csv')
    if args.label_type == 'item':
        train_df = train_df.where(df['champion'] != 'items')
        dev_df = train_df.where(df['champion'] != 'items')
        test_df = train_df.where(df['champion'] != 'items')

    n_train = len(train_df)
    n_dev = len(dev_df)
    n_test = len(test_df)
    class2id = None
    # PCACA: https://qiita.com/koshian2/items/78de8ccd09dd2998ddfc
    train_data = read_data(args.data_dir, train_df, class2id, args.batch_size, 
                           args.img_height, args.img_width, 
                           y_col=args.label_type,
                           shuffle=True)
    # class2id = train_data.class_indices
    # print(class2id)
    # id2class = [k for k in class2id]
    for img, lb in train_data:
        print(lb)
        exit(1)

    dev_data = read_data(args.data_dir, dev_df, class2id, args.batch_size, 
                         args.img_height, args.img_width, 
                         y_col=args.label_type,
                         shuffle=False)

    test_data = read_data(args.data_dir, test_df, class2id, test_batch_size, 
                          args.img_height, args.img_width, 
                          y_col=args.label_type,
                          shuffle=False)

    class_weight = get_class_weight(args.data_dir + '/train.csv', args.label_type, class2id) # Loss weights to handle imbalance classes.
    save_classes(args.model_root, class2id)

    input_shape = (args.img_height, args.img_width, 3)
    output_sizes = {args.label_type: len(class2id)}

    _, final_model_path = get_best_and_final_model_path(args.model_root)
    if final_model_path:
        model = load_model(final_model_path)
        initial_epoch, _ = parse_epoch_and_loss_from_path(final_model_path)
    else:
        model = define_model(input_shape, output_sizes, 
                             cnn_dims=args.cnn_dims,
                             dropout_rate=args.dropout_rate)
        initial_epoch = 0
    sys.stderr.write('Start training from Epoch %d.\n' % initial_epoch)

    # Multi-output にするなら自分でスケジューリングしてkeras.train_on_batchを使ったほうがいい？
    # https://www.pyimagesearch.com/2018/12/24/how-to-use-keras-fit-and-fit_generator-a-hands-on-tutorial/
    modelCheckpoint = ModelCheckpoint(
        filepath = args.model_root + '/checkpoints/ckpt.{epoch:02d}-{val_loss:.2f}.hdf5',
        monitor='val_loss',
        verbose=1,
        save_best_only=False,
        save_weights_only=False,
        mode='min',
        period=1)

    loss_type = 'categorical_crossentropy'

    opt = Adam(lr=args.init_lr)
    model.compile(optimizer=opt,
                  loss={
                      args.label_type: loss_type,
                  },
                  loss_weights={args.label_type: 1.0},
                  metrics=['accuracy'])
    model.summary()

    schedule = fixed_decay_scheduler(args.init_lr, 
                                     decay_rate=args.lr_decay_rate, 
                                     final_lr=args.final_lr)
    lr_decay = LearningRateScheduler(schedule)
    kwargs = {
        'steps_per_epoch': n_train // args.batch_size,
        'epochs':args.num_epochs,
        'validation_data': dev_data,
        'validation_steps' :n_dev // args.batch_size,
        'callbacks' :[modelCheckpoint, lr_decay],
        'initial_epoch': initial_epoch
    }
    if args.enable_class_weight:
        kwargs['class_weight'] = class_weight

    history = model.fit_generator(train_data, **kwargs)

    # TODO: save/load the models 
    # https://qiita.com/tom_eng_ltd/items/7ae0814c2d133431c84a

    best_model_path, _ = get_best_and_final_model_path(args.model_root)
    model = load_model(best_model_path)
    output_dir = args.model_root + '/evaluations'
    evaluation(model, output_dir, test_data, id2class, n_test)

if __name__ == "__main__":
    parser = get_train_parser()
    args = parser.parse_args()
    main(args)

# https://www.tensorflow.org/tutorials/images/classification
