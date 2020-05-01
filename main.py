# coding: utf-8
import random
import argparse, os, sys, glob, math, subprocess, yaml
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras import datasets, layers, models
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, Input

from dataset import read_data
from model import define_model
from util import plotImages, dotDict


def read_classes():
    classes = dotDict()
    for path in glob.glob('classes/*.txt'):
        label_type = path.split('/')[-1].split('.')[0]
        classes[label_type] = [l.strip() for l in open(path) if l.strip()]
    # classes.star = [l.strip() for l in open('classes/star.txt') if l.strip()]
    return classes


def save_args(args, argfile='config.yaml'):
    config_path = args.model_root + '/' + argfile
    with open(config_path, 'w') as f:
        f.write(yaml.dump(args.__dict__) + '\n')

        # if type(args) == recDotDict:
        #     f.write(yaml.dump(dict(args)) + '\n')
        # else:
        #     f.write(yaml.dump(args.__dict__) + '\n')

def make_model_dirs(args):
    model_root = args.model_root
    os.makedirs(model_root + '/checkpoints', exist_ok=True)
    os.makedirs(model_root + '/evaluations', exist_ok=True)
    save_args(args)

def main(args):
    make_model_dirs(args)
    os.environ['PYTHONHASHSEED'] = '0'
    random.seed(0)
    np.random.seed(0)
    tf.set_random_seed(0)
    sess = tf.InteractiveSession()


    train_df =  pd.read_csv(args.data_dir + '/train.csv')
    dev_df =  pd.read_csv(args.data_dir + '/dev.csv')
    test_df =  pd.read_csv(args.data_dir + '/test.csv')
    n_train = len(train_df)
    n_dev = len(dev_df)
    n_test = len(test_df)

    # id2class = read_classes()[args.label_type]
    # class2id = {k:i for i,k in enumerate(id2class)}
    class2id = None
    train_data = read_data(args.data_dir, train_df, class2id, args.batch_size, 
                           args.img_height, args.img_width, 
                           y_col=args.label_type,
                           shuffle=True)
    class2id = train_data.class_indices
    dev_data = read_data(args.data_dir, dev_df, class2id, args.batch_size, 
                         args.img_height, args.img_width, 
                         y_col=args.label_type,
                         shuffle=False)

    test_batch_size = 9 # To display 3x3 images in a test output.
    test_data = read_data(args.data_dir, test_df, class2id, test_batch_size, 
                          args.img_height, args.img_width, 
                          y_col=args.label_type,
                          shuffle=False)
    
    id2class = [k for k in class2id]

    # DEBUG
    # for tr_d, tr_l in dev_data:
    #     images = tr_d
    #     # labels = [classes[i] for i in np.nonzero(tr_l)[1]]
    #     print(tr_l)
    #     print(id2class)
    #     exit(1)
    #     plotImages(images, labels)
    #     exit(1)

    batch_size = args.batch_size
    input_shape = (args.img_height, args.img_width, 3)

    
    # output_sizes = {
    #     'champion': len(classes.champion),
    #     'star': len(classes.star),
    # }
    output_sizes = {args.label_type: len(class2id)}
    model = define_model(input_shape, output_sizes, 
                         cnn_dims=args.cnn_dims,
                         dropout_rate=args.dropout_rate)


    # https://www.pyimagesearch.com/2018/12/24/how-to-use-keras-fit-and-fit_generator-a-hands-on-tutorial/
    # Multi-output にするならkeras.train_on_batchを使ったほうがいい？

    # loss_type = 'sparse_categorical_crossentropy'
    loss_type = 'categorical_crossentropy'
    model.compile(optimizer='adam',
                  loss={
                      args.label_type: loss_type,
                  },
                  loss_weights={args.label_type: 1.0},
                  metrics=['accuracy'])
    model.summary()
    history = model.fit_generator(
        train_data,
        steps_per_epoch=n_train // args.batch_size,
        epochs=args.num_epochs,
        validation_data=dev_data,
        validation_steps=n_dev // args.batch_size
    )

    title_template = "Hyp: %s\n Ref: %s"
    for pic_idx, (images, labels) in enumerate(test_data):
        outputs = sess.run(model(images)) # [batch_size, num_classes]
        labels = np.argmax(labels, axis=1)
        # show top-1
        outputs = np.argmax(outputs, axis=1)
        predictions = [id2class[idx] for idx in outputs]

        # show top-3
        # outputs = (-outputs).argsort(axis=-1)[:, :3]
        # predictions = [', '.join([id2class[idx] for idx in idx_list]) for idx_list in outputs]
        ground_truths = [id2class[idx] for idx in labels]
        titles = [title_template % (predictions[i], ground_truths[i]) for i in range(len(outputs))]
        images = [images[i] for i in range(images.shape[0])]
        evaluation_path = args.model_root + '/evaluations/test.%02d.png' % pic_idx
        plotImages(images, titles, save_as=evaluation_path)
        if test_batch_size * (pic_idx + 1) >= n_test:
            break

if __name__ == "__main__":
      parser = argparse.ArgumentParser()
      parser.add_argument('model_root')
      parser.add_argument('--label-type', default='champion', 
                          choices=['champion', 'star', 'item'])
      parser.add_argument('--data-dir', default='datasets/clipped')
      # parser.add_argument('--img-height', default=90)
      # parser.add_argument('--img-width', default=75)
      parser.add_argument('--img-height', type=int, default=100)
      parser.add_argument('--img-width', type=int, default=80)
      parser.add_argument('--batch-size', type=int, default=9)
      parser.add_argument('--num-epochs', type=int, default=20)
      parser.add_argument('--cnn-dims', type=list, 
                          default=[32, 32], nargs='+')
      parser.add_argument('--dropout-rate', type=float, default=0.25)
      args = parser.parse_args()
      main(args)

# https://www.tensorflow.org/tutorials/images/classification
