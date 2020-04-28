# coding: utf-8

import argparse, os, sys
import numpy as np
import tensorflow as tf
import glob

from core.utils.common import dotDict, recDotDict
from core.dataset import read_tensor_from_image_file
from core.model import CNNClassifier
from core.vocabulary import LabelVocabulary

from tensorflow.keras import datasets, layers, models

def read_data(paths, vocab=None):
    data = np.empty([len(paths), args.img_height, args.img_width, 3], 
                          dtype=float)
    labels = []
    for i, path in enumerate(paths):
        label = path.split('/')[-2]
        tensor = read_tensor_from_image_file(path, 
                                             input_height=args.img_height,
                                             input_width=args.img_width)
        data[i] = tensor
        labels.append(label)
    if not vocab:
        vocab = LabelVocabulary(labels)
    labels =  np.array(vocab.tokens2ids(labels), dtype=np.int32)

    return data, labels, vocab


def define_model(input_shape, output_labels):
    # via keras
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    model.summary()
    return model
    
def main(args):
    os.makedirs(args.model_root, exist_ok=True)
    config = args
    vocab = dotDict()
    model = dotDict()
    train_data, train_labels, champion_vocab = read_data(
        glob.glob(args.data_dir + '/train/*/*'))
    test_data, test_labels, _ = read_data(
        glob.glob(args.data_dir + '/test/*/*'), vocab=champion_vocab)

    vocab.champion = champion_vocab

    with tf.compat.v1.Session() as sess:
        with tf.variable_scope('Champion'):
            #model.champion = CNNClassifier(sess, config, vocab)

        exit(1)
        while True:
            batch = dotDict()
            batch.images = train_data
            batch.labels = train_labels
            input_feed = model.get_input_feed(batch, is_training)


# def main2(args):
#     input_shape = (args.img_height, args.img_width, 3)
#     output_size = vocab.champion.size
#     model.champion = define_model(input_shape, output_size)
#     model.champion.compile(optimizer='adam',
#                            loss='sparse_categorical_crossentropy',
#                            metrics=['accuracy'])
#     print(dir(model.champion))


if __name__ == "__main__":
      parser = argparse.ArgumentParser()
      parser.add_argument('model_root')
      parser.add_argument('--data-dir', default='datasets')
      parser.add_argument('--img-height', default=90)
      parser.add_argument('--img-width', default=75)
      args = parser.parse_args()
      main(args)
