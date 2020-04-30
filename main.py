# coding: utf-8
import random
import argparse, os, sys, glob, math, subprocess
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

from dataset import read_data
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, Input

def define_model(input_shape, output_sizes, dropout_rate=0.1):
    '''
    <args>
    - input_shape: A tuple or list, the shape of input tensor (i.e., an image).
    - output_sizes: A dictionary of each size of the outputs keyed by the name of properties (e.g., {'champion': 30, 'items': 45})
    '''
    # via keras

    # model = models.Sequential()
    # model.add(layers.Conv2D(32, (3, 3), activation='relu', 
    #                         input_shape=input_shape))

    # model.add(layers.MaxPooling2D((2, 2)))
    # model.add(layers.Dropout(dropout_rate))
    # model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    # model.add(layers.MaxPooling2D((2, 2)))
    # model.add(layers.Dropout(dropout_rate))
    # model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    # model.add(layers.Flatten())
    # model.add(layers.Dense(64, activation='relu'))
    # model.add(layers.Dense(output_size, activation='softmax'))

    def cnn_layer(prev):
        conv = layers.Conv2D(32, (3, 3), activation='relu')(prev)
        pooling = layers.MaxPooling2D((2, 2))(conv)
        dropout = layers.Dropout(dropout_rate)(pooling)
        return dropout

    def output_layer(prev, output_name, output_size):
        dense = layers.Dense(64, activation='relu')(prev)
        output = layers.Dense(output_size, activation='softmax',
                              name=output_name)(dense)
        return output

    inputs = Input(shape = input_shape)
    layer1 = cnn_layer(inputs)
    layer2 = cnn_layer(layer1)
    flatten = layers.Flatten()(layer2)

    outputs = []
    for output_name, output_size in output_sizes.items():
        outputs.append(output_layer(flatten, output_name, output_size))

    model = Model(inputs=inputs, outputs=outputs)
    model.summary()
    return model
    

# (todo): 無限ループするカスタムジェネレータを作ってlabelimgでアノテーションしたマルチラベルに対応
def plotImages(images, labels=None, save_as=None, x=None, y=None):
    if type(images) == np.ndarray:
        images = [images[i] for i in range(images.shape[0])]
    if type(images) not in [list, tuple]:
        images = [images]
        if labels is not None and type(labels) not in [list, tuple]:
            labels = [labels]

    if not (x and y):
        n = math.sqrt(len(images))
        if n != int(n):
            n = int(n) + 1
        else:
            n = int(n)
        x = n
        y = n
    fig, axes = plt.subplots(y, x)
    
    if len(images) == 1:
        ax = axes
        ax.imshow(images[0])
        if labels is not None:
            ax.set_title(labels[0])
        ax.axis('off')

    else:
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

# https://www.tensorflow.org/tutorials/images/classification
# n_train = len(glob.glob(args.data_dir + '/train/*/*'))
# n_dev =  len(glob.glob(args.data_dir + '/dev/*/*'))
# n_test =  len(glob.glob(args.data_dir + '/test/*/*'))

# classes = [path.split('/')[-1] for path in glob.glob(args.data_dir + '/train/*')]
# train_data = read_data(args.data_dir + '/train', classes, args.batch_size, 
#                        args.img_height, args.img_width, shuffle=True)
# dev_data = read_data(args.data_dir + '/dev', classes, args.batch_size, 
#                      args.img_height, args.img_width, shuffle=False)
# test_data = read_data(args.data_dir + '/test', classes, args.batch_size, 
#                       args.img_height, args.img_width, shuffle=False)

class dotDict(dict):
  __getattr__ = dict.__getitem__
  __setattr__ = dict.__setitem__
  __delattr__ = dict.__delitem__

  def __getattr__(self, key):
    if key in self:
      return self[key]
    raise AttributeError("\'%s\' is not in %s" % (str(key), str(self.keys())))

def read_classes():
    classes = dotDict()
    for path in glob.glob('classes/*.txt'):
        label_type = path.split('/')[-1].split('.')[0]
        classes[label_type] = [l.strip() for l in open(path) if l.strip()]
    # classes.star = [l.strip() for l in open('classes/star.txt') if l.strip()]
    return classes


def main(args):
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
    test_data = read_data(args.data_dir, test_df, class2id, args.batch_size, 
                          args.img_height, args.img_width, 
                          y_col=args.label_type,
                          shuffle=False)
    
    id2class = [k for k in class2id]

    # for tr_d, tr_l in dev_data:
    #     images = tr_d
    #     # labels = [classes[i] for i in np.nonzero(tr_l)[1]]
    #     print(tr_l)
    #     print(id2class)
    #     exit(1)
    #     plotImages(images, labels)
    #     exit(1)

    # i = 0
    # for tr_d, tr_l in train_data:
    #     print(tr_l)
    #     i +=1 
    #     if i == 10:
    #         exit(1)
    # exit(1)

    batch_size = args.batch_size
    input_shape = (args.img_height, args.img_width, 3)

    
    # output_sizes = {
    #     'champion': len(classes.champion),
    #     'star': len(classes.star),
    # }
    output_sizes = {args.label_type: len(class2id)}
    model = define_model(input_shape, output_sizes, dropout_rate=args.dropout_rate)


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
    for images, labels in test_data:
        # outputs = model(images)
        outputs = sess.run(model(images)) # [batch_size, num_classes]
        outputs = np.argmax(outputs, axis=1)
        labels = np.argmax(labels, axis=1)
        predictions = [id2class[idx] for idx in outputs]
        ground_truths = [id2class[idx] for idx in labels]
        titles = [title_template % (predictions[i], ground_truths[i]) for i in range(len(outputs))]
        images = [images[i] for i in range(images.shape[0])]
        plotImages(images, titles, save_as='test.png')
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
      parser.add_argument('--batch-size', type=int, default=6)
      parser.add_argument('--num-epochs', type=int, default=20)
      parser.add_argument('--dropout-rate', type=float, default=0.25)
      args = parser.parse_args()
      main(args)
