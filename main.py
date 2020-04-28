# coding: utf-8
import random
import argparse, os, sys, glob
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import datasets, layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def define_model(input_shape, output_size, dropout_rate=0.1):
    # via keras
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', 
                            input_shape=input_shape))

    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(output_size, activation='softmax'))

    model.summary()
    return model
    

def read_data(data_dir, classes, batch_size, img_height, img_width, shuffle=False):
    image_generator = ImageDataGenerator(rescale=1./255) 
    data_gen = image_generator.flow_from_directory(
        batch_size=batch_size,
        directory=data_dir,
        shuffle=shuffle,
        target_size=(img_height, img_width),
        classes=classes,
        class_mode='categorical')
    return data_gen

def plotImages(images, labels=None, save_as=None):
    fig, axes = plt.subplots(1, len(images))
    axes = axes.flatten()
    for i, (img, ax) in enumerate(zip(images, axes)):
        ax.set_title('aaa')
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
def main(args):
    sess = tf.InteractiveSession()
    random.seed(0)
    np.random.seed(0)
    tf.set_random_seed(0)
    n_train =  len(glob.glob('datasets/train/*/*'))
    n_dev =  len(glob.glob('datasets/dev/*/*'))
    n_test =  len(glob.glob('datasets/test/*/*'))

    classes = [path.split('/')[-1] for path in glob.glob('datasets/train/*')]
    train_data = read_data('datasets/train', classes, args.batch_size, 
                           args.img_height, args.img_width, shuffle=True)
    dev_data = read_data('datasets/dev', classes, args.batch_size, 
                         args.img_height, args.img_width, shuffle=False)
    test_data = read_data('datasets/test', classes, args.batch_size, 
                          args.img_height, args.img_width, shuffle=False)

    tr_img, tr_label = next(train_data)

    batch_size = args.batch_size
    input_shape = (args.img_height, args.img_width, 3)
    output_size = len(classes)

    model = define_model(input_shape, output_size)
    model.compile(optimizer='adam',
                  # loss='sparse_categorical_crossentropy',
                  loss='categorical_crossentropy',
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
        predictions = [classes[idx] for idx in outputs]
        ground_truths = [classes[idx] for idx in labels]
        titles = [title_template % (predictions[i], ground_truths[i]) for i in range(len(outputs))]
        plotImages(images, titles, save_as='test.png')
        break

if __name__ == "__main__":
      parser = argparse.ArgumentParser()
      parser.add_argument('model_root')
      parser.add_argument('--data-dir', default='datasets')
      parser.add_argument('--img-height', default=90)
      parser.add_argument('--img-width', default=75)
      parser.add_argument('--batch-size', default=4)
      parser.add_argument('--num-epochs', default=10)
      args = parser.parse_args()
      main(args)
