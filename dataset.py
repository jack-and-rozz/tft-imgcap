
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from keras_preprocessing.image import ImageDataGenerator
import pandas as pd



def read_data(data_dir, df, classes, batch_size, img_height, img_width, 
              shuffle=False, x_col='clipped', y_col='champion', seed=0):
    image_generator = ImageDataGenerator(rescale=1./255) 

    data_dir = os.getcwd() + '/' + data_dir
    data_gen = image_generator.flow_from_dataframe(
        df, directory=data_dir,
        x_col=x_col,
        y_col=y_col,
        shuffle=shuffle, 
        target_size=(img_height, img_width),
        classes=classes,
        class_mode='categorical',
        batch_size=batch_size,
        seed=seed,
    )
    return data_gen 

# flow_from_dataframe(dataframe， directory=None， x_col='filename'， y_col='class'， target_size=(256， 256)， color_mode='rgb'， classes=None， class_mode='categorical'， batch_size=32， shuffle=True， seed=None， save_to_dir=None， save_prefix=''， save_format='png'， subset=None， interpolation='nearest'， drop_duplicates=True)

# def read_data(data_dir, classes, batch_size, img_height, img_width, shuffle=False):
#     image_generator = ImageDataGenerator(rescale=1./255) 
#     data_gen = image_generator.flow_from_directory(
#         batch_size=batch_size,
#         directory=data_dir,
#         shuffle=shuffle,
#         target_size=(img_height, img_width),
#         classes=classes,
#         class_mode='categorical')
#     return data_gen
