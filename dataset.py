
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from collections import defaultdict
from keras_preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np

# # Not used for now.
# class MultiCategoryIterator(object):
#     def __init__(self, data_gen, class_list):
#         self._data_gen = data_gen
#         self.class_list = class_list
#         self.rev_class_list = [{tok:i for i, tok in enumerate(l)} for l in class_list]
#     def __iter__(self):
#         return self

#     def __getattr__(self, name):
#         return self._data_gen.__getattr__(name)

#     def __next__(self):
#         data, labels = self._data_gen.__next__()
#         print(labels)
#         for i in range(len(labels)):
#             labels[i] = np.array([self.rev_class_list[i][label] for label in labels[i]], dtype=np.int32)
#         # print(labels)
#         # exit(1)
#         return data, labels

def read_df(path, label_type, class2id=None):
    df = pd.read_csv(path).fillna('-')

    # if label_type == 'item':  # Remove items?
    #     df = df[df['champion'] != 'items']

    if class2id is not None:
        pass
        # print(np.all(df['item1']) in class2id)
        # print(np.any(df['item1']) in class2id)
        # exit(1)
        # if label_type == 'item':
        #     df = df[df['item1'] in class2id][df['item2'] in class2id][df['item3'] in class2id]
        # else:
        #     df = df[df[label_type] in class2id]

    return df

def load_classes_from_definition(label_type):
    class_def = "classes/%s.txt" % label_type
    id2class = [c.strip() for c in open(class_def)]
    class2id = defaultdict(int)
    for i, c in enumerate(id2class):
        class2id[c] = i
    return id2class, class2id

# Not used for now.
class MultiOutputIterator(object):
    def __init__(self, data_gen, classes):
        self._data_gen = data_gen
        self.class2id = classes
        self.id2class = [tok for tok in classes]
    def __iter__(self):
        return self

    def __getattr__(self, name):
        return self._data_gen.__getattr__(name)

    def __next__(self):
        data, labels = self._data_gen.__next__()
        # for c in labels
        print(labels)
        labels = [self._data_gen.class_indices[c] for c in labels]
        print(labels)
        exit(1)
        # labels = [for]
        for i in range(len(labels)):
            print(labels[i])

            # labels[i] = np.array([self.class_list[i][label] for label in labels[i]], dtype=np.int32)
        # print(labels)
        # exit(1)
        return data, labels


def read_data(data_dir, df, classes, batch_size, img_height, img_width, 
              shuffle=False, x_col='clipped', y_col='champion', seed=0):

    if shuffle == True :
        # for data augmentation.
        image_generator = ImageDataGenerator(rescale=1./255, 
                                             rotation_range=5,
                                             width_shift_range=.10,
                                             height_shift_range=.10,
                                             zoom_range=0.1) 
    else:
        image_generator = ImageDataGenerator(rescale=1./255) 

    data_dir = os.getcwd() + '/' + data_dir
    class_mode = 'sparse'
    class_mode = 'multi_output'

    if type(y_col) != list:
        y_col=[y_col]

    _data_gen = image_generator.flow_from_dataframe(
        df, directory=data_dir,
        x_col=x_col,
        y_col=y_col,
        shuffle=shuffle, 
        classes=classes,
        target_size=(img_height, img_width),
        class_mode=class_mode,
        batch_size=batch_size,
        seed=seed,
    )
    data_gen = _data_gen
    return data_gen 

# When using only one label as the target.
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
