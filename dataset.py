
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from collections import defaultdict
from keras_preprocessing.image import ImageDataGenerator
from keras_preprocessing.image.dataframe_iterator import DataFrameIterator
import pandas as pd
import numpy as np
from util import dotDict

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


def load_classes_from_definition(label_types):
    def _load_classes_from_definition(label_type):
        class_def = "classes/%s.txt" % label_type
        _id2class = [c.strip() for c in open(class_def)]
        _class2id = defaultdict(int)
        for i, c in enumerate(_id2class):
            _class2id[c] = i
        return _id2class, _class2id

    id2class = dotDict()
    class2id = dotDict()
    for label_type in label_types:
        _id2class, _class2id = _load_classes_from_definition(label_type)
        id2class[label_type] = _id2class
        class2id[label_type] = _class2id
    return id2class, class2id


# # Not used for now.
class MultiOutputIterator(object):
    def __init__(self, data_gen, classes, y_cols):
        self._data_gen = data_gen
        self.class2id = classes
        self.id2class = [tok for tok in classes]
        self.y_cols = y_cols

    def __iter__(self):
        return self

    def __getattr__(self, name):
        # return self._data_gen.__getattr__(name)
        return getattr(self._data_gen, name)

    def __next__(self):
        data, labels = self._data_gen.__next__()
        assert len(labels) == len(self.y_cols)
        label_indice = []
        for i in range(len(labels)):
            # print(self.y_cols[i])
            # print(self.class2id[self.y_cols[i]])
            # print(labels[i])
            print(set(labels[i]) - set(self.class2id[self.y_cols[i]].keys()))
            
            indice = np.vectorize(self.class2id[self.y_cols[i]].get)(labels[i])
            label_indice.append(indice)
        return data, label_indice


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

    ### DEBUG: single output
    # class_mode = 'sparse'    
    # classes = classes[y_col[0]]
    # y_col = y_col[0]
    ###### ### ### ### ### 


    ### DEBUG: multi output
    class_mode = 'multi_output'
    if type(y_col) != list:
        y_col=[y_col]
    ########################

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
    # # print(data_gen.__class__)
    # # exit(1)
    # data_gen.__next__ = lambda : 1
    # print(data_gen.next())
    # print(next(data_gen))
    # # print(dir(data_gen))
    # # print(type(data_gen))
    # print(data_gen.shape)
    # exit(1)
    data_gen = MultiOutputIterator(_data_gen, classes, y_col)

    # keras_preprocessing requires this wrapper to convert the iterator to a generator for some reason?
    # https://github.com/keras-team/keras-preprocessing/issues/212
    def wrapper_gen(x): 
        yield from x
    data_gen = wrapper_gen(data_gen)
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
