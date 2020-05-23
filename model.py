# coding: utf-8
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, regularizers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, Input


def define_model(input_shape, output_sizes, 
                 cnn_dims=[32, 32], 
                 dropout_rate=0.1,
                 L2reg_factor=0.01,
                 activation='relu'):
    '''
    <args>
    - input_shape: A tuple or list, the shape of input tensor (i.e., an image).
    - output_sizes: A dictionary of each size of the outputs keyed by the name of properties (e.g., {'champion': 30, 'item': 45}).
    - cnn_dims: The numbers of dimensions in each CNN layer. CNN is applied as many times as the length of cnn_dims.
    '''

    def cnn_layer(prev, ndim):
        kernel_regularizer = regularizers.l2(L2reg_factor) if L2reg_factor > 0 else None
        bias_regularizer = regularizers.l2(L2reg_factor) if L2reg_factor > 0 else None
        conv = layers.Conv2D(
            ndim, (3, 3), 
            use_bias=True,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer, 
        )(prev)
        pooling = layers.MaxPooling2D((2, 2))(conv)
        dropout = layers.Dropout(dropout_rate)(pooling)
        return dropout

    def output_layer(prev, output_name, output_size, final_activation='softmax'):
        dense = layers.Dense(cnn_dims[-1], activation=activation, 
                             use_bias=True)(prev)
        output = layers.Dense(output_size, activation=final_activation,
                              name=output_name, use_bias=True)(dense)
        return output

    inputs = Input(shape = input_shape)
    hidden = inputs
    for ndim in cnn_dims:
        hidden = cnn_layer(hidden, ndim)
    flatten = layers.Flatten()(hidden)

    outputs = []
    for output_name, output_size in output_sizes.items():
        final_activation = 'softmax' if output_name != 'item' else 'relu'
        outputs.append(output_layer(flatten, output_name, output_size, 
                                    final_activation=final_activation))

    model = Model(inputs=inputs, outputs=outputs)
    return model
