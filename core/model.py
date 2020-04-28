# coding: utf-8 
import sys, random
import numpy as np
from pprint import pprint
import tensorflow as tf
from core.utils.common import dotDict

class ModelBase():
    def __init__(self, sess, config, vocab):
        self.sess = sess
        self.config = config
        self.vocab = vocab


class CNNClassifier(ModelBase):
    def __init__(self, sess, config, vocab):
        super(CNNClassifier, self).__init__(sess, config, vocab)
        self.ph = self.setup_placeholder(config)
        self.outputs = self.inference(self.ph)

    def inference(self, ph):
        print(ph)
        pass

    def setup_placeholder(self, config):
        batch_size = None
        height = config.img_height
        width = config.img_width
        rgb=3

        with tf.name_scope('Placeholder'):
            ph = dotDict()
            ph.is_training = tf.placeholder(tf.bool, name='is_training', shape=[]) 
            ph.images = tf.placeholder(name='images', dtype=tf.int32, 
                                      shape=[batch_size, height, width, rgb])
            ph.labels = tf.placeholder(name='labels', dtype=tf.int32, 
                                       shape=[batch_size])
        return ph



    def get_input_feed(self, batch, is_training):
        ph = self.ph
        input_feed = {}
        input_feed[self.is_training] = is_training
        for key in ph:
            if key in batch and len(batch[key]):
                input_feed[ph[key]] = batch[key]
        return input_feed
