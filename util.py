# coding: utf-8
import os, sys, glob, math, subprocess, re
import matplotlib.pyplot as plt
import numpy as np
from itertools import chain
import yaml
from collections import defaultdict

class dotDict(dict):
  __getattr__ = dict.__getitem__
  __setattr__ = dict.__setitem__
  __delattr__ = dict.__delitem__

  def __getattr__(self, key):
    if key in self:
      return self[key]
    raise AttributeError("\'%s\' is not in %s" % (str(key), str(self.keys())))


# collections.defaultdict returns None when dict.get('unknown_key') although dict['unknown_key'] returns a correct initial value....
class GettableDefaultDictWrapper(dict):
    def __init__(self, dic):
        assert isinstance(dic, defaultdict)
        self.dic = dic
    def __len__(self):
        return len(self.dic)

    def __getattr__(self, name):
        return getattr(dic, name)

    def __setitem__(self, key, value):
        self.dic[key] = value

    def __delitem__(self, key):
        del self.dic[key]

    def __getitem__(self, key):
        return self.dic[key]

    def get(self, key):
        return self.dic[key]


def flatten(l):
  return list(chain.from_iterable(l))


# (todo): 無限ループするカスタムジェネレータを作ってlabelimgでアノテーションしたマルチラベルに対応
def plotImages(images, labels=None, save_as=None, x=None, y=None, 
               figsize=(6.4, 4,8)):
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

    # fig = plt.figure(figsize)

    # axes = fig.axes
    fig, axes = plt.subplots(y, x, figsize=figsize)

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


def parse_epoch_and_loss_from_path(path):
    pattern = "ckpt.([0-9]+)-([0-9\.]+)\.hdf5"
    m = re.search(pattern, path.split('/')[-1])
    if not m:
        return None, None

    epoch, loss = m.groups()
    return int(epoch), float(loss)

def get_best_and_final_model_path(model_root):
    ckpt_dir = model_root + '/checkpoints'
    checkpoint_paths = glob.glob(ckpt_dir + '/*')
    if len(checkpoint_paths) == 0:
        return None, None
    val_losses = []
    for path in checkpoint_paths:
        epoch, loss = parse_epoch_and_loss_from_path(path)
        val_losses += [(epoch, loss, path)]
    val_losses = list(sorted(val_losses, key=lambda x: (x[1], -x[0])))
    best_model_path = val_losses[0][-1]
    val_losses = list(sorted(val_losses, key=lambda x: -x[0]))
    final_model_path = val_losses[0][-1]
    return best_model_path, final_model_path


def load_model_config(model_root, config_filename='config.yaml'):
    saved_args = yaml.load(open(model_root + '/' + config_filename))
    return dotDict(saved_args)
