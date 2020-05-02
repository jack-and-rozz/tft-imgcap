# coding: utf-8
import os, sys, glob, math, subprocess
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

class dotDict(dict):
  __getattr__ = dict.__getitem__
  __setattr__ = dict.__setitem__
  __delattr__ = dict.__delitem__

  def __getattr__(self, key):
    if key in self:
      return self[key]
    raise AttributeError("\'%s\' is not in %s" % (str(key), str(self.keys())))

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

