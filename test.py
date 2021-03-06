# coding: utf-8
import argparse, os, sys, glob, math, subprocess, yaml, random, re
import numpy as np
import tensorflow as tf
import pandas as pd
from tqdm import tqdm
from collections import Counter

from tensorflow.keras.models import load_model

from dataset import read_df, read_data, load_classes_from_definition
from model import define_model
from util import plotImages, dotDict, flatten, get_best_and_final_model_path
from option import get_test_parser, merge_with_saved_args

# for analysis
import matplotlib.pyplot as plt
import seaborn as sns
# from sklearn.metrics import confusion_matrix

# To display 3x3 images in a test output.
test_batch_size = 16

def load_classes_from_saved_model(model_root):
    id2class = [c.strip() for c in open(model_root + '/classes.txt')]
    class2id = {c:i for i,c in enumerate(id2class)}
    return id2class, class2id

def evaluation(model, output_dir, test_data, id2class, n_test=None):
    os.makedirs(output_dir, exist_ok=True)

    # all_hypotheses = {k:[] for k in id2class}
    # all_references = {k:[] for k in id2class}
    all_hypotheses = np.empty((0, len(id2class)))
    all_references = np.empty((0, len(id2class)))
    pbar = tqdm(total=n_test)

    def make_title(hypothesis, reference):
        title_template = "Hyp: %s\nRef: %s"
        hyp_str = ', '.join(hypothesis)
        ref_str = ', '.join(reference)
        return title_template % (hyp_str, ref_str)



    for pic_idx, (images, labels) in enumerate(test_data):
        # outputs = sess.run(model(images)) # [batch_size, num_classes]
        outputs = model.predict(images)
        # labels = np.argmax(labels, axis=1) # when reading in 'categorical' mode.

        # show top-1
        top1_outputs = [np.argmax(o, axis=1) for o in outputs] # [num_labels, num_examples]

        hypotheses = [np.array(dic)[o] for o, dic in zip(top1_outputs, id2class.values())] # hypotheses[label_type][example_idx] = val
        references = [np.array(dic)[l.astype(np.int32)] for l, dic in zip(labels, id2class.values())]

        hypotheses = np.concatenate(np.expand_dims(hypotheses, 0), axis=0).T
        references = np.concatenate(np.expand_dims(references, 0), axis=0).T

        #np.append(all_hypotheses, hypotheses, axis=0)
        #np.append(all_references, references, axis=0)

        all_hypotheses = np.vstack([all_hypotheses, hypotheses])
        all_references = np.vstack([all_references, references])

        # for i, k in enumerate(id2class.keys()):
        #     all_hypotheses[k].append(hypotheses)
        #     all_references[k].append(references)

        # print(hypotheses.T)
        # print(references.T)

        # titles = [title_template % (hypotheses[i], references[i]) for i in range(len(outputs))]
        titles = [make_title(h, r) for h, r in zip(hypotheses, 
                                                   references)]
        images = [images[i] for i in range(images.shape[0])]
        evaluation_path = output_dir + '/test.%02d.png' % pic_idx

        plotImages(images, labels=titles, figsize=(9.6*1.2, 7.2), 
                   save_as=evaluation_path)
        pbar.update(test_batch_size)

        # One-shot iterator is not available?
        if test_batch_size * (pic_idx + 1) >= n_test: 
            break


    def calc_accuracy(_hypotheses, _references):
        is_correct = [1 if hyp == ref else 0 for hyp, ref in zip(_hypotheses, 
                                                                 _references)]
        accuracy = float(sum(is_correct)) / len(is_correct)
        return accuracy


    print()
    for i, k in enumerate(id2class.keys()):
        acc = calc_accuracy(all_hypotheses[:, i], all_references[:, i])
        print('Accuracy (%s): %.03f' % (k, acc))

    print(file=sys.stderr)
    print("Evaluation results are saved to '%s'." % (output_dir), file=sys.stderr)

    return all_references, all_hypotheses


def confusion_matrix(refs, hyps, class2id):
    matrix = np.zeros((len(class2id), len(class2id)))
    for r, h in zip(refs, hyps):
        matrix[class2id[r]][class2id[h]] += 1
    matrix /= (np.sum(matrix, axis=1, keepdims=True) + 1e-9)
    return matrix


def plot_eval_stat(refs, hyps, class2id, id2class, output_dir):

    with sns.axes_style("darkgrid"):
        plt.subplots(figsize=(9.6, 7.2), tight_layout=True)
        cm = confusion_matrix(refs, hyps, class2id)
        annot = False
        sns.heatmap(cm, xticklabels=id2class, yticklabels=id2class, 
                    cmap="YlGnBu", annot=annot, vmin=0., vmax=1.)
        # sns.heatmap(cm, xticklabels=1, yticklabels=1, cmap="RdBu_r", annot=True)
        plt.xlabel('Hypothesis')
        plt.ylabel('Reference')
        plt.savefig(output_dir + '/cm.pdf')
        # plt.show()

    # with sns.axes_style("darkgrid"):
    #     # fig, axes = plt.subplots(ncols=2, sharey='all', figsize=(20, 8))
    #     fig, axes = plt.subplots(1, 1, figsize=(5, 10), tight_layout=True)
    #     plt.subplots_adjust(wspace=0.2, hspace=0.2)
  
    #     # ax = axes[0][0]
    #     ax = axes
    #     ax.barh(y, x1)
    #     ax.barh(y, x2, left=x1)
    #     ax.tick_params(labelrotation=0)
    #     ax.legend(['TP', 'FP'], loc='lower right')

    #     plt.show()

def main(args):
    # # DEBUG
    # id2class, class2id = load_classes_from_definition(args.label_types)
    # refs = [id2class[LABEL_TYPE][random.randint(0, 30)] for i in range(100)]
    # hyps = [id2class[LABEL_TYPE][random.randint(0, 30)] for i in range(100)]
    # output_dir = args.model_root + '/evaluations' if not args.output_dir else args.output_dir

    # plot_eval_stat(refs, hyps, class2id[LABEL_TYPE], id2class[LABEL_TYPE], output_dir)
    # exit(1)
    sess = tf.InteractiveSession()
    # id2class, class2id = load_classes_from_saved_model(args.model_root)
    id2class, class2id = load_classes_from_definition(args.label_types)
    test_df = read_df(args.data_dir + '/' + args.test_csv, args.label_types)

    n_test = len(test_df)
    test_data = read_data(args.data_dir, test_df, class2id, test_batch_size, 
                          args.img_height, args.img_width, 
                          y_col=args.label_types,
                          shuffle=False)

    best_model_path, _ = get_best_and_final_model_path(args.model_root)
    model = load_model(best_model_path)
    output_dir = args.model_root + '/evaluations' if not args.output_dir else args.output_dir
    refs, hyps = evaluation(model, output_dir, test_data, 
                            id2class, n_test)
    plot_eval_stat(refs[:, 0], hyps[:, 0], class2id[LABEL_TYPE], id2class[LABEL_TYPE], output_dir)
    
if __name__ == "__main__":
    parser = get_test_parser()
    args = parser.parse_args()
    args = merge_with_saved_args(args)

    global LABEL_TYPE
    LABEL_TYPE = args.label_types[0]
    main(args)
