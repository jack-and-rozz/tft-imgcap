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
from option import get_test_parser

# To display 3x3 images in a test output.
test_batch_size = 9

def load_classes_from_saved_model(model_root):
    id2class = [c.strip() for c in open(model_root + '/classes.txt')]
    class2id = {c:i for i,c in enumerate(id2class)}
    return id2class, class2id

def evaluation(model, output_dir, test_data, id2class, n_test=None):
    os.makedirs(output_dir, exist_ok=True)

    title_template = "Hyp: %s\nRef: %s"
    all_hypotheses = []
    all_references = []

    pbar = tqdm(total=n_test)
    for pic_idx, (images, labels) in enumerate(test_data):
        # outputs = sess.run(model(images)) # [batch_size, num_classes]
        outputs = model.predict(images)
        # labels = np.argmax(labels, axis=1) # when reading in 'categorical' mode.

        # show top-1
        outputs = np.argmax(outputs, axis=1)
        hypotheses = [id2class[idx] for idx in outputs]

        # show top-3
        # outputs = (-outputs).argsort(axis=-1)[:, :3]
        # hypotheses = [', '.join([id2class[idx] for idx in idx_list]) for idx_list in outputs]
        references = [id2class[idx] for idx in labels.astype(np.int32)]

        all_hypotheses += hypotheses
        all_references += references

        titles = [title_template % (hypotheses[i], references[i]) for i in range(len(outputs))]
        images = [images[i] for i in range(images.shape[0])]
        evaluation_path = output_dir + '/test.%02d.png' % pic_idx
        plotImages(images, titles, save_as=evaluation_path)
        pbar.update(test_batch_size)
        if test_batch_size * (pic_idx + 1) >= n_test:
            break

    is_correct = [1 if hyp == ref else 0 for hyp, ref in zip(all_hypotheses[:n_test], all_references[:n_test])]
    accuracy = float(sum(is_correct)) / len(is_correct)

    print('Accuracy: %.03f' % accuracy)
    print(file=sys.stderr)
    print("Evaluation results are saved to '%s'." % (output_dir), file=sys.stderr)



def main(args):
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
    evaluation(model, output_dir, test_data, id2class[LABEL_TYPE], n_test)

if __name__ == "__main__":
    parser = get_test_parser()
    args = parser.parse_args()

    global LABEL_TYPE
    LABEL_TYPE = args.label_types[0]
    main(args)
