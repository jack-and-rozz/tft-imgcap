# coding: utf-8
import argparse

def add_common_args(parser):
    parser.add_argument('model_root', help='Directory to save the trained model, evaluation results, etc.')
    return parser

def add_data_args(parser):
    # Data
    parser.add_argument('--label-types', type=str, nargs='+',
                        default=['champion', 'star', 'item1', 'item2', 'item3'],
                        choices=['champion', 'star', 'item1', 'item2', 'item3'], 
                        help=' ')
    parser.add_argument('--data-dir', default='datasets/clipped', 
                        help='Directory to store csv files of labels and images clipped from annotated screen shots.')
    parser.add_argument('--img-height', type=int, default=100, help=' ')
    parser.add_argument('--img-width', type=int, default=80, help=' ')
    return parser
    
def add_model_args(parser):
    parser.add_argument('--cnn-dims', metavar='N', type=int,
                        default=[64, 64, 32], nargs='+', help=' ')
    return parser

def add_train_args(parser):
    # Training 
    parser.add_argument('--max-epoch', type=int, default=200, help=' ')
    parser.add_argument('--batch-size', type=int, default=80, help=' ')
    parser.add_argument('--L2reg-factor', type=int, default=0.0, help=' ')
 
    parser.add_argument('--init-lr', type=float, default=1e-3, help=' ')
    parser.add_argument('--final-lr', type=float, default=1e-9, help=' ')
    parser.add_argument('--lr-decay-rate', type=float, default=0.95, 
                        help='Learning rate decay per epoch.')
    parser.add_argument('--dropout-rate', type=float, default=0.25, help=' ')
    parser.add_argument('--enable-class-weight', action="store_true", 
                        default=False, help=' ')

    parser.add_argument('--train-csv', type=str, default='train.csv', help=' ')
    parser.add_argument('--dev-csv', type=str, default='dev.csv', help=' ')

    return parser

def add_test_args(parser):
    parser.add_argument('--output-dir', type=str, default=None, help=' ')
    parser.add_argument('--test-csv', type=str, default='test.csv', help=' ')
    return parser

def get_train_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser = add_common_args(parser)
    parser = add_data_args(parser)
    parser = add_model_args(parser)
    parser = add_train_args(parser)
    return parser

def get_test_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser = add_common_args(parser)
    parser = add_data_args(parser)
    parser = add_test_args(parser)
    return parser
