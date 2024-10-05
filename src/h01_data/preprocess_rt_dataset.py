import os
import sys
import argparse

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from dataset import NaturalStoriesDataset
from utils import utils


def get_args():
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument('--input-path', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    # Output
    parser.add_argument('--output-fname', type=str, required=True)

    return parser.parse_args()


def get_text(args):
    if args.dataset == 'natural_stories':
        df = NaturalStoriesDataset.preprocess(args.input_path)
    # elif args.dataset == 'brown':
    #     text = get_brown_text(args)
    # elif args.dataset == 'provo':
    #     text = get_provo_text(args, main_time_field='time5')
    # elif args.dataset == 'provo_skip2zero':
    #     text = get_provo_text(args, main_time_field='time5', skip2zero=True)
    # elif args.dataset == 'dundee':
    #     text = get_dundee_text(args, main_time_field='ProgressiveFirstPassTime')
    # elif args.dataset == 'dundee_skip2zero':
    #     text = get_dundee_text(args, main_time_field='ProgressiveFirstPassTime', skip2zero=True)
    else:
        raise ValueError('Invalid dataset name: %s' % args.dataset)

    return df


def main():
    args = get_args()
    df = get_text(args)
    utils.write_tsv(df, args.output_fname)


if __name__ == '__main__':
    main()