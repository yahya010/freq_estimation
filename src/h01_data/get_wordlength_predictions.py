import os
import sys
import argparse
import math
import string
import pandas as pd

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utils import utils


def get_args():
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument('--input-fname', type=str, required=True)
    # Output
    parser.add_argument('--output-fname', type=str, required=True)

    return parser.parse_args()


def get_surprisals(input_fname):
    df = pd.read_csv(input_fname, sep='\t', keep_default_na=False, quoting=3)

    return df


def get_length_predictions(df):
    df['surprisal_squared'] = df['surprisal'].apply(lambda x: x**2)
    df['surprisal_buggy_squared'] = df['surprisal_buggy'].apply(lambda x: x**2)

    df['frequency'] = df.groupby('word')['surprisal'].transform('count')

    df_per_word = df.groupby('word')[['surprisal', 'surprisal_buggy', 'surprisal_squared', 
                                        'surprisal_buggy_squared', 'frequency']].agg('mean').reset_index()
    df_per_word['cch'] = df_per_word['surprisal_squared'] / df_per_word['surprisal']
    df_per_word['cch_buggy'] = df_per_word['surprisal_buggy_squared'] / df_per_word['surprisal_buggy']
    df_per_word['zipf'] = df_per_word['frequency'].apply(lambda x: math.log(x))
    df_per_word.rename(columns={'surprisal': 'piantadosi', 'surprisal_buggy': 'piantadosi_buggy'}, inplace=True)
    df_per_word['length'] = df_per_word['word'].apply(len)

    del df_per_word['surprisal_squared']
    del df_per_word['frequency']
    del df_per_word['surprisal_buggy_squared']

    return df_per_word


def contains_punctuation(text):
    # Check if the text contains only punctuation characters
    return any(char in string.punctuation or char.isspace() for char in text.strip())


def only_contains_latin_characters(text):
    # Check if the text contains only latin letters
    return all(char in string.ascii_letters for char in text)

def only_contains_lowercase_latin_characters(text):
    # Check if the text contains only lowercase latin letters
    return all(char in string.ascii_lowercase for char in text)


def drop_non_lowercase(df):
    # Improve quality of analysed data
    df['only_lowercase_letters'] = df.word.apply(only_contains_lowercase_latin_characters)
    df = df[df.only_lowercase_letters]

    return df

def process_dataset(args):
    df = get_surprisals(args.input_fname)
    df = get_length_predictions(df)
    df = drop_non_lowercase(df)
    return df


def main():
    args = get_args()
    df = process_dataset(args)
    utils.write_tsv(df, args.output_fname)


if __name__ == '__main__':
    main()