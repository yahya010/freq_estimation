import os
import re
import sys
import argparse
# import bisect
import math
# from string import punctuation

import numpy as np
import pandas as pd
# import mosestokenizer

sys.path.insert(1, os.path.join(sys.path[0], '..'))
# from corpus import process, metrics
from utils import utils


def get_args():
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument('--input-fname', type=str, required=True)
    # Output
    parser.add_argument('--output-fname', type=str, required=True)

    return parser.parse_args()


# def process_surprisals(args):
#     model_shortname = {
#         'pythia-70m': '70m', 
#         'pythia-160m': '160m', 
#         'pythia-410m': '410m', 
#         'pythia-14b': '1.4b', 
#         'pythia-28b': '2.8b', 
#         'pythia-69b': '6.9b', 
#         'pythia-120b': '12.0b'
#     }
#     start_of_word_symbol = 'Ġ'

#     input_fname = f'{args.input_path}/finished_probs/surprisals_wiki_en_{model_shortname[args.model]}.tsv'
#     print(input_fname)
#     model_probs = pd.read_csv(input_fname, sep='\t')
#     del model_probs['Unnamed: 0']
#     model_probs.tokens = model_probs.tokens.apply(str)

#     # model_probs.tokens.to_string().apply(lambda x: x[0] == start_of_word_symbol)
#     model_probs['is_bow'] = model_probs.tokens.apply(lambda x: x[0] == start_of_word_symbol)
#     model_probs['ones'] = 1
#     model_probs['text_pos'] = model_probs.groupby('text_id')['ones'].cumsum()
#     model_probs['is_bos'] = (model_probs['text_pos'] == 1)
#     del model_probs['ones']
#     del model_probs['text_pos']
#     model_probs['word_id'] = model_probs.groupby('text_id')['is_bow'].cumsum()
#     model_probs['is_eow'] = model_probs.groupby('text_id')['is_bow'].shift(-1)
#     model_probs.loc[model_probs['is_eow'].isna(), 'is_eow'] = True

#     model_probs['surprisal_fixed'] = model_probs['surprisal'] \
#                                      - model_probs['bow_fix'] * model_probs['is_bow'] \
#                                      - model_probs['bos_fix'] * model_probs['is_bos'] \
#                                      + model_probs['eow_fix'] * model_probs['is_eow']

#     word_probs = model_probs.groupby(['text_id', 'word_id']).agg('sum')

#     assert ((word_probs.is_bow + word_probs.is_bos) == 1).all()
#     assert (word_probs.is_eow <= 1).all()

#     word_probs['tokens'] = word_probs.tokens.apply(lambda x: x[1:] if (x[0] == start_of_word_symbol) else x)

#     return word_probs


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


def process_dataset(args):
    df = get_surprisals(args.input_fname)
    df = get_length_predictions(df)
    return df


def main():
    args = get_args()
    df = process_dataset(args)
    utils.write_tsv(df, args.output_fname)


if __name__ == '__main__':
    main()