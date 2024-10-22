import os
import sys
import argparse
import pandas as pd
from pathlib import Path
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# from dataset import NaturalStoriesDataset
from h01_data.models import unigram
from utils import utils


def get_args():
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument('--surprisal-fname', type=str, required=True)
    parser.add_argument('--rt-fname', type=str, required=True)
    # Output
    parser.add_argument('--output-fname', type=str, required=True)

    return parser.parse_args()


def merge_rt_and_surprisal(args):
    merge_columns = ['text_id', 'word_id']

    df_surprisals = pd.read_csv(args.surprisal_fname, sep='\t', keep_default_na=False)
    df_surprisals['text_id'] = df_surprisals['text_id'] + 1 # Fix text indexing

    df_rt = pd.read_csv(args.rt_fname, sep='\t', index_col=0, keep_default_na=False)
    df_rt = df_rt[~df_rt.outlier]
    del df_rt['outlier']
    df_rt = df_rt.groupby(merge_columns + ['ref_token']).agg('mean').reset_index()

    df = df_rt.set_index(merge_columns).join(
        df_surprisals.set_index(merge_columns),
        how='outer').reset_index()
    
    assert not df.surprisal.isna().any()
    assert (df.word == df.ref_token).all()
    return df


def get_frequencies(df):
    df['freq'] = df['word'].apply(
        lambda x: unigram.frequency(x))
    CURRENT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
    ROOT_DIR = CURRENT_DIR.parent.parent
    # add if statement for each dataset
    new_freq_path = Path(ROOT_DIR) / "corpora" / "rt" / "natstor_wiki_freqs.tsv"
    df2 = pd.read_csv(new_freq_path, sep='\t')
    df2 = df2.drop(columns=['Unnamed: 0'], errors='ignore')
    df = pd.merge(df, df2, on='word', how='left')
    return df


def get_spillover_vars(df):
    for variable in ['word', 'surprisal', 'surprisal_buggy', 'freq', 'freq_full_wiki', 'freq_3_4_wiki', 'freq_half_wiki', 'freq_1_5_wiki', 'freq_1_10_wiki','word_len']:
        df['prev_' + variable] = df.groupby("text_id", sort=False)[variable].shift(periods=1, fill_value=None)
        df['prev2_' + variable] = df.groupby("text_id", sort=False)[variable].shift(periods=2, fill_value=None)
        df['prev3_' + variable] = df.groupby("text_id", sort=False)[variable].shift(periods=3, fill_value=None)

def get_rt_with_surprisal_dataset(args):
    df = merge_rt_and_surprisal(args)
    df = get_frequencies(df)
    get_spillover_vars(df)

    return df


def main():
    args = get_args()
    df = get_rt_with_surprisal_dataset(args)
    utils.write_tsv(df, args.output_fname)


if __name__ == '__main__':
    main()
