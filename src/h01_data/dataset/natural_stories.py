import bisect
import numpy as np
import pandas as pd

from .base import BaseDataset
from utils import utils


class NaturalStoriesDataset(BaseDataset):
    unused_columns = ['WorkerId', 'WorkTimeInSeconds', 'correct', 'zone', 
                      'meanItemRT', 'sdItemRT', 'gmeanItemRT', 'gsdItemRT',
                      'word']

    @staticmethod
    def get_stories_text(input_path):
        gpt3_probs = pd.read_csv("%s/all_stories_gpt3.csv" % (input_path))

        # To get same indexing as stories db
        gpt3_probs["story"] = gpt3_probs["story"] + 1

        # Use offset to compute string length, and add whitespaces back when needed
        gpt3_probs['len'] = gpt3_probs.groupby("story", sort=False)['offset'].shift(periods=-1, fill_value=0) - gpt3_probs['offset']
        gpt3_probs['token_with_ws'] = gpt3_probs.apply(lambda x: x['token'] if x['len'] == len(x['token']) else x['token'] + ' ', axis=1)

        # Concatenate words in a story, and stories together
        stories_df = gpt3_probs.groupby(by=["story"], sort=False).agg({"token_with_ws": utils.string_join}).reset_index()
        # stories = '\n'.join(stories_df['token_with_ws'].apply(lambda x: x.strip()).values)
        stories = list(zip(stories_df['story'], stories_df['token_with_ws']))

        return stories

    @classmethod
    def get_text(cls, input_path):
        stories = cls.get_stories_text(input_path)
        stories_text = [x[1].strip() for x in stories]

        return '\n'.join(stories_text)
    
    @classmethod
    def preprocess(cls, input_path):
        # Get natural stories text
        stories = cls.get_stories_text(input_path)
        ns_stats = cls.get_corpus_stats(stories)

        # Get natural stories RTs
        df = pd.read_csv("%s/processed_RTs.tsv" % (input_path), sep='\t').drop_duplicates()
        df.rename(columns={'RT': 'time', 'item': 'text_id'}, inplace=True)
        df['word_id'] = df['zone'] - 1

        # Exclude outliers
        df = utils.find_outliers(df, transform=np.log)

        # Create preprocessed dataframe
        df = cls.create_analysis_dataframe(df, ns_stats)

        # Check word matches ref_token (except for peeked vs peaked distinction)
        assert ((df['word'] == df['ref_token']) | (df['word'] == 'peaked')).all()

        # Deleted unused info from dataframe
        cls.remove_unused_columns(df)
        return df
    
    # @staticmethod
    # def remove_unused_columns(df):
    #     for col in unused_columns:
    #         del df[col]