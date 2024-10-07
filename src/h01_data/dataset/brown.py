import numpy as np
import pandas as pd

from .base import BaseDataset
from utils import utils


class BrownDataset(BaseDataset):
    unused_columns = ['word_in_exp', 'WorkerId', 'code', 'word']

    @classmethod
    def get_text(cls, input_path):
        stories = cls.get_stories_text(input_path)
        stories_text = [x[1].strip() for x in stories]

        return '\n'.join(stories_text)
    
    @classmethod
    def preprocess(cls, input_path):
        # Get text
        stories = cls.get_stories_text(input_path)
        ns_text_words = cls.get_corpus_words(stories)

        # Get RTs
        df = cls.read_data(input_path)

        # Find outliers
        df = utils.find_outliers(df, transform=np.log)

        # Create dataframe
        df = cls.create_analysis_dataframe(df, ns_text_words)

        # Check word matches ref_token (except for peeked vs peaked distinction)
        assert (df['word'] == df['ref_token']).all()

        # Deleted unused info from dataframe
        cls.remove_unused_columns(df, cls.unused_columns)
        return df

    @classmethod
    def get_stories_text(cls, input_path):
        df = cls.read_data(input_path)

        df = df[['text_id','word_id','word']].drop_duplicates()
        df.sort_values(['text_id', 'word_id'], inplace=True)
        df_stories = df.groupby('text_id')['word'].aggregate(lambda x: ' '.join(x))

        paragraphs = [(text_id, text_str) for text_id, text_str in df_stories.items()]

        return paragraphs
    
    @staticmethod
    def read_data(input_path):
        df = pd.read_csv(f'{input_path}/brown_spr.csv')
        df = df.drop(columns='Unnamed: 0')
        df.rename(columns = {'subject': 'WorkerId',
                             'text_pos': 'word_id'}, inplace = True)
        df['text_id'] = df['text_id'] + 1 # Make text_id start at 1 to match other datasets
   
        # Merge a few whitespace-separated words
        df['word'] = df['word'].apply(lambda x: x.replace(' --', ','))
        df['word'] = df['word'].apply(lambda x: x.replace('N. Y.', 'N.Y.'))
        df['word'] = df['word'].apply(lambda x: x.replace('N. H.', 'N.H.')) 

        return df
