# import bisect
from string import punctuation
import numpy as np
import pandas as pd
import mosestokenizer

from .base import BaseDataset
from utils import utils


class ProvoDataset(BaseDataset):
    # unused_columns = ['RECORDING_SESSION_LABEL', 'Word_Unique_ID', 'sentence_num', 'Word_In_Sentence_Number']
    # 'Word_Number',
    unused_columns_initial = ['RECORDING_SESSION_LABEL', 'Word_Unique_ID',
                            'sentence_num', 'Word_In_Sentence_Number',
                            'Word_Cleaned', 'Word_Length', 'Total_Response_Count', 'Unique_Count',
                            'OrthographicMatch', 'OrthoMatchModel', 'IsModalResponse',
                            'ModalResponse', 'ModalResponseCount', 'Certainty', 'POS_CLAWS',
                            'Word_Content_Or_Function', 'Word_POS', 'POSMatch', 'POSMatchModel',
                            'InflectionMatch', 'InflectionMatchModel', 'LSA_Context_Score',
                            'LSA_Response_Match_Score', 'IA_ID', 'IA_LABEL', 'TRIAL_INDEX',
                            'IA_LEFT', 'IA_RIGHT', 'IA_TOP', 'IA_BOTTOM', 'IA_AREA',
                            'IA_FIRST_FIXATION_INDEX', 'IA_FIRST_FIXATION_VISITED_IA_COUNT',
                            'IA_FIRST_FIXATION_X', 'IA_FIRST_FIXATION_Y',
                            'IA_FIRST_FIX_PROGRESSIVE', 'IA_FIRST_FIXATION_RUN_INDEX',
                            'IA_FIRST_FIXATION_TIME', 'IA_FIRST_RUN_FIXATION_COUNT',
                            'IA_FIRST_RUN_END_TIME', 'IA_FIRST_RUN_FIXATION_.',
                            'IA_FIRST_RUN_START_TIME', 'IA_FIXATION_COUNT', 'IA_RUN_COUNT',
                            'IA_REGRESSION_IN', 'IA_REGRESSION_IN_COUNT',
                            'IA_REGRESSION_OUT', 'IA_REGRESSION_OUT_COUNT',
                            'IA_REGRESSION_OUT_FULL', 'IA_REGRESSION_OUT_FULL_COUNT',
                            'IA_REGRESSION_PATH_DURATION', 'IA_FIRST_SACCADE_AMPLITUDE',
                            'IA_FIRST_SACCADE_ANGLE', 'IA_FIRST_SACCADE_END_TIME',
                            'IA_FIRST_SACCADE_START_TIME']
    unused_columns_final = ['word', 'WorkerId','Word_Number', 'time_v3', 'time_v2',
                            'time_v1', 'time_v4', 'time_v5','centered_time']
    # Word_Number
    # we define 5 types of aggregated reading time from v1 to v5
    # time_v1 -> IA_DWELL_TIME
    # time_v2 -> IA_FIRST_RUN_DWELL_TIME
    # time_v3 -> IA_FIRST_FIXATION_DURATION
    # time_v4 -> First Pass First Fixation Time
    # time_v5 -> First Pass Dwell Time
    main_time_field='time_v5'
    skip2zero = True
    
    @classmethod
    def get_stories_text(cls, input_path):
        # Get provo data
        provo_text = pd.read_csv('%s/provo_norms.csv' % (input_path), encoding='latin-1')

        # Drop duplicate entries. Text_ID = 27 is duplicated, but one entry has 
        # doesnÕt and other doesn't, so this is handled below as well!
        provo_text = provo_text[['Text_ID', 'Text']].drop_duplicates().sort_values(by=['Text_ID'])
        provo_text.drop(provo_text[(provo_text.Text_ID == 27) & (~provo_text.Text.str.contains('doesn\'t', regex=False))].index, inplace=True)

        # Get a dict of text_id and text
        paragraphs = [(text_id, text_str) for text_id, text_str in 
                      provo_text[['Text_ID', 'Text']].itertuples(index=False, name=None)]
        return paragraphs
    @staticmethod
    def ref_sanity_check(df):
        df['word'] = df['word'].apply(lambda x: x.lower().strip(punctuation))
        df['ref_token'] = df['ref_token'].apply(lambda x: x.lower().strip(punctuation))
        assert (df['word'] == df['ref_token']).all()
    @classmethod
    def get_text(cls, input_path):
        stories = cls.get_stories_text(input_path)
        stories_text = [x[1].strip() for x in stories]

        return '\n'.join(stories_text)
    
    # @classmethod
    # def get_stories_text2(cls, input_path):
    #     # Get provo data
    #     df = cls.read_data(input_path)

    #     moses_normaliser = mosestokenizer.MosesPunctuationNormalizer('en')
    #     df['word'] = df.apply(lambda x: moses_normaliser(x['word']).strip(), axis=1)

    #     # fixing small discrepancy
    #     df.loc[df['word'] == '0.9', 'word'] = '90%'
    #     df.loc[df['word'] == 'women?s', 'word'] = 'womenõs'


    #     # provo_text = pd.read_csv('%s/provo_norms.csv' % (input_path), encoding='latin-1')
    #     # provo_text = provo_text[['Text_ID','Text']].drop_duplicates().sort_values(by=['Text_ID'])
    #     # provo_text.drop(provo_text[(provo_text.Text_ID == 27) & (~provo_text.Text.str.contains('doesn\'t', regex=False))].index, inplace=True)

    #     # # inds = provo_text.apply(lambda x: list(range(1, len(x['Text'].split()) + 1)), axis=1)
    #     # # inds = {i: j for i, j in zip(provo_text['Text_ID'], inds)}
        
    #     # paragraphs = {i: j.replace(u'\uFFFD', '?') for i, j in provo_text[['Text_ID', 'Text']].itertuples(index=False, name=None)}
    #     paragraphs = cls.get_stories_text(input_path)

    #     paragraphs_split = {i: [k.strip(punctuation) for k in j.lower().split()] for i, j in paragraphs.items()}

    #     provo_stats = cls.get_corpus_stats(paragraphs.items())

    #     df['word_id'] = df['Word_Number'] - 2
    #     df['word_id'] = df.apply(lambda x: x['word_id'] + paragraphs_split[x['text_id']][x['word_id']:].index(x['word'].lower().strip(punctuation)), axis=1)
    #     # df['sentence_num'] = df.apply(lambda x: bisect.bisect(provo_stats['sent_markers'][x['text_id']], x['new_ind']), axis=1)

    #     df['time'] = df[cls.main_time_field]

    #     if not cls.skip2zero:
    #         df = utils.find_outliers(df.loc[df['time'] != 0].copy(), transform=np.log)
    #     else:
    #         df = utils.find_outliers(df, transform=np.log, ignore_zeros=True)
    #         df['skipped'] = (df['IA_SKIP'] == 1)

    #     df = cls.create_analysis_dataframe(df, provo_stats, dataset='provo')
    #     cls.ref_sanity_check(df)
    #     import ipdb; ipdb.set_trace()
    #     return df

    #     # stories = cls.get_stories_text(input_path)
    #     # stories_text = [x[1].strip() for x in stories]

    #     # return '\n'.join(stories_text)
    
    @classmethod
    def preprocess(cls, input_path):
        # Get provo rt data
        df = cls.read_data(input_path)
        cls.remove_unused_columns(df, cls.unused_columns_initial)

        # moses_normaliser = mosestokenizer.MosesPunctuationNormalizer('en')
        # df['word'] = df.apply(lambda x: moses_normaliser(x['word']).strip(), axis=1)
        # df['word2'] = df['word']

        # fixing small discrepancy
        df.loc[df['word'] == '0.9', 'word'] = '90%'
        df.loc[df['word'] == 'women?s', 'word'] = 'womenõs'
        # ????????????
        

        # Get provo text
        stories = cls.get_stories_text(input_path)

        provo_text_words = cls.get_corpus_words(stories)

        paragraphs_split = {i: [k.strip(punctuation) for k in j.lower().split()] for i, j in stories}


        # Set word_id to be equal to Word_Number - 2. Word_Number, however, skips some values (word 42 is followed by 44 in story 3)
        # So fix this by looking at which position in string paragraphs_split[x['text_id']] matches the word.
        df['word_id'] = df['Word_Number'] - 2
        df['word_id'] = df.apply(
            lambda x: x['word_id'] + 
            paragraphs_split[x['text_id']][x['word_id']:].index(
                x['word'].lower().strip(punctuation)), axis=1)
        
        # df['sentence_num'] = df.apply(lambda x: bisect.bisect(provo_stats['sent_markers'][x['text_id']], x['new_ind']), axis=1)
        
        df['time'] = df[cls.main_time_field]

        if not cls.skip2zero:
            df = utils.find_outliers(df.loc[df['time'] != 0].copy(), transform=np.log)
        else:
            df = utils.find_outliers(df, transform=np.log, ignore_zeros=True)
            df['skipped'] = (df['skipped'] == 1)

        df = cls.create_analysis_dataframe(df, provo_text_words, dataset='provo')
        cls.ref_sanity_check(df)
        import ipdb; ipdb.set_trace()

        # Check word matches ref_token (except for peeked vs peaked distinction)
        # assert ((df['word'] == df['ref_token']) | (df['word'] == 'peaked')).all()

        # Deleted unused info from dataframe
        cls.remove_unused_columns(df, cls.unused_columns_final)
        return df
    
    @staticmethod
    def read_data(input_path):
        provo = pd.read_csv('%s/provo.csv' % (input_path))
        provo.rename(columns = {'IA_DWELL_TIME':'time_v1', 'Participant_ID': 'WorkerId', 'Word': 'word',
                                'Text_ID':'text_id', 'Sentence_Number': 'sentence_num', 'IA_SKIP': 'skipped',
                                'IA_FIRST_RUN_DWELL_TIME': 'time_v2', 'IA_FIRST_FIXATION_DURATION': 'time_v3'}, inplace = True)
        provo = provo.dropna(subset=['Word_Number'])
        provo = provo.astype({'Word_Number': 'Int64', 'sentence_num': 'Int64'})

        # First Pass First Fixation Time
        provo['time_v4'] = provo['time_v3']
        provo.loc[provo['skipped'] == 1, 'time_v4'] = 0

        # First Pass Dwell Time
        provo['time_v5'] = provo['time_v2']
        provo.loc[provo['skipped'] == 1, 'time_v5'] = 0

        # Zero time if not fixated
        provo.loc[provo['time_v2'].isna(), 'time_v2'] = 0
        provo.loc[provo['time_v3'].isna(), 'time_v3'] = 0

        return provo
