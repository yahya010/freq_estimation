import os
import bisect
from string import punctuation
import numpy as np
import pandas as pd
import mosestokenizer
import re
from .base import BaseDataset
from utils import utils


class DundeeDataset(BaseDataset):
    unused_columns_initial = ['index', 'TEXT', 'LINE', 'OLEN', 'WLEN', 'XPOS', 'OBLP', 'WDLP', 'TXFR']
    unused_columns_final = ['index', 'word', 'Word_Number', 'TotalReadingTime', 'FirstFixationTime', 'FirstPassTime',
                            'ProgressiveFirstFixationTime', 'ProgressiveFirstPassTime', 'word_orig', 'ref_token2',
                            'first_pass', 'is_progressive', 'is_regression', 'WorkerId']
    main_time_field = 'ProgressiveFirstPassTime'
    skip2zero = True

    @staticmethod
    def ordered_string_join(x, j=''):
        s = sorted(x, key=lambda x: x[0])
        a, b = list(zip(*s))
        return a, j.join(b)

    @classmethod
    def get_stories_text(cls, input_path):
        dundee_text_dir = '%s/texts' % (input_path)
        textList = [os.path.join(dundee_text_dir, f) for f in os.listdir(dundee_text_dir) if re.match(r'tx\d+wrdp\.dat', f)]
        cols = ['word', 'text_id', 'screen_nr', 'line_nr', 'pos_on_line', 'serial_nr', 'initial_letter_position', 'word_len_punct', 'word_len', 'punc_code', 'n_chars_before','n_chars_after', 'Word_Number', 'local_word_freq']
        dundeeTexts = pd.DataFrame(columns = cols)
        dfs_temp = []
        for text in textList:
            temp = pd.read_csv(text, sep='\s+', names=cols, quoting=3, encoding='Windows-1252', keep_default_na=False)
            dfs_temp += [temp]

        dundeeTexts = pd.concat(dfs_temp)
        dundeeTexts['word'] = dundeeTexts.word.astype('string')

        _, paragraphs = zip(*dundeeTexts[['text_id', 'Word_Number', 'word']].drop_duplicates().dropna().groupby(by = ['text_id']).apply(lambda x: cls.ordered_string_join(zip(x['Word_Number'], x['word']), ' ')))
        paragraphs = list(enumerate(paragraphs, 1))
        return paragraphs

    @classmethod
    def get_text(cls, input_path):
        stories = cls.get_stories_text(input_path)
        stories_text = [x[1].strip() for x in stories]

        return '\n'.join(stories_text)
    
    @classmethod
    def preprocess(cls, input_path):
        # Get text
        stories = cls.get_stories_text(input_path)
        dundee_text_words = cls.get_corpus_words(stories)

        # Get rt data
        df = cls.read_data(input_path)
        cls.remove_unused_columns(df, cls.unused_columns_initial)

        # Annotate regressions, first pass, and progressive fixations
        df = cls.mark_regressions(df)
        df = cls.mark_first_pass(df)
        df['is_progressive'] = df.first_pass & (~df.is_regression)

        # Compute agg reading time measurements for each word
        cls.get_agg_reading_time_measurements(df)

        # See Smith & Levy 2013
        df.drop(df.loc[df.WorkerId=='sg'].index, inplace=True)

        # Normalise text with moses
        moses_normaliser = mosestokenizer.MosesPunctuationNormalizer("en")
        df['word'] = df.apply(
            lambda x: moses_normaliser(x['word'].strip().replace('""','"').replace('\n',' ')), axis=1)

        # Set the time which will be analysed to the user's choice        
        df['time'] = df[cls.main_time_field]
        df['word_id'] = df['Word_Number'] - 1
        df = df.reset_index()

        # Either drop or not skipped words. Find outliers
        if not cls.skip2zero:
            df = utils.find_outliers(df.loc[df['time'] != 0].copy(), transform=np.log)
        else:
            df = utils.find_outliers(df, transform=np.log, ignore_zeros=True)
            df['skipped'] = (df['ProgressiveFirstFixationTime'] == 0)

        # Create dataframe and sanity check it
        df = cls.create_analysis_dataframe(df, dundee_text_words)
        cls.tokens_sanity_check(df)

        # Deleted unused info from dataframe
        cls.remove_unused_columns(df, cls.unused_columns_final)
        return df
    
    @staticmethod
    def tokens_sanity_check(df):
        df['word_orig'] = df['word']
        df['word'] = df['word'].apply(
            lambda x: x.strip(punctuation + ' \t').replace('"', '‚').replace(',‚', '‚,'))
        df['ref_token2'] = df['ref_token'].apply(
            lambda x: x.strip(punctuation + ' \t').strip('‚"').replace('”', '‚'))

        assert ((df['word'] == df['ref_token2']) | (df['word_orig'] == ' (HFEA) ') | (df['word_orig'].apply(len) == 20)).all()

    @classmethod
    def read_data(cls, input_path):
        dundee_eyetracking_dir = '%s/eye-tracking' % (input_path)
        fileList = [
            os.path.join(dundee_eyetracking_dir, f)
            for f in os.listdir(dundee_eyetracking_dir) if re.match(r's\w\d+ma2p*\.dat', f)]

        # Read all dundee files into a single dataframe
        dfs_temp = []
        for file in fileList:
            df_temp = pd.read_csv(file, sep='\s+', quoting=3, encoding='Windows-1252', keep_default_na=False)

            # Get experiment metadata from file name
            match = re.search(r'(s\w)(\d+)ma2p*\.dat', file.split('/')[-1])
            worked_id = match.group(1)
            text_id = int(match.group(2))
            df_temp.insert(loc=0, column='text_id', value=text_id)
            df_temp.insert(loc=0, column='WorkerId', value=worked_id)

            dfs_temp += [df_temp]
        dundee = pd.concat(dfs_temp)

        # Rename columns and change data format
        dundee.rename(columns={'FDUR': 'time', 'WORD': 'word', 'WNUM': 'Word_Number',
                               'FXNO': 'fixation_number'}, inplace = True)
        dundee['time'] = dundee.time.astype('int64')
        dundee['fixation_number'] = dundee.fixation_number.astype('int64')
        dundee['Word_Number'] = dundee.Word_Number.astype('int64')
        dundee.reset_index(inplace=True)

        return dundee

    def mark_regressions(df):
        df = df.iloc[::-1].copy()

        # Mark skipped words as having infinite fixation number
        df['fixation_number_temp'] = df['fixation_number']
        df.loc[df['fixation_number_temp'] == 0, 'fixation_number_temp'] = float('inf')

        # Get index of earliest fixation to a future word. If current fixation is larger 
        # than that, this word was visited after a future one, so this is a regression
        df['future_min_fixation'] = df.groupby(['WorkerId', 'text_id'])['fixation_number_temp'].agg('cummin')
        df['is_regression'] = df['fixation_number'] > df['future_min_fixation']

        # Delete new columns
        del df['fixation_number_temp']
        del df['future_min_fixation']
        return df.iloc[::-1]

    def mark_first_pass(df):
        # Sort fixations by each stories' fixation number 
        df_temp = df[df.fixation_number > 0].sort_values(['WorkerId', 'text_id', 'fixation_number'])

        # Mark word of previous fixation
        df_temp['word_number_prev'] = df_temp.groupby(['WorkerId', 'text_id']).Word_Number.shift(1)
        df_temp.loc[df_temp['word_number_prev'].isna(), 'word_number_prev'] = 0

        # If current word is the same as the previous fixation's word, then this is a continued gaze 
        # at same word, else it's a new word. Count number of gazes until current fixation, this forms gaze_ids
        df_temp['continued_pass'] = (df_temp['word_number_prev'] == df_temp['Word_Number'])
        df_temp['start_pass'] = ~(df_temp['continued_pass'])
        df_temp['pass_number'] = df_temp.groupby(['WorkerId', 'text_id']).start_pass.cumsum()

        # Save number of gazes until current fixation
        df['pass_number'] = df_temp['pass_number']
        df.loc[df['pass_number'].isna(), 'pass_number'] = 0

        # Get the smallest gaze_id is the same word. If the same a current gaze_id, this is a first pass on a word
        df['pass_number_min_temp'] = df.groupby(['WorkerId', 'text_id', 'Word_Number']).pass_number.transform('min')
        df['first_pass'] = (df.pass_number == df.pass_number_min_temp) | (df.pass_number.isna())

        # Delete unused columns
        del df['pass_number_min_temp']
        return df
    
    @staticmethod
    def get_agg_reading_time_measurements(df):
        df['TotalReadingTime'] = df.groupby(by=["WorkerId","text_id", "Word_Number"]).time.transform('sum')
        df['FirstFixationTime'] = df.time
        df['FirstPassTime'] = df.groupby(by=["WorkerId","text_id", "Word_Number", 'pass_number']).time.transform('sum')
        df['ProgressiveFirstFixationTime'] = df.FirstFixationTime * df.is_progressive
        df['ProgressiveFirstPassTime'] = df.FirstPassTime * df.is_progressive
        df.drop_duplicates(subset=["WorkerId","text_id", 'Word_Number'], inplace=True)
