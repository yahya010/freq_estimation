from collections import defaultdict
from string import punctuation
import pandas as pd

from utils import utils


class BaseDataset():
    @staticmethod
    def get_corpus_stats(stories):
        stats = defaultdict(dict)
        for i, s in stories:
            # remove leading and trailing white space
            s = s.strip()
            stats['split_string'][i] = s.split()
            # sents = tokenize_to_sents(s) if split_sens else [s]
            # lens = [len(sen.split()) for sen in sents]
            # assert len(s.split()) == sum(lens)
            # stats['len'][i] = np.array(lens)
            # stats['sent_markers'][i] = np.cumsum(lens)
            # stats['ch_len'][i] = np.array([sum([len(ch) for ch in sen.split()]) for sen in sents])
            # stats['uni_log_probs'][i] = np.array(metrics.string_to_uni_log_probs(s))

        # infotheoretic_stats = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

        # # # lang_mean = get_corpus_mean(model)
        # # # print("Using language mean surprisal:", lang_mean)

        # get_per_token_info(stories, model, infotheoretic_stats, stats)
        # # get_per_token_info(stories, model, score, log_prob_stats, stats, 'agg_log_prob')
        # # get_per_token_info(stories, model, get_entropies, log_prob_stats, stats, 'agg_entropy')
        # # get_per_token_info(stories, model, make_renyi_entropies_func(float('inf')), log_prob_stats, stats, 'agg_entropy_argmin')

        # # for alpha in constants.RENYI_RANGE:
        # #     get_per_token_info(stories, model, make_renyi_entropies_func(alpha), log_prob_stats, stats, 'agg_renyi_%.2f' % alpha)

        # stats.update(infotheoretic_stats)
        return stats

    @classmethod
    def create_analysis_dataframe(cls, df, stats, lang="en", dataset=None):
        df = df.copy()
        # get standard corpus statistics
        df = cls.add_standard_columns(df, stats['split_string'], lang=lang, dataset=dataset)
        # # get log_prob related corpus statistics
        # df = add_log_prob_columns(df, stats, model=model)

        return df

    @staticmethod
    def add_standard_columns(df, split_strings, lang="en", dataset=None):
        # ref_token is used for sanity checking. should be same as word
        df['ref_token'] = df[['text_id', 'word_id']].apply(
            lambda x: split_strings[x['text_id']][x['word_id']], axis=1)

        if dataset == 'provo':
            df['word'] = df['word'].apply(lambda x: x.lower().strip(punctuation))
            df['ref_token'] = df['ref_token'].apply(lambda x: x.lower().strip(punctuation))
            assert (df['word'] == df['ref_token']).all()

        elif dataset == 'dundee':
            df['word_orig'] = df['word']
            df['word'] = df['word'].apply(lambda x: x.strip(punctuation + ' \t').replace('"', '‚').replace(',‚', '‚,'))
            df['ref_token2'] = df['ref_token'].apply(lambda x: x.strip(punctuation + ' \t').strip('‚"').replace('”', '‚'))

            assert ((df['word'] == df['ref_token2']) | (df['word_orig'] == ' (HFEA) ') | (df['word_orig'].apply(len) == 20)).all()
        else:
            pass
            # assert ((df['word'] == df['ref_token']) | (df['word'] == 'peaked')).all()

        # # Get previous word
        # df['prev_word'] = df[['text_id', 'new_ind']].apply(
        #     lambda x: split_strings[x['text_id']][x['new_ind'] - 1] if x['new_ind'] - 1 >= 0 else '', axis=1)
        # df['prev2_word'] = df[['text_id', 'new_ind']].apply(
        #     lambda x: split_strings[x['text_id']][x['new_ind'] - 2] if x['new_ind'] - 2 >= 0 else '', axis=1)
        # df['prev3_word'] = df[['text_id', 'new_ind']].apply(
        #     lambda x: split_strings[x['text_id']][x['new_ind'] - 3] if x['new_ind'] - 3 >= 0 else '', axis=1)

        # Center times per worker
        df['centered_time'] = df['time'] - df.groupby(by=["WorkerId"])["time"].transform('mean')

        # Get word length
        df['word_len'] = df['word'].apply(len)
        # df['prev_word_len'] = df['prev_word'].apply(lambda x: len(x))
        # df['prev2_word_len'] = df['prev2_word'].apply(lambda x: len(x))
        # df['prev3_word_len'] = df['prev3_word'].apply(lambda x: len(x))

        # # Get word frequency
        # df['freq'] = df['word'].apply(
        #     lambda x: unigram.frequency(x.strip().strip(punctuation).lower(), lang))
        # df['prev_freq'] = df['prev_word'].apply(
        #     lambda x: unigram.frequency(x.strip().strip(punctuation).lower(), lang))
        # df['prev2_freq'] = df['prev2_word'].apply(
        #     lambda x: unigram.frequency(x.strip().strip(punctuation).lower(), lang))
        # df['prev3_freq'] = df['prev3_word'].apply(
        #     lambda x: unigram.frequency(x.strip().strip(punctuation).lower(), lang))

        return df
    
    @classmethod
    def remove_unused_columns(cls, df):
        for col in cls.unused_columns:
            del df[col]
