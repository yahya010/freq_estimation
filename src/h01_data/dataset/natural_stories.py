import pandas as pd

from utils import utils


class NaturalStoriesDataset():

    @staticmethod
    def get_text(input_path):
        gpt3_probs = pd.read_csv("%s/all_stories_gpt3.csv" % (input_path))

        # To get same indexing as stories db
        gpt3_probs["story"] = gpt3_probs["story"] + 1

        # Use offset to compute string length, and add whitespaces back when needed
        gpt3_probs['len'] = gpt3_probs.groupby("story", sort=False)['offset'].shift(periods=-1, fill_value=0) - gpt3_probs['offset']
        gpt3_probs['token_with_ws'] = gpt3_probs.apply(lambda x: x['token'] if x['len'] == len(x['token']) else x['token'] + ' ', axis=1)

        # Concatenate words in a story, and stories together
        stories_df = gpt3_probs.groupby(by=["story"], sort=False).agg({"token_with_ws": utils.string_join}).reset_index()
        stories = '\n'.join(stories_df['token_with_ws'].apply(lambda x: x.strip()).values)

        return stories