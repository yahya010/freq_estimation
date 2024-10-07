import os
import sys
import argparse
import pandas as pd

import matplotlib as mpl
from matplotlib.ticker import FormatStrFormatter
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utils import constants
from utils import plot as utils_plot

MODEL_SIZES = {
        'gpt2-small': 117,
        'gpt2-medium': 345,
        'gpt2-xl': 1500,
        'gpt2-large': 762,
        'pythia-160m': 160,
        'pythia-410m': 410,
        'pythia-70m': 70,
        'pythia-14b': 1400,
        'pythia-28b': 2800,
        'pythia-69b': 6900,
        'pythia-120b': 12000
    }

PRETTY_NAMES = {'pythia': 'Pythia', 'gpt2': 'GPT-2'}
PRED_NAMES = {'surprisal_buggy': 'Surprisal (buggy)', 
              'surprisal': 'Surprisal (corrected)'}

def get_args():
    parser = argparse.ArgumentParser()
    # Results
    parser.add_argument('--input-path', type=str, required=True)
    parser.add_argument('--output-path', type=str, required=True)
    # Model
    parser.add_argument('--datasets', type=str, nargs='+', default=constants.DATASETS)
    parser.add_argument('--models', type=str, nargs='+', default=list(MODEL_SIZES.keys()))
    # parser.add_argument('--glm-type', type=str, default='merged-linear')

    return parser.parse_args()


def get_entropy_llh(model, dataset, input_path):
    dfs = []
    # fname_base = os.path.join('checkpoints', 'delta_llh/%s-%s-%s.tsv')
    fname = f'{input_path}/llh-{dataset}-{model}.tsv'
    # fname = fname_base % (dataset, model)

    try:
        df = pd.read_csv(fname, sep='\t')
    except FileNotFoundError:
        print(fname, ' not found')
        return pd.DataFrame()

    #df = df.groupby(['name', 'predictor','predictor_type']).aggregate(['mean', 'std']).reset_index()
    #df.columns = ["_".join(x) if x[1] else x[0] for x in df.columns.tolist() ]
    df['model'] = model
    df['model_family'] = PRETTY_NAMES.get(model.split('-')[0], model.split('-')[0])
    df['dataset'] = dataset
    df['dataset'] = df.dataset.apply(lambda x: constants.DATASET_NAMES[x])
    df['name'] = df.name.apply(lambda x: PRED_NAMES.get(x,x))

    drop_keywords = ['prev', 'budget', 'delta', 'next']
    df = df[df.name.apply(lambda x: all([keyword not in x for keyword in drop_keywords]))]
    df = df[(df.predictor_type == 1)]
    df['size'] = df.model.apply(lambda x: MODEL_SIZES[x])

    return df


def plot_datasets(datasets, args):
    all_dfs = []
    for dataset in datasets:
        for model in args.models:
            df = get_entropy_llh(model, dataset, args.input_path)
            all_dfs.append(df)
    all_dfs = pd.concat(all_dfs)
    #import ipdb; ipdb.set_trace()
    
    all_dfs.sort_values(['size', 'dataset', 'name','model_family'], inplace=True)
    all_dfs['diff'] = all_dfs['diff_empty'] * 100
    
    sns.set_theme(font="DejaVu Serif", style="whitegrid")
    plt.tight_layout()
    plt.xscale('log')

    g = sns.FacetGrid(all_dfs, col="dataset", height=3.5, aspect=.65, sharey=False, gridspec_kws={"wspace":0.3}) #
    g.set_titles(col_template="{col_name}",fontsize=11)

    g.map_dataframe(sns.lineplot, x='size', y='diff', hue='name', style='model_family', markers=['o','^'], errorbar=None)#, errorbar=('ci', 95), n_boot=20000)
    
    g.add_legend()
    g.set_axis_labels("", r'$\Delta_{\mathrm{llh}}$ ($10^{-2}$ nats)', fontsize=13)
    g.legend.get_texts()[0].set_text('')
    g.legend.get_texts()[3].set_text('')
    for i,ax in enumerate(g.axes.flat):
        if i == 2:
            ax.set_xlabel("# of Parameters (in Millions)")
        ax.xaxis.set_tick_params(pad=0)
        ax.yaxis.set_tick_params(pad=0)
        ax.set(xscale="log")
        for label in ax.get_yticklabels():
            label.set_fontsize(10)
        for label in ax.get_xticklabels():
            label.set_fontsize(11)

    plt.subplots_adjust(bottom=0.2)
    g.savefig(f'{args.output_path}/rt_llh.pdf', dpi=300)
        

def main():
    args = get_args()
    utils_plot.config_plots(width=4, height=6)

    plot_datasets(args.datasets, args)


if __name__ == '__main__':
    main()