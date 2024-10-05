# probability-of-a-word-experiments
Code to run experiments of paper "How to Compute the Probability of a Word"

## Install Dependencies

First, install R if you don't already have it in your computer. Then, create a conda environment with
```bash
$ conda env create -f scripts/environment.yml
```
Activate the environment and install your appropriate version of [PyTorch](https://pytorch.org/get-started/locally/).
```bash
$ conda install pytorch::pytorch torchvision torchaudio -c pytorch
$ # conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
$ pip install transformers
$ pip install wordsprobability
```

Also, install the necessary nltk sub-package. In a Python sheel, run:
```python
$ import nltk
$ nltk.download('punkt')
```

Finally, install the required R libraries with:
```bash
$ Rscript scripts/r_installer.R
```

## Running project's analysis

First, get the Dundee data `dundee.zip` and add it to folder `corpora`.
Then, get the data for the other datasets in this project by running:
```bash
$ make get_data
```

After that, you can easily run the entire pipeline for a dataset by running:
```bash
$ make DATASET=<dataset>
```
where dataset can be one of: `brown`, `natural_stories`, `dundee`, `provo`, `dundee_skip2zero`, `provo_skip2zero` (the last two of which include skipped words).

## Extra Information

#### Citation

If this code or the paper were usefull to you, consider citing it:

```bash
@article{pimentel-etal-2024-howto,
    title = "How to Compute the Probability of a Word",
    author = "Pimentel, Tiago and
    Meister, Clara",
    year = "2024",
}
```


#### Contact

To ask questions or report problems, please open an [issue](https://github.com/tpimentelms/probability-of-a-word-experiments/issues).
