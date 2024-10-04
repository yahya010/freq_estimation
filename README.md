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