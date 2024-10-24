#!/bin/bash

for model in 'gpt2-small' 'gpt2-medium' 'gpt2-large' 'gpt2-xl' \
             'pythia-70m' 'pythia-160m' 'pythia-410m' 'pythia-14b' 'pythia-28b' 'pythia-69b' 'pythia-120b'
do
    for dataset in 'natural_stories' 'provo' 'dundee' 'brown'
    do
        CUDA_VISIBLE_DEVICES=-1 make process_data MODEL=${model} DATASET=${dataset}
        make get_llh MODEL=${model} DATASET=${dataset}
    done
done

for model in 'pythia-70m' 'pythia-160m' 'pythia-410m' 'pythia-14b' 'pythia-28b' 'pythia-69b'
do
    for language in 'en'
    do
        make get_llh -f MakefileWordLengths MODEL=${model}
    done
done
