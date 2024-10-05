#!/bin/bash

for model in 'gpt-small' 'gpt-medium' 'gpt-large' 'gpt-xl' \
             'pythia-70m' 'pythia-160m' 'pythia-410m' 'pythia-14b' 'pythia-28b' 'pythia-69b'
            #   'pythia-120b'
do
    # for dataset in 'natural_stories' 'brown' 'provo_skip2zero' 'dundee_skip2zero'   
    for dataset in 'natural_stories'
    do
        make process_data MODEL=${model} DATASET=${dataset}
        make get_llh MODEL=${model} DATASET=${dataset}
    done
done

# for model in 'pythia-70m' 'pythia-160m' 'pythia-410m' 'pythia-14b' 'pythia-28b' 'pythia-69b'
# do
#     for dataset in 'wiki40b'   
#     do
#         make get_length_predictions MODEL=${model} DATASET=${dataset}
#     done
# done