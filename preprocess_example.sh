#!/bin/bash

# Instructions on how to preprocess the figurines dataset. Can probably preprocess other lerf datasets the same way. 
# Just change all occurences of the word 'figurines' below with any other scene from lerf_ovs directory.

mkdir -p data/figurines_processed

cp -r lerf_ovs/figurines/sparse data/figurines_processed

ns-process-data images --data lerf_ovs/figurines/images --output-dir data/figurines_processed --skip-colmap --colmap-model-path sparse/0
