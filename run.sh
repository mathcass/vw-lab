#!/bin/bash

TRAIN="./data/work/train.conllu"
MODEL="./data/work/model.vw"
TAGS="${TRAIN%.conllu}.tags"

N_TAGS=$(wc -l "$TAGS" | awk '{print $1}')
echo "Using $N_TAGS tags"

vw -k -c -d "$TRAIN" --passes 100 \
   --search_task sequence --search $N_TAGS --search_alpha 1e-6 \
   --search_max_bias_ngram_length 2 --search_max_quad_ngram_length 1 \
   --holdout_off \
   -f "$MODEL"
