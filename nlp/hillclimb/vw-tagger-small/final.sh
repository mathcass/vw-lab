#!/usr/bin/env bash
# Final runs: retrain each winner once, keep the model, score dev and test.
set -eu
cd "$(dirname "$0")"
NF='--search_neighbor_features -1:w,1:w,-2:w,2:w,-1:s,1:s,-1:h,1:h'
SMALL="$NF --predict_only_model -b 26 --l1 4e-8 -q ln"
HC_TEST=1 python3 hc.py final-5b-gold w,l,s4,p3,h,f,n1 3 $SMALL
HC_TEST=1 HC_DATA=data/spacy-ewt-work python3 hc.py final-d1c-spacy w,l,s4,p3,h,f,n1 3 $SMALL
