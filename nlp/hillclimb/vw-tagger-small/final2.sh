#!/usr/bin/env bash
# Final runs with teacher forcing: retrain each winner once, keep the model, score dev and test.
set -eu
cd "$(dirname "$0")"
NF='--search_neighbor_features -1:w,1:w,-2:w,2:w,-1:s,1:s,-1:h,1:h'
R='--search_rollin ref --search_rollout none'
POS="$NF --predict_only_model -b 26 -q ln $R"
NER="--search_task sequence --search 37 --predict_only_model $R $NF -q ln"
HC_TEST=1 python3 hc.py final-s10a-gold w,l,s4,p3,h,f,n1 3 $POS --l1 7e-7
HC_TEST=1 python3 hc.py final-s10b-gold w,l,s4,p3,h,f,n1 3 $POS --l1 3e-7
HC_TEST=1 HC_DATA=data/spacy-ewt-work python3 hc.py final-s10a-spacy w,l,s4,p3,h,f,n1 3 $POS --l1 7e-7
HC_TEST=1 HC_DATA=data/spacy-ewt-work python3 hc.py final-s10b-spacy w,l,s4,p3,h,f,n1 3 $POS --l1 3e-7
HC_TEST=1 HC_DATA=data/spacy-ner-all-work python3 hc.py final-n5b-ner w,l,s4,p3,h,f,n1 3 $NER -b 24
HC_TEST=1 HC_DATA=data/spacy-ner-all-work python3 hc.py final-n6c-ner w,l,s4,p3,h,f,n1 3 $NER -b 26 --l1 2e-7
