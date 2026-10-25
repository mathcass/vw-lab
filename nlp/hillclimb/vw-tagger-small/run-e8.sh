#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")"
B=$(grep '^e8a' e8-c4.cfg | cut -d'|' -f4)
eval "extra=($B)"
HC_TEST=1 HC_DATA=data/spacy-ner-c4-work python3 hc.py e8a-c4-4m-l1x6e8 w,l,s4,p3,h,f,n2 3 "${extra[@]}"
tail -1 test-results-ner.tsv
