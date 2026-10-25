#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")"
B=$(grep '^e5a' e5.cfg | cut -d'|' -f4)
eval "extra=($B)"
HC_TEST=1 HC_DATA=data/spacy-ner-wiki10-work python3 hc.py e5a-wiki10-l1x27e9 w,l,s4,p3,h,f,n2 3 "${extra[@]}"
tail -1 test-results-ner.tsv
