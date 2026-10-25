#!/bin/bash
# Train VW --lda K on paragraph (or chunk) documents, extract word-topic vectors, evaluate.
#   train_lda.sh NAME K "GEN_DOCS ARGS" "VW_ARGS" [NOTES]
# GEN_DOCS ARGS go to gen_docs.py (--max-paras, --chunk, --passes). --lda_D is
# set to the number of documents per pass.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
TOP="$(dirname "$HERE")"
PY="uv run --project $TOP python"
NAME="$1"; K="$2"; GEN="$3"; VWARGS="$4"; NOTES="${5:-}"
W="${OUT:-${W2V_WORK:-$TOP/../data/word2vec-work}/runs}"
mkdir -p "$W"
# shellcheck disable=SC2086
D=$($PY "$HERE/gen_docs.py" $GEN --count-docs)
start=$(date +%s)
# shellcheck disable=SC2086
nice -n 19 $PY "$HERE/gen_docs.py" $GEN \
  | nice -n 19 $PY -m vowpalwabbit -d /dev/stdin --lda "$K" --lda_D "$D" -b 15 $VWARGS \
      -f "$W/$NAME.model" 2> "$W/$NAME.log"
secs=$(( $(date +%s) - start ))
$PY "$HERE/extract_lda.py" "$W/$NAME.model" "$K" "$W/$NAME" > /dev/null
for s in ptw hel; do
  $PY "$TOP/evalvec.py" "$W/$NAME.$s.npz" --name "$NAME.$s" --train-s "$secs" \
    --notes "lda | docs $D | gen: $GEN | vw: $VWARGS | $NOTES"
done
rm -f "$W/$NAME.model"
