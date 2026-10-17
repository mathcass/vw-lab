#!/bin/bash
# Train the tagger. PASSES and MODEL can be overridden from the environment.

# The vowpalwabbit wheel has no standalone `vw` binary; this is its CLI.
VW=(uv run python -m vowpalwabbit)

TRAIN="./data/work/train.conllu"
MODEL="${MODEL:-./data/work/model.vw}"
PASSES="${PASSES:-100}"
TAGS="${TRAIN%.conllu}.tags"

N_TAGS=$(wc -l "$TAGS" | awk '{print $1}')
echo "Using $N_TAGS tags, $PASSES passes"

"${VW[@]}" -k -c -d "$TRAIN" --passes "$PASSES" \
   --search_task sequence --search $N_TAGS --search_alpha 1e-6 \
   --search_max_bias_ngram_length 2 --search_max_quad_ngram_length 1 \
   --holdout_off \
   -f "$MODEL"
