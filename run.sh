#!/bin/bash
# Train the tagger. PASSES and MODEL can be overridden from the environment.

# The vowpalwabbit wheel has no standalone `vw` binary; this is its CLI.
VW=(uv run python -m vowpalwabbit)

TRAIN="./data/work/train.conllu"
MODEL="${MODEL:-./data/work/model.vw}"
PASSES="${PASSES:-3}"
TAGS="${TRAIN%.conllu}.tags"

N_TAGS=$(wc -l "$TAGS" | awk '{print $1}')
echo "Using $N_TAGS tags, $PASSES passes"

# Settings chosen by the hill-climb in .claude/hillclimb/vw-tagger/ (see the
# README benchmark):
#   -b 26                  big hash table; at the default 18 bits, collisions
#                          between features cost ~13 points of accuracy
#   neighbor features      words two either side, plus suffixes and shapes one
#                          either side (namespaces come from conllu-to-vw.py)
#   -q 'w\x83'             current word x neighbor features; \x83 is the hidden
#                          namespace VW puts neighbor features in. Lets the
#                          model learn e.g. "to" + verb -> TO, "to" + noun -> IN
#   3 passes               as accurate as 5, and more passes overfit
"${VW[@]}" -k -c -d "$TRAIN" --passes "$PASSES" \
   --search_task sequence --search $N_TAGS --search_alpha 1e-6 \
   --search_max_bias_ngram_length 2 --search_max_quad_ngram_length 1 \
   --search_neighbor_features -1:w,1:w,-2:w,2:w,-1:s,1:s,-1:h,1:h \
   -q 'w\x83' -b 26 \
   --holdout_off \
   -f "$MODEL"
