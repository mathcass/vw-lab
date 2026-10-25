#!/bin/bash
# Train a model. TASK, PASSES, MODEL, SIZE and DATA can be overridden from the
# environment. Work folders are in DATA, the repo's shared data/ by default.
#   TASK=pos (default)  POS tagger on data/work (`make prepare`)
#     SIZE=accurate (default)  ~8.4 MB model, 93.5% test
#     SIZE=small               ~0.4 MB model, 90.9% test
#   TASK=ner            spaCy-distilled entity tagger on data/ner-work
#                       (`make ner-prepare`): ~9.3 MB, F1 71.8 vs spaCy on test,
#                       ~15 min to train
#   TASK=spacy-pos      tagger trained on spaCy's tags for EWT train text, on
#                       data/spacy-pos-work (`make spacy-pos-prepare`): ~8.3 MB,
#                       ~94.5% agreement with spaCy
#   TASK=sent           sentence splitter trained on spaCy's splits of whole
#                       paragraphs, on data/sent-work (`make sent-prepare`)
#   TASK=dep            spaCy-distilled dependency parser on data/dep-work
#                       (`make dep-prepare`): ~4.9 MB, LAS ~77.6 vs spaCy on
#                       test, ~13 min to train

# The vowpalwabbit wheel has no standalone `vw` binary; this is its CLI.
VW=(uv run python -m vowpalwabbit)

DATA="${DATA:-$(cd "$(dirname "$0")/../data" && pwd)}"
TASK="${TASK:-pos}"
PASSES="${PASSES:-3}"
SIZE="${SIZE:-accurate}"

case "$TASK" in
  pos)
    WORK=$DATA/work
    case "$SIZE" in
      accurate) L1=3e-7 ;;
      small) L1=7e-7 ;;
      *) echo "SIZE must be accurate or small, not $SIZE" >&2; exit 1 ;;
    esac ;;
  # 25x more training text than the tagger, so far more updates; L1 is
  # applied on each one, so it has to be much weaker for the same size.
  ner) WORK=$DATA/ner-work; L1=6e-8 ;;
  spacy-pos) WORK=$DATA/spacy-pos-work; L1=3e-7 ;;
  sent) WORK=$DATA/sent-work; L1="${L1:-3e-6}" ;;
  dep) WORK=$DATA/dep-work; L1=5e-8 ;;
  *) echo "TASK must be pos, ner, spacy-pos, sent or dep, not $TASK" >&2; exit 1 ;;
esac

TRAIN="$WORK/train.conllu"
MODEL="${MODEL:-$WORK/model.vw}"
TAGS="${TRAIN%.conllu}.tags"

N_TAGS=$(wc -l "$TAGS" | awk '{print $1}')
echo "TASK=$TASK: $N_TAGS labels, $PASSES passes, --l1 $L1"

if [ "$TASK" = dep ]; then
  # VW's transition-based parser (arc-hybrid). Labels are relations; the
  # parser builds its own feature crosses, so there are no -q or neighbor
  # features. Teacher forcing beat the parser's dynamic oracle (--cost_to_go)
  # and arc-eager in the parser hill-climb; -b 26 avoids collisions in its
  # crosses, and L1 sets the size as for the taggers.
  ROOT=$(awk -F'\t' '$1 == "root" {print $2}' "$TAGS")
  exec "${VW[@]}" -k -c -d "$TRAIN" --passes "$PASSES" \
     --search_task dep_parser --search $N_TAGS --num_label $N_TAGS --root_label "$ROOT" \
     --search_rollin ref --search_rollout none \
     -b 26 --l1 "$L1" --predict_only_model \
     --holdout_off \
     -f "$MODEL"
fi

# Settings chosen by the hill-climbs in ../.claude/hillclimb/ (see the README):
#   --search_rollin ref --search_rollout none
#                          teacher forcing: learn each step from the true
#                          previous tags, no rollouts. More accurate than VW's
#                          default search settings and ~7x faster to train
#   neighbor features      words two either side, plus suffixes and shapes one
#                          either side (namespaces come from conllu-to-vw.py)
#   -q ln                  lowercased word x the neighbor words in `n` (one
#                          either side for tags, two for entities); lets the
#                          model learn e.g. "to" + verb -> TO
#   -b 26 --l1             big hash table (few collisions), with L1 dropping
#                          the weights that don't matter; L1 sets the size
#   --predict_only_model   don't save the optimizer's training state, which
#                          would double the file and keep zeroed weights
#   3 passes
"${VW[@]}" -k -c -d "$TRAIN" --passes "$PASSES" \
   --search_task sequence --search $N_TAGS --search_alpha 1e-6 \
   --search_max_bias_ngram_length 2 --search_max_quad_ngram_length 1 \
   --search_rollin ref --search_rollout none \
   --search_neighbor_features -1:w,1:w,-2:w,2:w,-1:s,1:s,-1:h,1:h \
   -q ln -b 26 --l1 "$L1" --predict_only_model \
   --holdout_off \
   -f "$MODEL"
