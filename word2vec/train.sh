#!/bin/bash
# Train one VW embedding model from a streamed example generator, then
# extract the word vectors and score them.
#
#   train.sh NAME K "GENERATOR ARGS" "VW ARGS" [NOTES]
#
#   NAME            run name: vectors go to $OUT/NAME.{w,c,wc}.npz
#   K               rank (vector size)
#   GENERATOR ARGS  script + args, e.g. "gen.py cbow --epochs 5"
#   VW ARGS         loss, optimizer and learning-rate flags
#
# Environment:
#   ARCH    lrq (default: --lrq wcK) | gdmf (-q wc --rank K)
#   BITS    -b; default 25 for lrq, 18 for gdmf. gd_mf's slot is
#           2^ceil(log2(2K+1)) floats, so -b 25 would need 16 GB.
#   VOCAB   words to extract (default $W2V_WORK/vocab.tsv)
#   NS      namespaces to extract: "wc" (default) or e.g. "w,cb"
#   HASHED  1 if the generator writes words instead of ids (gen.py --words,
#           gen_online.py); extraction then hashes the words like VW does
#   INIT    start from this model (experiments/init_model.py); its stored
#           --lrq/-b are used, so they are not passed again
#   CAP     memory cap such as 3G: run in a systemd scope with swap off, so a
#           run that needs more is killed instead of swapping
#   PROBE   stop after this many seconds (memory probe; no extraction)
#   VOCAB_LABEL  vocabulary size to log in memory.tsv if not VOCAB's (one-pass runs)
#
# Appends NAME.w / .c / .wc rows to results/results.tsv (train_s is wall time
# for generator + VW, which run concurrently on one core each) and one row
# with peak memory to results/memory.tsv.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="uv run --project $HERE python"
WORK="${W2V_WORK:-$HERE/../data/word2vec-work}"
OUT="${OUT:-$WORK/runs}"
NAME="$1"; K="$2"; GEN="$3"; VWARGS="$4"; NOTES="${5:-}"
ARCH="${ARCH:-lrq}"
VOCAB="${VOCAB:-$WORK/vocab.tsv}"
mkdir -p "$OUT"

if [ "$ARCH" = gdmf ]; then
  MODEL_ARGS="-q wc --rank $K -b ${BITS:-18}"; EXTRACT_ARGS="--gdmf"
else
  MODEL_ARGS="--lrq wc$K -b ${BITS:-25}"; EXTRACT_ARGS=""
fi
[ -n "${INIT:-}" ] && MODEL_ARGS="-i $INIT"
[ "${HASHED:-0}" = 1 ] && EXTRACT_ARGS="$EXTRACT_ARGS --hashed"
SAVE="-f $OUT/$NAME.model"; LIMIT=""
if [ -n "${PROBE:-}" ]; then SAVE=""; LIMIT="timeout $PROBE"; fi
SCOPE=""
[ -n "${CAP:-}" ] && SCOPE="systemd-run --user --scope -q -p MemoryMax=$CAP -p MemorySwapMax=0 --"

# python -m vowpalwabbit does not read stdin unless told to (-d /dev/stdin).
# VW prints "NAN prediction" warnings on stdout, one per example: hundreds of
# millions of lines when a run diverges, so they are counted, not kept.
start=$(date +%s)
# shellcheck disable=SC2086
$SCOPE bash -c "set -o pipefail
  /usr/bin/time -f 'maxrss %M' -o $OUT/$NAME.gen.time $LIMIT nice -n 19 $PY $HERE/$GEN \
  | /usr/bin/time -f 'maxrss %M' -o $OUT/$NAME.vw.time nice -n 19 \
      $PY -m vowpalwabbit -d /dev/stdin $MODEL_ARGS --holdout_off $VWARGS $SAVE 2> $OUT/$NAME.log \
  | awk '/NAN/ {n++} END {print n + 0}' > $OUT/$NAME.nan"
status=$?
secs=$(( $(date +%s) - start ))

mb() { awk '/maxrss/ {printf "%.0f", $2 / 1024}' "$1" 2>/dev/null; }
gen_mb=$(mb "$OUT/$NAME.gen.time"); vw_mb=$(mb "$OUT/$NAME.vw.time")
state=ok
if [ -n "${PROBE:-}" ]; then state="probe ${PROBE}s"
elif [ $status != 0 ]; then state="failed (exit $status)"; fi
nan=$(cat "$OUT/$NAME.nan" 2>/dev/null || echo 0)
[ "$nan" != 0 ] && state="$state, $nan NaN predictions"
case "$(basename "$WORK")" in word2vec-work) corpus=10M ;; word2vec-big) corpus=100M ;; *) corpus=$(basename "$WORK") ;; esac
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$NAME" "$corpus" \
  "${VOCAB_LABEL:-$(wc -l < "$VOCAB")}" "$secs" "${gen_mb:--}" "${vw_mb:--}" "$(( ${gen_mb:-0} + ${vw_mb:-0} ))" \
  "$state" "$GEN | vw $MODEL_ARGS $VWARGS" "$(date '+%Y-%m-%d %H:%M')" >> "$HERE/results/memory.tsv"
echo "$NAME: ${secs}s, peak memory generator ${gen_mb:-?} MB + VW ${vw_mb:-?} MB [$state]"
[ -n "${PROBE:-}" ] && exit 0
[ $status != 0 ] && { echo "see $OUT/$NAME.log" >&2; exit 1; }

loss=$(grep -E '^average loss' "$OUT/$NAME.log" | awk '{print $4}')
[ "$nan" != 0 ] && loss="$loss, $nan NaN-prediction warnings"
# shellcheck disable=SC2086
$PY "$HERE/extract.py" "$OUT/$NAME.model" "$K" "$OUT/$NAME" $EXTRACT_ARGS \
  --ns "${NS:-wc}" --vocab "$VOCAB"
for s in w c wc; do
  [ -f "$OUT/$NAME.$s.npz" ] || continue
  $PY "$HERE/evalvec.py" "$OUT/$NAME.$s.npz" --name "$NAME.$s" --train-s "$secs" \
    --notes "loss $loss | $ARCH | gen: $GEN | vw: $VWARGS | peak MB gen ${gen_mb:-?} vw ${vw_mb:-?} | $NOTES"
done
rm -f "$OUT/$NAME.model"
