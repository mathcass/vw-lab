#!/bin/bash
# Train one non-VW reference (gensim word2vec or PPMI+SVD), score it, and log
# its time and peak memory.
#
#   baseline.sh NAME METHOD [baselines.py ARGS...]
#
#   METHOD  w2v (sentences in memory), w2v_stream (streams from disk), ppmi
#
# Environment: W2V_WORK (corpus dir, default data/word2vec-work), OUT (default
# $W2V_WORK/runs), CAP (memory cap such as 3G, see train.sh).
# Appends NAME to results/results.tsv and results/memory.tsv.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="uv run --project $HERE python"
WORK="${W2V_WORK:-$HERE/../data/word2vec-work}"
OUT="${OUT:-$WORK/runs}"
NAME="$1"; METHOD="$2"; shift 2
mkdir -p "$OUT"
SCOPE=""
[ -n "${CAP:-}" ] && SCOPE="systemd-run --user --scope -q -p MemoryMax=$CAP -p MemorySwapMax=0 --"

start=$(date +%s)
# shellcheck disable=SC2086
$SCOPE /usr/bin/time -f "maxrss %M" -o "$OUT/$NAME.time" \
  $PY "$HERE/baselines.py" "$METHOD" "$OUT/$NAME.npz" --work "$WORK" "$@" > "$OUT/$NAME.log" 2>&1
status=$?
secs=$(( $(date +%s) - start ))
train=$(grep -o 'train_s=[0-9]*' "$OUT/$NAME.log" | cut -d= -f2)
mb=$(awk '/maxrss/ {printf "%.0f", $2 / 1024}' "$OUT/$NAME.time" 2>/dev/null)
state=ok; [ $status != 0 ] && state="failed (exit $status, see $OUT/$NAME.log)"
case "$(basename "$WORK")" in word2vec-work) corpus=10M ;; word2vec-big) corpus=100M ;; *) corpus=$(basename "$WORK") ;; esac
printf "%s\t%s\t%s\t%s\t-\t%s\t%s\t%s\t%s\t%s\n" "$NAME" "$corpus" "$(wc -l < "$WORK/vocab.tsv")" \
  "${train:-$secs}" "${mb:--}" "${mb:--}" "$state" "baselines.py $METHOD $*" \
  "$(date '+%Y-%m-%d %H:%M')" >> "$HERE/results/memory.tsv"
echo "$NAME: ${train:-$secs}s, peak memory ${mb:-?} MB [$state]"
[ $status != 0 ] && exit 1
$PY "$HERE/evalvec.py" "$OUT/$NAME.npz" --name "$NAME" --train-s "${train:-$secs}" \
  --notes "baselines.py $METHOD $* | corpus $corpus | peak MB ${mb:-?}"
