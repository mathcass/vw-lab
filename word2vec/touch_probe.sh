#!/bin/bash
# Resident memory of a VW embedding model once every word's slots are in use.
#
#   touch_probe.sh NAME VOCAB.tsv "VW MODEL ARGS" [ids|words] [CTX_NS]
#
# VW reserves its whole weight table at start-up but Linux only counts pages
# that get written, so RSS creeps up as new words are first seen and a short
# run undercounts. This feeds one example per vocabulary word ("1 |w X |c X"),
# which touches every slot a full training run would, and reports VW's peak
# RSS (and the reserved size, VmPeak). Appends a row to results/memory.tsv.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="uv run --project $HERE python"
NAME=$1; VOCAB=$2; ARGS=$3; MODE=${4:-ids}; CNS=${5:-c}
T="${OUT:-${W2V_WORK:-$HERE/../data/word2vec-work}/runs}"
mkdir -p "$T"
start=$(date +%s)
awk -F'\t' -v m="$MODE" -v c="$CNS" '{x = (m == "words") ? $2 : $1; print "1 |w " x " |" c " " x}' "$VOCAB" \
  | systemd-run --user --scope -q -p MemoryMax="${CAP:-3G}" -p MemorySwapMax=0 -- \
    /usr/bin/time -f "maxrss %M" -o "$T/$NAME.vw.time" \
    $PY -c "
import sys
from vowpalwabbit import Workspace
vw = Workspace('$ARGS --quiet --holdout_off')
for line in sys.stdin:
    vw.learn(line)
vm = [l for l in open('/proc/self/status') if l.startswith(('VmPeak', 'VmHWM'))]
print(' '.join(l.split()[0] + l.split()[1] for l in vm))
" > "$T/$NAME.vm"
secs=$(( $(date +%s) - start ))
rss=$(awk '/maxrss/ {printf "%.0f", $2 / 1024}' "$T/$NAME.vw.time")
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$NAME" "-" "$(wc -l < "$VOCAB")" "$secs" - "$rss" "$rss" \
  "touch probe ($(cat "$T/$NAME.vm"))" "vw $ARGS ($MODE, ctx ns $CNS)" "$(date '+%Y-%m-%d %H:%M')" >> "$HERE/results/memory.tsv"
echo "$NAME: VW peak RSS ${rss}MB; $(cat "$T/$NAME.vm")"
