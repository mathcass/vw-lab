#!/usr/bin/env bash
# Run parser configs one at a time (memory is tight). Usage: round.sh CONFIGFILE
# Each line: NAME|FEATS|PASSES|VW ARGS. Lines starting with # are skipped.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
while IFS='|' read -r name feats passes args; do
  [[ -z "$name" || "$name" == \#* ]] && continue
  if grep -q "^$name	" "$HERE/results.tsv" "$HERE/test-results.tsv" 2>/dev/null; then echo "skip $name (done)"; continue; fi
  eval "extra=($args)"
  echo "== $name $(date +%H:%M)"
  python3 "$HERE/hc.py" "$name" "$feats" "$passes" "${extra[@]}" || echo "FAILED $name"
done < "$1"
