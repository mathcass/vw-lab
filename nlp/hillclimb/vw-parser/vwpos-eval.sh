#!/usr/bin/env bash
# Score a kept parser model on dev/test with VW-predicted POS tags. Usage: vwpos-eval.sh RUN
S="${HC_SCRATCH:-../../../data/hillclimb/dep}"
cd "$(dirname "$0")"
for s in dev test; do
  uv run --project ../.. python -m vowpalwabbit -t -i $S/runs/$1/model.vw -d "$S/data/$s-w,p-posvw.vw" -p $S/runs/$1/$s-posvw.pred --quiet
  python3 -c "import hc; print('$1 $s vw-pos', ' '.join(f'{x:.2f}' for x in hc.score('$S/data/$s-w,p-posvw.vw', '$S/runs/$1/$s-posvw.pred', None)))"
done
