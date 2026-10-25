#!/bin/bash
cd "$(dirname "$0")/.."   # decisions/
export HF_HUB_OFFLINE=1
for s in lockbox dev_s400 seen_test_s400; do
  .venv-gliner/bin/python -I bench/gliner_decide.py data/processed/eval_f0_$s.jsonl runs/gliner/f0/preds_$s.jsonl 2>&1 | grep -v -i -E "warn|instantiate a model" && echo "gliner done: $s" || echo "gliner FAILED: $s"
done
echo "GLINER ALL DONE"
