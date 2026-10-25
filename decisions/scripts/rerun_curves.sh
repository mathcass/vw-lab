#!/bin/bash
# reruns after the load_pretrained fix; one job at a time (memory)
set -u
cd "$(dirname "$0")/.."   # decisions/
uv run python -I -m vwjev.lockbox_curve --model runs/v6_generic_shuf --openai luna_f0 > runs/lockbox_curve_v6.log 2>&1 && echo "lockbox curve done" || echo "lockbox curve FAILED"
for d in "Spam detection" "Tool and workflow decisions" "Sentiment, emotion, and moderation" "Rule-conditioned decisions" "Calibrated probability" "Response preference and quality" "Sensory quality rating"; do
  uv run python -I -m vwjev.feedback --model runs/v6_generic_shuf --domain "$d" >> runs/feedback_v6_fixed.log 2>&1 && echo "feedback done: $d" || echo "feedback FAILED: $d"
done
echo "ALL DONE"
