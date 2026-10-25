#!/bin/sh
# Cold-start cascade simulation, all fold-0 held-out domains (see vwjev.cascade)
set -e
cd "$(dirname "$0")/.."   # decisions/
for d in "Spam detection" "Tool and workflow decisions" "Sentiment, emotion, and moderation" \
         "Rule-conditioned decisions" "Calibrated probability" "Sensory quality rating" \
         "Response preference and quality"; do
  systemd-run --user --scope -q -p MemoryMax=3G uv run python -I -m vwjev.cascade run --domain "$d"
done
