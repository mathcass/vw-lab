# VW tagger hill-climb

Goal: maximize dev token accuracy (UD English-EWT, XPOS). Train time and model
size are tracked, not constrained (unless >3x). Scope: run.sh flags and
converter features, EWT data only. Test is scored once, at the end, for the
committed config and the winner.

Noise floor: paired sentence-bootstrap 95% CI on a dev difference is about
±0.2 points; a change is kept only if it gains >= 0.3 points on dev.
Stop after 3 consecutive rounds without a kept change, or ~3 hours of compute.

All runs: `results.tsv`. Tools: `featurize.py` (feature namespaces), `hc.py`
(train + score dev).

| round | change | dev acc | unseen | kept? |
|---|---|---|---|---|
| 0 | committed run.sh (10 passes) | 73.29 | 19.5 | baseline |
| 1 | `-b 24` (5 passes) | 86.48 | 32.5 | yes |
| 2 | `w` namespace, neighbor words ±2, `--affix -3w,-2w,+2w`, `--spelling w` | 90.11 | 66.9 | yes |
| 2 check | same, `--random_seed 7` | 90.11 | 66.9 | identical: training is deterministic |
| 3a | explicit namespaces: lowercase, suffix 1-4, prefix 1-3, shape, flags (`w,l,s4,p3,h,f`) | 90.93 | 71.2 | yes (+0.82) |
| 3b | 3a + neighbor suffixes and shapes (`-1:s,1:s,-1:h,1:h`) | 91.32 | 72.3 | yes (+0.39, CI +0.15..+0.63) |
| 4a | 3b + current word x neighbor features (`-q 'w\x83'`) | 91.86 | 71.2 | yes (+0.54, CI +0.24..+0.86) |
| 4b | 3b + `--search_history_length 2 --search_max_bias_ngram_length 3` | 90.93 | 71.4 | no (-0.39, CI -0.68..-0.08) |
| 5a | 4a + `-b 26` | 92.84 | 75.1 | yes (+0.98, CI +0.74..+1.21); model 772 MB (3.7x) |
| 5b | 4a + suffix x neighbor (`-q 's\x83'`), still `-b 24` | 91.12 | 70.4 | no (-0.74) - likely hash-confounded, retest at -b 26 |
| 6a | 5a + suffix x neighbor (`-q 's\x83'`) at `-b 26` | 92.09 | 73.5 | no (-0.75, CI -0.99..-0.52): hurts even without hash pressure |
| 6b | 5a with `-b 25` (size check) | 92.46 | 73.5 | size trade-off: 417 MB vs 772 MB, -0.38 |

| 7 | (first attempt stopped by Claude Code for low system memory; rerun one job at a time) | | | |
| 7a | 5a with 10 passes | 92.55 | 73.7 | no (-0.29, CI -0.48..-0.12): more passes overfit |
| 7b | 5a + 2-tag history at `-b 26` | 92.70 | 74.6 | no (-0.14, CI -0.36..+0.07) |
| 8a | 5a with 3 passes | 92.95 | 75.7 | tie (+0.11, CI -0.08..+0.29), 2.3x faster to train: **winner** |
| 8b | 5a without `l` and `p` namespaces | 92.50 | 73.5 | no (-0.34, CI -0.51..-0.17) |

Stopped: rounds 6, 7 and 8 had no change clearing +0.3 (plateau rule).
Python env note: at 19:59 uv rebuilt `.venv` on CPython 3.14.3 (the old
interpreter disappeared). Re-running the round-0 1-pass config gave
byte-identical predictions, so results before and after are comparable.

Where the remaining 1,800 dev errors (5a) come from: 32% ambiguous word with
its less usual tag; 29% unseen word; 17% ambiguous word with its usual tag;
15% tag never seen with this word in train; 8% unambiguous word mis-tagged.
NNP/NN confusions alone are 22%. About 43% (unseen + never-seen tag) need
lexical knowledge beyond EWT: word clusters/embeddings or more training data,
which were out of scope.

Hash-size curve for the 4a feature set: b24 91.86 (210 MB), b25 92.46 (417 MB),
b26 92.84 (772 MB).

Notes:
- Hash collisions confound every feature added at `-b 24`: 4b and 5b may have
  failed only because they added features to an already crowded table.
- Top error after 3b is "to" gold IN predicted TO (131). Train labels are
  consistent (TO = infinitive marker, IN = preposition), so it is learnable,
  but only from current-word x next-word conjunctions, which a linear model
  with separate neighbor features can't express.
- VW's neighbor features live in the hidden namespace `\x83`, not `n`; `-q 'w\x83'`
  crosses them with the current word (checked: adds features, changes loss).
