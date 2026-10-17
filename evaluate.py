#!/usr/bin/env python
"""Score VW sequence-tagger predictions against gold VW-format data.

Usage:
    evaluate.py GOLD PREDICTIONS --train TRAIN --tags TAGS

GOLD and TRAIN are files written by conllu-to-vw.py, PREDICTIONS is the
`vw -t -p` output (one line per sentence, space-separated label ids), and TAGS
maps tag names to label ids.

Also reports a most-frequent-tag baseline: each word gets the tag it had most
often in TRAIN, and unseen words get the most common tag overall.
"""

import argparse
from collections import Counter, defaultdict
from pathlib import Path


def read_vw(path):
    sentences, sentence = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                if sentence:
                    sentences.append(sentence)
                sentence = []
                continue
            # "LABEL | word" or "LABEL |namespace word"
            label, features = line.split("|", 1)
            word = features.split(" ", 1)[1]
            sentence.append((int(label), word))
    if sentence:
        sentences.append(sentence)
    return sentences


def read_predictions(path):
    with open(path) as f:
        return [[int(float(x)) for x in line.split()] for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser(description="Score VW tagger predictions.")
    parser.add_argument("gold", type=Path)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--tags", type=Path, required=True)
    parser.add_argument("--top-errors", type=int, default=8)
    args = parser.parse_args()

    id_to_tag = {}
    with open(args.tags) as f:
        for line in f:
            tag, id_ = line.rstrip("\n").split("\t")
            id_to_tag[int(id_)] = tag

    train = read_vw(args.train)
    gold = read_vw(args.gold)
    preds = read_predictions(args.predictions)
    if len(gold) != len(preds):
        raise SystemExit(f"{len(gold)} gold sentences but {len(preds)} predictions")

    word_tags = defaultdict(Counter)
    all_tags = Counter()
    for sentence in train:
        for label, word in sentence:
            word_tags[word][label] += 1
            all_tags[label] += 1
    default_tag = all_tags.most_common(1)[0][0]

    def baseline(word):
        counts = word_tags.get(word)
        return counts.most_common(1)[0][0] if counts else default_tag

    tokens = correct = base_correct = 0
    oov = oov_correct = oov_base_correct = 0
    sent_correct = 0
    confusions = Counter()

    for sentence, pred in zip(gold, preds):
        if len(sentence) != len(pred):
            raise SystemExit(f"length mismatch: {len(sentence)} gold vs {len(pred)} predicted")
        all_right = True
        for (label, word), p in zip(sentence, pred):
            tokens += 1
            hit = p == label
            base_hit = baseline(word) == label
            correct += hit
            base_correct += base_hit
            if word not in word_tags:
                oov += 1
                oov_correct += hit
                oov_base_correct += base_hit
            if not hit:
                all_right = False
                confusions[(id_to_tag[label], id_to_tag.get(p, str(p)))] += 1
        sent_correct += all_right

    pct = lambda n, d: 100 * n / d if d else float("nan")
    print(f"sentences            {len(gold):>8}")
    print(f"tokens               {tokens:>8}")
    print(f"unseen-word tokens   {oov:>8}  ({pct(oov, tokens):.1f}%)")
    print()
    print(f"{'':20} {'model':>8} {'baseline':>9}")
    print(f"{'token accuracy':20} {pct(correct, tokens):>7.2f}% {pct(base_correct, tokens):>8.2f}%")
    print(f"{'seen-word accuracy':20} {pct(correct - oov_correct, tokens - oov):>7.2f}% "
          f"{pct(base_correct - oov_base_correct, tokens - oov):>8.2f}%")
    print(f"{'unseen-word accuracy':20} {pct(oov_correct, oov):>7.2f}% {pct(oov_base_correct, oov):>8.2f}%")
    print(f"{'sentence accuracy':20} {pct(sent_correct, len(gold)):>7.2f}%")
    print()
    print("most common errors (gold -> predicted):")
    for (g, p), n in confusions.most_common(args.top_errors):
        print(f"  {g:>6} -> {p:<6} {n:>5}")


if __name__ == "__main__":
    main()
