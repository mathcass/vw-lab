#!/usr/bin/env python
"""Score VW sequence-tagger predictions against gold VW-format data.

Usage:
    evaluate.py GOLD PREDICTIONS --train TRAIN --tags TAGS

GOLD and TRAIN are files written by conllu-to-vw.py, PREDICTIONS is the
`vw -t -p` output (one line per sentence, space-separated label ids), and TAGS
maps tag names to label ids.

Also reports a most-frequent-tag baseline: each word gets the tag it had most
often in TRAIN, and unseen words get the most common tag overall.

When TAGS holds entity tags (B-ORG, I-ORG, ...; written by
`conllu-to-vw.py --label ner`), it scores entities instead: precision, recall
and F1 over exact spans (type and boundaries must both match; an I- tag that
doesn't continue a span starts a new one, as in conlleval), with a 95%
sentence-bootstrap interval for F1 and a per-type breakdown. --train isn't
needed then.

For sentence-split data (`conllu-to-vw.py --label sent`: one sequence per
paragraph, B = starts a sentence), it scores sentence starts: precision,
recall and F1 over every token but each paragraph's first (which always
starts one), with a 95% paragraph-bootstrap interval, and the share of
sentences found exactly. The baseline starts a sentence after `.`, `!` or
`?`.

For parser data (`HEAD,REL` labels, written by conllu-to-dep.py; TAGS is the
relation map), it scores attachment: UAS (right head) and LAS (right head and
relation), over all tokens and without punctuation, with 95%
sentence-bootstrap intervals. PREDICTIONS is then VW's `head:rel` output.
VW 9.x's parser leaves a sentence's last token unattached (`9999999:9999999`)
in most sentences; it's attached to the sentence's root first, as `punct` if
it's punctuation and `dep` otherwise (attach_unattached; pipeline.py does
the same), and the count is reported. --raw scores the output as is.
"""

import argparse
import random
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
            # "LABEL | word ..." or "LABEL |w word |l ..."; the word comes first.
            label, features = line.split("|", 1)
            word = features.split(" ")[1]
            sentence.append((int(label), word))
    if sentence:
        sentences.append(sentence)
    return sentences


def read_predictions(path):
    with open(path) as f:
        return [[int(float(x)) for x in line.split()] for line in f if line.strip()]


def spans(labels, id_to_tag):
    """Entity spans (type, start, end) from label ids."""
    out, cur = [], None
    for i, y in enumerate(labels + [None]):
        tag = "O" if y is None else id_to_tag.get(y, "O")
        if cur and (tag == "O" or tag.startswith("B-") or tag[2:] != cur[0]):
            out.append((cur[0], cur[1], i))
            cur = None
        if tag != "O" and cur is None:
            cur = (tag[2:], i)
    return out


def ner_report(gold, preds, id_to_tag, n_boot=1000):
    rows, tokens, correct = [], 0, 0
    by_type = Counter()
    for sentence, pred in zip(gold, preds):
        labels = [label for label, _ in sentence]
        if len(labels) != len(pred):
            raise SystemExit(f"length mismatch: {len(labels)} gold vs {len(pred)} predicted")
        tokens += len(labels)
        correct += sum(a == b for a, b in zip(labels, pred))
        g, p = set(spans(labels, id_to_tag)), set(spans(pred, id_to_tag))
        rows.append((len(g & p), len(g), len(p)))
        for kind, found in (("tp", g & p), ("gold", g), ("pred", p)):
            for span in found:
                by_type[span[0], kind] += 1

    def f1(idx):
        tp = sum(rows[i][0] for i in idx)
        return 200 * tp / max(sum(rows[i][1] + rows[i][2] for i in idx), 1)

    tp, n_gold, n_pred = (sum(r[k] for r in rows) for k in range(3))
    rng = random.Random(0)
    n = len(rows)
    boot = sorted(f1([rng.randrange(n) for _ in range(n)]) for _ in range(n_boot))
    lo, hi = boot[int(0.025 * n_boot)], boot[int(0.975 * n_boot) - 1]
    print(f"sentences            {len(gold):>8}")
    print(f"tokens               {tokens:>8}")
    print(f"gold entities        {n_gold:>8}")
    print(f"predicted entities   {n_pred:>8}")
    print()
    print(f"entity precision     {100 * tp / max(n_pred, 1):>7.2f}%")
    print(f"entity recall        {100 * tp / max(n_gold, 1):>7.2f}%")
    print(f"entity F1            {f1(range(n)):>7.2f}   (95% CI {lo:.2f}-{hi:.2f})")
    print(f"token accuracy       {100 * correct / tokens:>7.2f}%")
    print()
    print(f"{'type':12} {'gold':>5} {'F1':>6}")
    for kind in sorted({k for k, _ in by_type}, key=lambda k: -by_type[k, "gold"]):
        t = 200 * by_type[kind, "tp"] / max(by_type[kind, "gold"] + by_type[kind, "pred"], 1)
        print(f"{kind:12} {by_type[kind, 'gold']:>5} {t:>6.1f}")


def bootstrap(rows, n_boot=1000):
    """95% interval for sum(hits) / sum(totals), resampling sentences."""
    rng = random.Random(0)
    n = len(rows)
    boot = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        boot.append(100 * sum(rows[i][0] for i in idx) / max(sum(rows[i][1] for i in idx), 1))
    boot.sort()
    return boot[int(0.025 * n_boot)], boot[int(0.975 * n_boot) - 1]


def sent_report(gold, preds, start):
    def sentences(labels):
        starts = [0] + [i for i in range(1, len(labels)) if labels[i] == start]
        return set(zip(starts, starts[1:] + [len(labels)]))

    rows, base_rows = [], []
    exact = n_sent = 0
    for paragraph, pred in zip(gold, preds):
        labels = [label for label, _ in paragraph]
        if len(labels) != len(pred):
            raise SystemExit(f"length mismatch: {len(labels)} gold vs {len(pred)} predicted")
        words = [w for _, w in paragraph]
        base = [start] + [start if words[i - 1][-1:] in ".!?" else 0 for i in range(1, len(words))]
        for out, guess in ((rows, pred), (base_rows, base)):
            g = {i for i in range(1, len(labels)) if labels[i] == start}
            p = {i for i in range(1, len(labels)) if guess[i] == start}
            out.append((len(g & p), len(g), len(p)))
        gs, ps = sentences(labels), sentences([start] + pred[1:])
        exact += len(gs & ps)
        n_sent += len(gs)

    def prf(rs, idx=None):
        idx = range(len(rs)) if idx is None else idx
        tp, g, p = (sum(rs[i][k] for i in idx) for k in range(3))
        return 100 * tp / max(p, 1), 100 * tp / max(g, 1), 200 * tp / max(g + p, 1)

    rng = random.Random(0)
    n = len(rows)
    boot = sorted(prf(rows, [rng.randrange(n) for _ in range(n)])[2] for _ in range(1000))
    lo, hi = boot[25], boot[974]
    print(f"paragraphs           {len(gold):>8}")
    print(f"tokens               {sum(len(p) for p in gold):>8}")
    print(f"sentences            {n_sent:>8}")
    print(f"sentence starts      {sum(r[1] for r in rows):>8}   (not counting paragraph starts)")
    print()
    print(f"{'':20} {'model':>8} {'baseline':>9}")
    for k, name in enumerate(("start precision", "start recall", "start F1")):
        print(f"{name:20} {prf(rows)[k]:>7.2f}% {prf(base_rows)[k]:>8.2f}%")
    print(f"{'':20} (95% CI {lo:.2f}-{hi:.2f})")
    print(f"{'sentences exact':20} {100 * exact / n_sent:>7.2f}%")


def read_dep(path, sep):
    """Sentences of (head, rel, word); word is None for prediction files."""
    sentences, sentence = [], []
    with open(path) as f:
        for line in f:
            if not line.strip():
                if sentence:
                    sentences.append(sentence)
                sentence = []
                continue
            label, _, features = line.partition(" ")
            head, rel = label.split(sep)
            word = features.split("|w ", 1)[1].split(" ", 1)[0].strip() if "|w " in features else None
            sentence.append((int(head), int(rel), word))
    if sentence:
        sentences.append(sentence)
    return sentences


UNATTACHED = 9999999


def attach_unattached(heads, words, rel_to_id):
    """Attach tokens VW's parser left without a head (in practice, the last
    token) to the sentence's root: as punct if the word has no letters or
    digits, else dep. heads is a list of (head, rel id); returns a new list
    and how many were attached."""
    root = next((i + 1 for i, (h, _) in enumerate(heads) if h == 0), 0)
    out, n = [], 0
    for (h, r), word in zip(heads, words):
        if h == UNATTACHED:
            n += 1
            punct = not any(c.isalnum() for c in word)
            h, r = root, rel_to_id["punct" if punct else "dep"]
        out.append((h, r))
    return out, n


def dep_report(gold_path, pred_path, id_to_rel, raw=False):
    gold, pred = read_dep(gold_path, ","), read_dep(pred_path, ":")
    if len(gold) != len(pred):
        raise SystemExit(f"{len(gold)} gold sentences but {len(pred)} predicted")
    rel_to_id = {v: k for k, v in id_to_rel.items()}
    n_fixed = 0
    for k, (g, p) in enumerate(zip(gold, pred)):
        if not raw:
            fixed, n = attach_unattached([(h, r) for h, r, _ in p], [w for _, _, w in g], rel_to_id)
            pred[k] = [(h, r, None) for h, r in fixed]
            n_fixed += n
    gold = [[(h, r) for h, r, _ in s] for s in gold]
    pred = [[(h, r) for h, r, _ in s] for s in pred]
    rows = {k: [] for k in ("uas", "las", "uas_np", "las_np")}
    for g, p in zip(gold, pred):
        if len(g) != len(p):
            raise SystemExit(f"length mismatch: {len(g)} gold vs {len(p)} predicted")
        u = l = u_np = l_np = n_np = 0
        for (gh, gr), (ph, pr) in zip(g, p):
            u += gh == ph
            l += gh == ph and gr == pr
            if id_to_rel.get(gr) != "punct":
                n_np += 1
                u_np += gh == ph
                l_np += gh == ph and gr == pr
        rows["uas"].append((u, len(g)))
        rows["las"].append((l, len(g)))
        rows["uas_np"].append((u_np, n_np))
        rows["las_np"].append((l_np, n_np))
    print(f"sentences            {len(gold):>8}")
    print(f"tokens               {sum(len(g) for g in gold):>8}")
    if not raw:
        print(f"unattached, fixed    {n_fixed:>8}")
    print()
    for name, key in (("UAS", "uas"), ("LAS", "las"),
                      ("UAS without punct", "uas_np"), ("LAS without punct", "las_np")):
        r = rows[key]
        score = 100 * sum(h for h, _ in r) / max(sum(t for _, t in r), 1)
        lo, hi = bootstrap(r)
        print(f"{name:20} {score:>7.2f}   (95% CI {lo:.2f}-{hi:.2f})")


def main():
    parser = argparse.ArgumentParser(description="Score VW tagger predictions.")
    parser.add_argument("gold", type=Path)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--train", type=Path, help="training data (needed for tags, not entities)")
    parser.add_argument("--tags", type=Path, required=True)
    parser.add_argument("--top-errors", type=int, default=8)
    parser.add_argument("--raw", action="store_true",
                        help="parser: score VW's output without attaching unattached tokens")
    args = parser.parse_args()

    id_to_tag = {}
    with open(args.tags) as f:
        for line in f:
            tag, id_ = line.rstrip("\n").split("\t")
            id_to_tag[int(id_)] = tag

    with open(args.gold) as f:
        if "," in f.readline().split(" ", 1)[0]:
            return dep_report(args.gold, args.predictions, id_to_tag, args.raw)

    gold = read_vw(args.gold)
    preds = read_predictions(args.predictions)
    if len(gold) != len(preds):
        raise SystemExit(f"{len(gold)} gold sentences but {len(preds)} predictions")
    if set(id_to_tag.values()) == {"I", "B"}:
        return sent_report(gold, preds, next(i for i, t in id_to_tag.items() if t == "B"))
    if any(tag.startswith("B-") for tag in id_to_tag.values()):
        return ner_report(gold, preds, id_to_tag)
    if not args.train:
        parser.error("--train is required for tag accuracy")
    train = read_vw(args.train)

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
