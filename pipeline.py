#!/usr/bin/env python
"""Run raw text through the VW models: sentences, tags, entities, parses.

Usage:
    uv run --group spacy python pipeline.py INPUT.txt OUTPUT.conllu

INPUT.txt has one paragraph per line (a leading `# text = ` is dropped and
blank lines skipped, as in spacy-label.py). The steps:

  1. tokens     spaCy's rule-based English tokenizer (`spacy.blank("en")`:
                the spaCy library, but no trained model)
  2. sentences  data/sent-work/model.vw   (TASK=sent)
  3. tags       data/spacy-pos-work/model.vw   (TASK=spacy-pos)
  4. entities   data/ner-work/model.vw   (TASK=ner)
  5. parses     data/dep-work/model.vw   (TASK=dep), given the tags from 3;
                a token the parser leaves unattached (VW leaves most
                sentences' last token so) goes to the root, as in evaluate.py

Each VW step writes its input with the repo's converters (conllu-to-vw.py,
conllu-to-dep.py) and runs `vw -t` on the whole file. OUTPUT is CoNLL-U in
spacy-label.py's format (XPOS, HEAD, DEPREL, NER= and SpaceAfter=No in MISC,
`# newpar` on each paragraph's first sentence), so compare.py can score it
against spaCy's own output for the same text. Timings go to stderr.
"""

import subprocess
import sys
import tempfile
import time
from pathlib import Path

import spacy

from evaluate import attach_unattached

REPO = Path(__file__).resolve().parent
MODELS = {
    "sent": REPO / "data/sent-work",
    "pos": REPO / "data/spacy-pos-work",
    "ner": REPO / "data/ner-work",
    "dep": REPO / "data/dep-work",
}


def read_tags(path):
    return {int(v): k for k, v in (l.split("\t") for l in path.read_text().splitlines())}


def run(*cmd):
    subprocess.run([sys.executable, *map(str, cmd)], check=True)


def vw(work, data, pred):
    run("-m", "vowpalwabbit", "-t", "-i", work / "model.vw", "-d", data, "-p", pred, "--quiet")


def read_predictions(path):
    """Sequence-task output: one line of label ids per sequence."""
    return [[int(float(x)) for x in line.split()] for line in open(path) if line.strip()]


def read_dep_predictions(path):
    """dep_parser output: `head:rel` per token, a blank line between sentences."""
    out, cur = [], []
    for line in open(path):
        if not line.strip():
            if cur:
                out.append(cur)
            cur = []
            continue
        head, rel = line.split(":")
        cur.append((int(head), int(rel)))
    if cur:
        out.append(cur)
    return out


def write_conllu(path, sentences, xpos=None, heads=None, ner=None, newpar=None):
    """sentences: lists of (word, space_after). Unknown columns are written as
    placeholders the converters accept (they're ignored by `vw -t`)."""
    with open(path, "w", encoding="utf-8") as f:
        for k, sent in enumerate(sentences):
            if newpar and newpar[k]:
                f.write("# newpar\n")
            text = "".join(w + (" " if sp else "") for w, sp in sent).strip()
            f.write(f"# sent_id = vw-{k + 1}\n# text = {text}\n")
            for i, (word, space) in enumerate(sent):
                tag = xpos[k][i] if xpos else "_"
                head, rel = heads[k][i] if heads else ("_", "_")
                misc = [f"NER={ner[k][i] if ner else 'O'}"] + ([] if space else ["SpaceAfter=No"])
                f.write("\t".join([str(i + 1), word, "_", "_", tag, "_", str(head), rel, "_",
                                   "|".join(misc)]) + "\n")
            f.write("\n")


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    src, out = sys.argv[1], Path(sys.argv[2])
    times = {}
    t_all = t = time.perf_counter()

    with open(src, encoding="utf-8") as f:
        texts = [l.removeprefix("# text = ").rstrip("\n") for l in f]
    texts = [t_ for t_ in texts if t_.strip()]
    tokenizer = spacy.blank("en").tokenizer
    paragraphs = [[(tok.text, bool(tok.whitespace_)) for tok in doc if not tok.is_space]
                  for doc in tokenizer.pipe(texts)]
    paragraphs = [p for p in paragraphs if p]
    times["tokens"] = time.perf_counter() - t

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        # Sentences: one sequence per paragraph; the first token always starts one.
        t = time.perf_counter()
        write_conllu(tmp / "par.conllu", paragraphs, newpar=[True] * len(paragraphs))
        run(REPO / "conllu-to-vw.py", "--label", "sent", tmp / "par.conllu", tmp / "sent.vw")
        vw(MODELS["sent"], tmp / "sent.vw", tmp / "sent.pred")
        start = next(k for k, v in read_tags(MODELS["sent"] / "train.tags").items() if v == "B")
        sentences, newpar = [], []
        for par, pred in zip(paragraphs, read_predictions(tmp / "sent.pred")):
            for i, tok in enumerate(par):
                if i == 0 or pred[i] == start:
                    sentences.append([])
                    newpar.append(i == 0)
                sentences[-1].append(tok)
        times["sentences"] = time.perf_counter() - t

        # Tags. The converter needs a known tag in the XPOS column.
        t = time.perf_counter()
        pos_tags = read_tags(MODELS["pos"] / "train.tags")
        dummy = [[pos_tags[1]] * len(s) for s in sentences]
        write_conllu(tmp / "sents.conllu", sentences, xpos=dummy)
        run(REPO / "conllu-to-vw.py", tmp / "sents.conllu", tmp / "pos.vw",
            "--tags", MODELS["pos"] / "train.tags")
        vw(MODELS["pos"], tmp / "pos.vw", tmp / "pos.pred")
        xpos = [[pos_tags[y] for y in p] for p in read_predictions(tmp / "pos.pred")]
        times["tags"] = time.perf_counter() - t

        # Entities.
        t = time.perf_counter()
        run(REPO / "conllu-to-vw.py", "--label", "ner", "--context", "2", tmp / "sents.conllu", tmp / "ner.vw")
        vw(MODELS["ner"], tmp / "ner.vw", tmp / "ner.pred")
        ner_tags = read_tags(MODELS["ner"] / "train.tags")
        ner = [[ner_tags.get(y, "O") for y in p] for p in read_predictions(tmp / "ner.pred")]
        times["entities"] = time.perf_counter() - t

        # Parses, from the predicted tags.
        t = time.perf_counter()
        write_conllu(tmp / "tagged.conllu", sentences, xpos=xpos)
        run(REPO / "conllu-to-dep.py", tmp / "tagged.conllu", tmp / "dep.vw",
            "--tags", MODELS["dep"] / "train.tags")
        vw(MODELS["dep"], tmp / "dep.vw", tmp / "dep.pred")
        rels = read_tags(MODELS["dep"] / "train.tags")
        rel_to_id = {v: k for k, v in rels.items()}
        heads = []
        for sent, pred in zip(sentences, read_dep_predictions(tmp / "dep.pred")):
            pred, _ = attach_unattached(pred, [w for w, _ in sent], rel_to_id)
            heads.append([(h, "root" if h == 0 else rels[r]) for h, r in pred])
        times["parses"] = time.perf_counter() - t

    write_conllu(out, sentences, xpos=xpos, heads=heads, ner=ner, newpar=newpar)
    total = time.perf_counter() - t_all
    n_tok = sum(len(p) for p in paragraphs)
    print(f"{out}: {len(paragraphs)} paragraphs, {len(sentences)} sentences, {n_tok} tokens", file=sys.stderr)
    for step, s in times.items():
        print(f"  {step:10} {s:7.2f} s", file=sys.stderr)
    print(f"  {'total':10} {total:7.2f} s  ({n_tok / total:,.0f} tokens/s)", file=sys.stderr)


if __name__ == "__main__":
    main()
