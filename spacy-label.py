#!/usr/bin/env python
"""Label raw sentence text with spaCy and write the result as CoNLL-U.

Usage:
    uv run --group spacy python spacy-label.py OUTPUT INPUT.conllu [INPUT.conllu ...]

Reads the `# text = ...` line of each sentence in the inputs (the gold
annotation is ignored) and runs spaCy's small English pipeline on it. An
input ending in .txt is read as one paragraph per line instead (a leading
`# text = ` is dropped and blank lines skipped; ud-paragraphs.py,
c4-to-text.py and wikitext-to-text.py write these), and each paragraph's
first sentence gets a `# newpar` comment, so spaCy's sentence splits can be
learned. Each sentence spaCy finds becomes one CoNLL-U sentence, with spaCy's
tokens and:

  LEMMA  token.lemma_      UPOS    token.pos_
  XPOS   token.tag_        FEATS   token.morph
  HEAD   parser head       DEPREL  token.dep_
  MISC   NER=B-ORG / I-ORG / O, and SpaceAfter=No where it applies

This is training data for distilling spaCy into VW: the labels are spaCy's
predictions, not gold. Whitespace-only tokens are dropped; a token whose head
was one is attached to that token's head instead.
"""

import sys

import spacy


def texts(paths):
    """(text, is_paragraph) for each text to label."""
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if path.endswith(".txt"):
                    line = line.removeprefix("# text = ").rstrip("\n")
                    if line.strip():
                        yield line, True
                elif line.startswith("# text = "):
                    yield line[len("# text = "):].rstrip("\n"), False


def write_sentence(out, sent, n_written, newpar=False):
    keep = [t for t in sent if not t.is_space]
    ids = {t.i: k + 1 for k, t in enumerate(keep)}

    def head_id(t):
        while t.head.i != t.i and t.head.i not in ids:
            t = t.head  # skip over dropped whitespace tokens
        return 0 if t.head.i == t.i else ids[t.head.i]

    if newpar:
        out.write("# newpar\n")
    out.write(f"# sent_id = spacy-{n_written}\n# text = {sent.text.strip()}\n")
    for t in keep:
        ner = "O" if t.ent_iob_ == "O" else f"{t.ent_iob_}-{t.ent_type_}"
        misc = [f"NER={ner}"] + ([] if t.whitespace_ else ["SpaceAfter=No"])
        dep = "root" if head_id(t) == 0 else (t.dep_ or "dep")
        feats = str(t.morph) or "_"
        out.write("\t".join([str(ids[t.i]), t.text, t.lemma_ or "_", t.pos_ or "_", t.tag_ or "_",
                             feats, str(head_id(t)), dep, "_", "|".join(misc)]) + "\n")
    out.write("\n")


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    output, inputs = sys.argv[1], sys.argv[2:]
    nlp = spacy.load("en_core_web_sm")
    n = 0
    with open(output, "w", encoding="utf-8") as out:
        for doc, is_par in nlp.pipe(texts(inputs), batch_size=256, as_tuples=True):
            first = is_par
            for sent in doc.sents:
                if any(not t.is_space for t in sent):
                    n += 1
                    write_sentence(out, sent, n, newpar=first)
                    first = False
    print(f"{output}: {n} sentences", file=sys.stderr)


if __name__ == "__main__":
    main()
