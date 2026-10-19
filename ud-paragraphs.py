#!/usr/bin/env python
"""Write the raw text of CoNLL-U files as paragraphs, one per line.

Usage:
    uv run python ud-paragraphs.py OUTPUT.txt INPUT.conllu [INPUT.conllu ...]

The `# text` lines of each paragraph are joined with spaces. Paragraphs are
marked by `# newpar` (EWT has them); a file without any uses `# newdoc`
instead (GUM, LinES, PUD), and a file with neither (ParTUT) gets one
paragraph per sentence. The output is input for spacy-label.py and
pipeline.py, which then do their own sentence splitting.
"""

import sys


def paragraphs(path):
    with open(path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    marker = next((m for m in ("# newpar", "# newdoc") if any(l.startswith(m) for l in lines)), None)
    out, cur = [], []
    for line in lines:
        if marker and line.startswith(marker) and cur:
            out.append(cur)
            cur = []
        elif line.startswith("# text = "):
            cur.append(line[len("# text = "):].strip())
            if not marker:
                out.append(cur)
                cur = []
    if cur:
        out.append(cur)
    return [" ".join(p) for p in out]


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    n = 0
    with open(sys.argv[1], "w", encoding="utf-8") as out:
        for path in sys.argv[2:]:
            for p in paragraphs(path):
                out.write(p + "\n")
                n += 1
    print(f"{sys.argv[1]}: {n} paragraphs", file=sys.stderr)


if __name__ == "__main__":
    main()
