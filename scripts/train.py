import sys

from vowpalwabbit import pyvw



with open(sys.argv[2]) as f:
    tags = f.readlines()

n_tags = len([t for t in tags if t.strip()])
train = sys.argv[1]

opts = [
    "-k", "-c", f"-d {train}", "--passes 100",
    "--search_task sequence",
    f"--search {n_tags}",
    "--search_alpha 1e-6",
    "--search_max_bias_ngram_length 2",
    "--search_max_quad_ngram_length 1",
    "--holdout_off"
]


model = pyvw.vw(" ".join(opts))

model.save(sys.argv[3])
