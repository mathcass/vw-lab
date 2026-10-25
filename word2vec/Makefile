# Word embeddings with Vowpal Wabbit, against gensim word2vec and PPMI+SVD.
# See README.md. Run from this folder (`make -C word2vec TARGET` from the root).
#
#   make setup benchmarks corpus   once: Python env, benchmark files, 10M-token corpus
#   make smoke                     ~2 min check that the whole pipeline works
#   make fast                      VW gd_mf, the practical recipe (~21 min)
#   make best                      VW lrq + PiSTOL, the best vectors (~2.4 h)
#   make baselines                 gensim word2vec + PPMI+SVD on the same tokens
#   make dev-optimizers            PiSTOL vs FTRL vs default on the 2M-token dev slice (~12 min)
#   make dev                       key runs on the dev slice (~1.3 h)
#   make big-corpus scale probes   memory/scale runs at 100M tokens (~3.5 h)
#   make table                     results of the runs above
#
# Downloads and work files go in the repo's shared data/ (not committed);
# results/results.tsv and results/memory.tsv are committed.

DATA := ../data
WORK := $(DATA)/word2vec-work
BIG := $(DATA)/word2vec-big
BENCH := $(DATA)/word-benchmarks
PY := uv run python

# VW flags for the two recipes
LRQ := --loss_function logistic --noconstant --pistol
GDMF := --loss_function logistic --noconstant --power_t 0 -l 0.02
DEV := --max-paras 22000

simlex_sha256 := 30f40eb18f41b08b65dfbc328f473e8b69ba444b1c9d053c1adde2324fc76315
ws353_sha256 := 4f12a5fe9650184662306f8ae72c7e6bb44c2627e0275e9f9c6eb067e3441a22
w2v_rev := 20c129af10659f7c50e86e3be406df663beff438
analogy_sha256 := 8c29b3332afc46f3fb8be04cb5297bf96f39aa7131272dff57869b4485b22a36

setup:
	uv sync

# --- evaluation data -------------------------------------------------------

benchmarks: $(BENCH)/SimLex-999.txt $(BENCH)/ws353.csv $(BENCH)/questions-words.txt

$(BENCH)/SimLex-999.txt:
	mkdir -p $(BENCH)
	curl -fL -o $(BENCH)/SimLex-999.zip https://fh295.github.io/SimLex-999.zip
	echo "$(simlex_sha256)  $(BENCH)/SimLex-999.zip" | sha256sum -c -
	unzip -p $(BENCH)/SimLex-999.zip SimLex-999/SimLex-999.txt > $@

# The official zip has CRLF line endings; the results used the file without them.
$(BENCH)/ws353.csv:
	mkdir -p $(BENCH)
	curl -fL -o $(BENCH)/wordsim353.zip https://gabrilovich.com/resources/data/wordsim353/wordsim353.zip
	echo "$(ws353_sha256)  $(BENCH)/wordsim353.zip" | sha256sum -c -
	unzip -p $(BENCH)/wordsim353.zip combined.csv | tr -d '\r' > $@

$(BENCH)/questions-words.txt:
	mkdir -p $(BENCH)
	curl -fL -o $@.part https://raw.githubusercontent.com/tmikolov/word2vec/$(w2v_rev)/questions-words.txt
	echo "$(analogy_sha256)  $@.part" | sha256sum -c -
	mv $@.part $@

# --- corpora ---------------------------------------------------------------

# The text comes from the root Makefile's pinned WikiText and C4 downloads.
$(DATA)/wikitext/wiki-10m.txt $(DATA)/wikitext/train-00000.parquet $(DATA)/c4/c4-train-00000.json.gz:
	$(MAKE) -C .. $(patsubst ../%,%,$@)

# 10M tokens of WikiText, one tokenizer for every method, 20k-word vocabulary.
corpus: $(WORK)/vocab.tsv
$(WORK)/vocab.tsv: $(DATA)/wikitext/wiki-10m.txt
	mkdir -p $(WORK)
	$(PY) corpus.py $< $(WORK) --vocab 20000

# 100M tokens (WikiText-103 shard, then C4), vocabulary = words seen >= 5 times.
big-corpus: $(BIG)/vocab.tsv
$(BIG)/vocab.tsv: $(DATA)/wikitext/train-00000.parquet $(DATA)/c4/c4-train-00000.json.gz
	mkdir -p $(BIG)
	$(PY) bigcorpus.py $(BIG)

# --- 10M-token runs (the main results table) -------------------------------
# Run names match the rows already in results/results.tsv, so a rerun can be
# compared with the recorded one (`make table` shows the latest of each).

best: corpus benchmarks
	./train.sh full_cbow_e5_pistol_k100 100 "gen.py cbow --epochs 5" "$(LRQ)" "lrq rank 100, 5 epochs"

fast: corpus benchmarks
	ARCH=gdmf ./train.sh full_gdmf_cbow_e5_k100 100 "gen.py cbow --epochs 5" "$(GDMF)" "gd_mf rank 100, 5 epochs"

fast50: corpus benchmarks
	ARCH=gdmf ./train.sh full_gdmf_cbow_e5_l02 50 "gen.py cbow --epochs 5" "$(GDMF)" "gd_mf rank 50, 5 epochs"

best50: corpus benchmarks
	./train.sh full_cbow_e5_pistol 50 "gen.py cbow --epochs 5" "$(LRQ)" "lrq rank 50, 5 epochs"

baselines: corpus benchmarks
	./baseline.sh gensim_sgns w2v --dim 100
	./baseline.sh full_w2v50 w2v --dim 50
	./baseline.sh full_w2vcbow50 w2v --dim 50 --sg 0
	./baseline.sh ppmi_svd ppmi --dim 100
	./baseline.sh full_ppmi50 ppmi --dim 50

# --- 2M-token dev slice (where the settings were chosen) --------------------

smoke: corpus benchmarks
	ARCH=gdmf ./train.sh smoke 50 "gen.py cbow --epochs 1 $(DEV)" "$(GDMF)" "smoke test"
	./baseline.sh smoke_w2v w2v --dim 50 --epochs 1 $(DEV)

# The optimizer comparison (1 epoch each, ~4 min each): PiSTOL > FTRL > default.
dev-optimizers: corpus benchmarks
	./train.sh dev_cbow_e1 50 "gen.py cbow --epochs 1 $(DEV)" "--loss_function logistic --noconstant" "default update"
	./train.sh dev_cbow_ftrl 50 "gen.py cbow --epochs 1 $(DEV)" "--loss_function logistic --noconstant --ftrl" "ftrl"
	./train.sh dev_cbow_pistol 50 "gen.py cbow --epochs 1 $(DEV)" "$(LRQ)" "pistol"

dev: corpus benchmarks dev-optimizers
	./baseline.sh dev_w2v w2v --dim 50 $(DEV)
	./baseline.sh dev_ppmi ppmi --dim 50 $(DEV)
	./train.sh dev_cbow_e5_pistol 50 "gen.py cbow --epochs 5 $(DEV)" "$(LRQ)" "pistol 5 epochs"
	./train.sh dev_sgns_e5_pistol 50 "gen.py sgns --epochs 5 $(DEV)" "$(LRQ)" "skip-gram pairs, pistol 5 epochs"
	ARCH=gdmf ./train.sh dev_gdmf_cbow_e5_l0.05 50 "gen.py cbow --epochs 5 $(DEV)" \
	  "--loss_function logistic --noconstant -l 0.05 --power_t 0" "gd_mf lr 0.05"
	ARCH=gdmf ./train.sh dev_gdmf_cbow_e5_l0.02 50 "gen.py cbow --epochs 5 $(DEV)" "$(GDMF)" "gd_mf lr 0.02"

# --- 100M tokens: memory and scale (one epoch each, 3 GB cap, swap off) -----

scale: corpus big-corpus benchmarks
	CAP=3G ./baseline.sh 10m_gensim_sg w2v_stream --dim 100 --epochs 1
	CAP=3G ./baseline.sh 10m_gensim_cbow w2v_stream --dim 100 --epochs 1 --sg 0
	CAP=3G ./baseline.sh 10m_ppmi ppmi --dim 100
	CAP=3G ARCH=gdmf ./train.sh 10m_vw_gdmf 100 "gen.py cbow --epochs 1" "$(GDMF)" "1 epoch"
	CAP=3G W2V_WORK=$(BIG) ./baseline.sh 100m_gensim_sg w2v_stream --dim 100 --epochs 1
	CAP=3G W2V_WORK=$(BIG) ./baseline.sh 100m_gensim_cbow w2v_stream --dim 100 --epochs 1 --sg 0
	-CAP=3G W2V_WORK=$(BIG) ./baseline.sh 100m_ppmi ppmi --dim 100   # needs > 3 GB: killed
	CAP=3G W2V_WORK=$(BIG) ARCH=gdmf ./train.sh 100m_vw_gdmf_b18 100 \
	  "gen.py cbow --epochs 1 --corpus $(BIG)/corpus.ids" "$(GDMF)" "ids, -b 18"
	CAP=3G W2V_WORK=$(BIG) ARCH=gdmf BITS=17 HASHED=1 ./train.sh 100m_vw_gdmf_hash17 100 \
	  "gen.py cbow --epochs 1 --corpus $(BIG)/corpus.ids --words $(BIG)/vocab.tsv" "$(GDMF)" "hashed words, -b 17"
	CAP=3G W2V_WORK=$(BIG) ARCH=gdmf BITS=17 HASHED=1 VOCAB_LABEL="730k (all)" ./train.sh 100m_vw_gdmf_online17 100 \
	  "gen_online.py $(BIG)/corpus.txt" "$(GDMF)" "one pass, no vocabulary, -b 17"

# Resident memory once every word's weights exist (VW only; seconds each).
probes: corpus big-corpus
	./touch_probe.sh probe_10m_lrq_b25 $(WORK)/vocab.tsv "--lrq wc100 -b 25 $(LRQ)"
	./touch_probe.sh probe_10m_gdmf_b18 $(WORK)/vocab.tsv "-q wc --rank 100 -b 18 $(GDMF)"
	./touch_probe.sh probe_100m_lrq_b26 $(BIG)/vocab.tsv "--lrq wc100 -b 26 $(LRQ)" ids cb
	./touch_probe.sh probe_100m_gdmf_b18 $(BIG)/vocab.tsv "-q wc --rank 100 -b 18 $(GDMF)"
	./touch_probe.sh probe_100m_gdmf_hash17 $(BIG)/vocab.tsv "-q wc --rank 100 -b 17 $(GDMF)" words

# --- results ---------------------------------------------------------------

table:
	@$(PY) summarize.py full_cbow_e5_pistol_k100.wc full_gdmf_cbow_e5_k100.wc gensim_sgns ppmi_svd \
	  full_cbow_e5_pistol.wc full_gdmf_cbow_e5_l02.wc full_w2v50 full_w2vcbow50 full_ppmi50 \
	  dev_w2v dev_ppmi dev_cbow_e1.wc dev_cbow_ftrl.wc dev_cbow_pistol.wc dev_cbow_e5_pistol.wc \
	  dev_sgns_e5_pistol.wc dev_gdmf_cbow_e5_l0.05.wc dev_gdmf_cbow_e5_l0.02.wc \
	  10m_gensim_sg 10m_vw_gdmf.wc 100m_gensim_sg 100m_gensim_cbow 100m_vw_gdmf_b18.wc \
	  100m_vw_gdmf_hash17.wc 100m_vw_gdmf_online17.wc

.PHONY: setup benchmarks corpus big-corpus best best50 fast fast50 baselines smoke dev-optimizers dev \
	scale probes table
