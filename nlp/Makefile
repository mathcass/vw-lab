# VW POS tagger (gold UD tags), and the spaCy copies: tagger, sentence
# splitter, entity tagger, parser, and the pipeline that chains them. See
# README.md. Run from this folder (`make -C nlp TARGET` from the root).
# Data lives in the repo's shared data/; downloads are in the root Makefile.

DATA := ../data
ud := $(DATA)/ud-treebanks-v2.7
data_prefix := $(ud)/UD_English-EWT/en_ewt-ud
work := $(DATA)/work
VW := uv run python -m vowpalwabbit

# NER: spaCy's entities on UD text + 4M words of C4 web text, distilled into VW.
spacy := $(DATA)/spacy
c4 := $(DATA)/c4
wiki := $(DATA)/wikitext
ner_work := $(DATA)/ner-work
dep_work := $(DATA)/dep-work
spacy_pos_work := $(DATA)/spacy-pos-work
sent_work := $(DATA)/sent-work
pipeline_work := $(DATA)/pipeline
SPACY := uv run --group spacy python
# "All text": EWT train plus the other English treebanks with full text.
spacy_train_src := $(data_prefix)-train.conllu \
	$(foreach t,GUM LinES ParTUT PUD,$(sort $(wildcard $(ud)/UD_English-$(t)/*.conllu)))

# UD v2.7 (then untar ud-treebanks-v2.7.tgz in data/).
download:
	$(MAKE) -C .. download

# Shared downloads (pinned and checksummed), built by the root Makefile.
$(c4)/c4-train-00000.json.gz $(wiki)/wiki-10m.txt:
	$(MAKE) -C .. $(patsubst ../%,%,$@)

# Dev and test reuse train's tag map so label ids line up with the model.
# The VW cache is removed so training doesn't reuse stale data.
prepare:
	mkdir -p $(work)
	rm -f $(work)/train.conllu.cache $(work)/dev.tags $(work)/test.tags
	uv run python conllu-to-vw.py $(data_prefix)-train.conllu $(work)/train.conllu
	uv run python conllu-to-vw.py $(data_prefix)-dev.conllu $(work)/dev.conllu --tags $(work)/train.tags
	uv run python conllu-to-vw.py $(data_prefix)-test.conllu $(work)/test.conllu --tags $(work)/train.tags

train:
	time ./run.sh

eval:
	for split in dev test; do \
	  echo "== $$split =="; \
	  $(VW) -t -i $(work)/model.vw -d $(work)/$$split.conllu -p $(work)/$$split.pred --quiet && \
	  uv run python evaluate.py $(work)/$$split.conllu $(work)/$$split.pred \
	    --train $(work)/train.conllu --tags $(work)/train.tags || exit 1; \
	done

# spaCy labels for the treebank text (~3 min).
$(spacy)/train.conllu:
	mkdir -p $(spacy)
	$(SPACY) spacy-label.py $@ $(spacy_train_src)

$(spacy)/%.conllu:
	mkdir -p $(spacy)
	$(SPACY) spacy-label.py $@ $(data_prefix)-$*.conllu

# Whole paragraphs for sentence splitting: UD text joined into paragraphs,
# then labeled by spaCy, which splits the sentences itself.
$(spacy)/train-par.txt:
	mkdir -p $(spacy)
	uv run python ud-paragraphs.py $@ $(spacy_train_src)

$(spacy)/%-par.txt:
	mkdir -p $(spacy)
	uv run python ud-paragraphs.py $@ $(data_prefix)-$*.conllu

# A static pattern, so these don't fall to the $(spacy)/%.conllu rule above.
$(foreach s,train dev test,$(spacy)/$(s)-par.conllu): %.conllu: %.txt
	$(SPACY) spacy-label.py $@ $<

$(spacy)/train-ewt.conllu:
	mkdir -p $(spacy)
	$(SPACY) spacy-label.py $@ $(data_prefix)-train.conllu

$(c4)/c4-4m.txt: $(c4)/c4-train-00000.json.gz
	uv run python c4-to-text.py $< $@ 4000000

# spaCy labels for the 4M-word sample (~11 min).
$(c4)/c4-4m.conllu: $(c4)/c4-4m.txt
	$(SPACY) spacy-label.py $@ $<

# WikiText: the earlier NER recipe (e5a in the README); not used by ner-prepare.
# spaCy labels for the 10M-word sample (~17 min; the sample comes from the root Makefile).
$(wiki)/wiki-10m.conllu: $(wiki)/wiki-10m.txt
	$(SPACY) spacy-label.py $@ $<

# 5.1M tokens, shuffled together (~2 min).
ner-prepare: $(spacy)/train.conllu $(spacy)/dev.conllu $(spacy)/test.conllu $(c4)/c4-4m.conllu
	mkdir -p $(ner_work)
	rm -f $(ner_work)/train.conllu.cache
	uv run python conllu-to-vw.py --label ner --context 2 --shuffle 0 \
	  $(spacy)/train.conllu --add $(c4)/c4-4m.conllu $(ner_work)/train.conllu
	for split in dev test; do \
	  uv run python conllu-to-vw.py --label ner --context 2 $(spacy)/$$split.conllu $(ner_work)/$$split.conllu || exit 1; \
	done

ner-train:
	TASK=ner time ./run.sh

ner-eval:
	for split in dev test; do \
	  echo "== $$split =="; \
	  $(VW) -t -i $(ner_work)/model.vw -d $(ner_work)/$$split.conllu -p $(ner_work)/$$split.pred --quiet && \
	  uv run python evaluate.py $(ner_work)/$$split.conllu $(ner_work)/$$split.pred \
	    --tags $(ner_work)/train.tags || exit 1; \
	done

# Tagger trained on spaCy's tags for EWT train text (the tagger for pipeline.py).
spacy-pos-prepare: $(spacy)/train-ewt.conllu $(spacy)/dev.conllu $(spacy)/test.conllu
	mkdir -p $(spacy_pos_work)
	rm -f $(spacy_pos_work)/train.conllu.cache
	uv run python conllu-to-vw.py $(spacy)/train-ewt.conllu $(spacy_pos_work)/train.conllu
	for split in dev test; do \
	  uv run python conllu-to-vw.py $(spacy)/$$split.conllu $(spacy_pos_work)/$$split.conllu \
	    --tags $(spacy_pos_work)/train.tags || exit 1; \
	done

spacy-pos-train:
	TASK=spacy-pos time ./run.sh

spacy-pos-eval:
	for split in dev test; do \
	  echo "== $$split =="; \
	  $(VW) -t -i $(spacy_pos_work)/model.vw -d $(spacy_pos_work)/$$split.conllu -p $(spacy_pos_work)/$$split.pred --quiet && \
	  uv run python evaluate.py $(spacy_pos_work)/$$split.conllu $(spacy_pos_work)/$$split.pred \
	    --train $(spacy_pos_work)/train.conllu --tags $(spacy_pos_work)/train.tags || exit 1; \
	done

sent-prepare: $(spacy)/train-par.conllu $(spacy)/dev-par.conllu $(spacy)/test-par.conllu
	mkdir -p $(sent_work)
	rm -f $(sent_work)/train.conllu.cache
	uv run python conllu-to-vw.py --label sent --shuffle 0 $(spacy)/train-par.conllu $(sent_work)/train.conllu
	for split in dev test; do \
	  uv run python conllu-to-vw.py --label sent $(spacy)/$$split-par.conllu $(sent_work)/$$split.conllu || exit 1; \
	done

sent-train:
	TASK=sent time ./run.sh

sent-eval:
	for split in dev test; do \
	  echo "== $$split =="; \
	  $(VW) -t -i $(sent_work)/model.vw -d $(sent_work)/$$split.conllu -p $(sent_work)/$$split.pred --quiet && \
	  uv run python evaluate.py $(sent_work)/$$split.conllu $(sent_work)/$$split.pred \
	    --tags $(sent_work)/train.tags || exit 1; \
	done

# Dependency parser trained on spaCy's parses of all UD text, with spaCy's
# tags as input (dep-eval scores it the same way; pipeline.py gives it the
# VW tagger's tags instead).
dep-prepare: $(spacy)/train.conllu $(spacy)/dev.conllu $(spacy)/test.conllu
	mkdir -p $(dep_work)
	rm -f $(dep_work)/train.conllu.cache
	uv run python conllu-to-dep.py $(spacy)/train.conllu $(dep_work)/train.conllu
	for split in dev test; do \
	  uv run python conllu-to-dep.py $(spacy)/$$split.conllu $(dep_work)/$$split.conllu \
	    --tags $(dep_work)/train.tags || exit 1; \
	done

dep-train:
	TASK=dep time ./run.sh

dep-eval:
	for split in dev test; do \
	  echo "== $$split =="; \
	  $(VW) -t -i $(dep_work)/model.vw -d $(dep_work)/$$split.conllu -p $(dep_work)/$$split.pred --quiet && \
	  uv run python evaluate.py $(dep_work)/$$split.conllu $(dep_work)/$$split.pred \
	    --tags $(dep_work)/train.tags || exit 1; \
	done

# Raw EWT dev/test paragraphs through all four VW models (pipeline.py), scored
# against spaCy's full pipeline on the same text. Needs the spacy-pos, ner,
# sent and dep models.
pipeline-eval: $(spacy)/dev-par.conllu $(spacy)/test-par.conllu
	mkdir -p $(pipeline_work)
	for split in dev test; do \
	  echo "== $$split =="; \
	  $(SPACY) pipeline.py $(spacy)/$$split-par.txt $(pipeline_work)/$$split.conllu && \
	  uv run python compare.py $(spacy)/$$split-par.conllu $(pipeline_work)/$$split.conllu || exit 1; \
	done

.PHONY: download prepare train eval ner-prepare ner-train ner-eval \
	spacy-pos-prepare spacy-pos-train spacy-pos-eval sent-prepare sent-train sent-eval \
	dep-prepare dep-train dep-eval pipeline-eval
