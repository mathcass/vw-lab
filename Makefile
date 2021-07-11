
vw := pipenv run python -m vw
python := pipenv run python

# Downloading UD data isn't working right now

data/ud-treebanks-v2.7.tgz: data/work
	cd data && curl --remote-name-all https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-3424/ud-treebanks-v2.7.tgz

data/ud-treebanks-v2.7:
	cd data && tar zxvf ud-treebanks-v2.7.tgz

data_prefix := data/ud-treebanks-v2.7/UD_English-EWT/en_ewt-ud
data/work/train.conllu: data/ud-treebanks-v2.7
	python conllu-to-vw.py $(data_prefix)-train.conllu data/work/train.conllu
	python conllu-to-vw.py $(data_prefix)-dev.conllu data/work/dev.conllu
	python conllu-to-vw.py $(data_prefix)-test.conllu data/work/test.conllu

train: data/work/train.conllu
	$(MAKE) run-vw DATA=data/work/train.conllu TAGS=data/work/train.tags MODEL=data/work/train.model
	# time ./run.sh

data/work:
	mkdir -p data/work

scripts/parse_as_conllu.py:
	mkdir -p scripts/
	cd scripts && wget "https://raw.githubusercontent.com/rgalhama/spaCy2CoNLLU/master/parse_as_conllu.py"
	patch scripts/parse_as_conllu.py scripts/parse_as_conllu.patch

data/work/pride_sample.txt: data/work
	wget -O data/work/pride.txt "https://www.gutenberg.org/files/1342/1342-0.txt"
	head -1000 data/work/pride.txt | tail -500 | tr -s '\r\n' ' ' > data/work/pride_sample.txt

data/work/pride_sample.conllu: scripts/parse_as_conllu.py data/work/pride_sample.txt
	$(python) scripts/parse_as_conllu.py \
		--input_file data/work/pride_sample.txt \
		--model en_core_web_sm \
		> data/work/pride_sample.conllu

data/work/pride_sample.vw: data/work/pride_sample.conllu
	$(python) conllu-to-vw.py data/work/pride_sample.conllu data/work/pride_sample.vw

run-vw:  # Runs vw with $(DATA) and $(TAGS) saving $(MODEL)
	# Use TAGS as the search count
	$(vw) -k -c -d "$(DATA)" --passes 100 \
		--search_task sequence --search $(shell wc -l $(TAGS) | awk '{print $$1}') \
		--search_alpha 1e-6 \
		--search_max_bias_ngram_length 2 --search_max_quad_ngram_length 1 \
		--holdout_off \
		-f "$(MODEL)"

spacy-baseline: data/work/pride_sample.vw
	# $(python) scripts/train.py data/work/pride_sample.vw data/work/pride_sample.tags data/work/pride_sample.model
	$(MAKE) run-vw DATA=data/work/pride_sample.vw TAGS=data/work/pride_sample.tags MODEL=data/work/pride_sample.model
	echo "Done"

clean:
	rm -rf data/work/
