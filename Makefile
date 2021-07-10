
data_prefix := data/ud-treebanks-v2.7/UD_English-EWT/en_ewt-ud

download:
	cd data && curl --remote-name-all https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-3424{/ud-treebanks-v2.7.tgz,/ud-documentation-v2.7.tgz,/ud-tools-v2.7.tgz}

prepare:
	mkdir -p data/work
	python conllu-to-vw.py $(data_prefix)-train.conllu data/work/train.conllu
	python conllu-to-vw.py $(data_prefix)-dev.conllu data/work/dev.conllu
	python conllu-to-vw.py $(data_prefix)-test.conllu data/work/test.conllu

train:
	time ./run.sh
