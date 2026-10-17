data_prefix := data/ud-treebanks-v2.7/UD_English-EWT/en_ewt-ud
work := data/work
VW := uv run python -m vowpalwabbit

download:
	cd data && curl --remote-name-all https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-3424{/ud-treebanks-v2.7.tgz,/ud-documentation-v2.7.tgz,/ud-tools-v2.7.tgz}

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

.PHONY: download prepare train eval
