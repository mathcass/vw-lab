# Shared downloads for nlp/ and word2vec/. Everything lands in data/ (not
# committed). Each project's own Makefile calls these targets when it needs
# a file, so they rarely need running by hand.
#
#   make download                      UD v2.7 (then untar ud-treebanks-v2.7.tgz)
#   make data/c4/c4-train-00000.json.gz     first C4 en shard (319 MB)
#   make data/wikitext/wiki-10m.txt         10M-word WikiText-103 sample

ud := data/ud-treebanks-v2.7
c4 := data/c4
wiki := data/wikitext
c4_rev := 1588ec454efa1a09f29cd18ddd04fe05fc8653a2
c4_sha256 := 8ef8d75b0e045dec4aa5123a671b4564466b0707086a7ed1ba8721626dfffbc9
wiki_rev := b08601e04326c79dfdd32d625aee71d232d685c3
wiki_sha256 := 74da360f23826045b3e6ac6375411fdb15f003030aa74f2596ed08b857cb9212

download:
	mkdir -p data
	cd data && curl --remote-name-all https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-3424{/ud-treebanks-v2.7.tgz,/ud-documentation-v2.7.tgz,/ud-tools-v2.7.tgz}

# The first C4 (en) training shard (319 MB), pinned to one dataset revision.
$(c4)/c4-train-00000.json.gz:
	mkdir -p $(c4)
	curl -fL -o $@.part https://huggingface.co/datasets/allenai/c4/resolve/$(c4_rev)/en/c4-train.00000-of-01024.json.gz
	echo "$(c4_sha256)  $@.part" | sha256sum -c -
	mv $@.part $@

# The first WikiText-103 training shard (157 MB), pinned to one dataset revision.
$(wiki)/train-00000.parquet:
	mkdir -p $(wiki)
	curl -fL -o $@.part https://huggingface.co/datasets/Salesforce/wikitext/resolve/$(wiki_rev)/wikitext-103-raw-v1/train-00000-of-00002.parquet
	echo "$(wiki_sha256)  $@.part" | sha256sum -c -
	mv $@.part $@

# Seed-0 sample of 10M words of paragraphs, as `# text` lines.
$(wiki)/wiki-10m.txt: $(wiki)/train-00000.parquet
	uv run python wikitext-to-text.py $< $@ 10000000 0

.PHONY: download
