# DocPrompting: Generating Code by Retrieving the Docs

DocPrompting has two main components (1) a retriever and (2) a generator
## Retrieval
### Dense retrieval 
(`CoNaLa` as an example)

### Sparse retrieval 
(`tldr` as an example)

There are two stages in the retrieval procedure in `tldr`.
The first stage retrieves the bash command and the second stage retrieves the potentially relevant paragraphs that describe the usage of the arguments
1. build index with Elasticsearch
```bash
python retriever/bm25/main.py \
  --retrieval_stage 0
```
2. first stage retrieval
```bash
python retriever/bm25/main.py \
  --retrieval_stage 1 \
  --split {cmd_train, cmd_dev, cmd_test}
```
3. second stage retrieval
```bash
python retriever/bm25/main.py \
  --retrieval_stage 2 \
  --split {cmd_train, cmd_dev, cmd_test}
```

---
## Generation
### FID generation

---
## Data
The `data` folder contains the two benchmarks we curated or re-splitted.
* tldr
* CoNaLa

On each dataset, we provide 
1. Natural language intent (entry `nl`)
2. Oracle code (entry `cmd`) 
  * Bash for tldr
  * Python for CoNaLa
3. Oracle docs (entry `oracle_man`) 
  * In the data files, we only provide the manual ids, their contents could be found in the `{tldr, conala}_docs.json` of `docs.zip`.

## Resources 
* [tldr](https://github.com/tldr-pages/tldr) github repo
* [conla](https://conala-corpus.github.io)

## Citation
@article{zhou22docprompting,
    title = {# DocPrompting: Generating Code by Retrieving the Docs},
    author = {Shuyan Zhou and Uri Alon and Frank F. Xu and and Zhiruo Wang and Zhengbao Jiang and Graham Neubig},
    year = {2022}
}
