# DocPrompting: Generating Code by Retrieving the Docs

DocPrompting has two main components (1) a retriever and (2) a generator which can be instantiated with different implementations.
In the current version, we provide the *best* model on each setting.
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
The code is based on [FiD](https://github.com/facebookresearch/FiD)
A training or evaluation file should be converted to the format compatible with FiD. 
An example is [here](./data/conala/example_fid_data.json)
> **Warning**: FiD has a strong dependency on the version of `transformers` (3.0.2).
> Unable to match the version might result in inreproducible results.
1. Training. If you want to use our [trained model] on Python CoNaLa(./models/generator/), directly check the second step of running generation.
```bash
python generator/fid/train_reader.py \
    --seed 1996 \
    --train_data data/conala/fid.cmd_train.codet5.t10.json \
    --eval_data data/conala/fid.cmd_dev.codet5.t10.json \
    --model_name models/generator/codet5-base \ # initialize with the codet5-base model \
    --per_gpu_batch_size 4 \
    --n_context 10 \
    --name conala.fid.codet5.top10 \
    --checkpoint_dir models/generator/ \
    --eval_freq 500 \
    --accumulation_steps 2 \
    --main_port 30843 \
    --total_steps 20000 \
    --warmup_steps 2000
```
2. Run generation
```bash
python generator/fid/test_reader_simple.py \
    --model_path models/generator/conala.fid.codet5.top10/checkpoint/best_dev \
    --tokenizer_name models/generator/codet5-base \
    --eval_data data/conala/fid.cmd_test.codet5.t10.json \
    --per_gpu_batch_size 8 \
    --n_context 10 \
    --name conala.fid.codet5.top10 \
    --checkpoint_dir models/generator  \
    --result_tag test_same \
    --main_port 81692
```
The results will be saved to `models/generator/{name}/test_results_test_same.json`
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
* [CoNaLa](https://conala-corpus.github.io)

## Citation
```
@article{zhou22docprompting,
    title = {# DocPrompting: Generating Code by Retrieving the Docs},
    author = {Shuyan Zhou and Uri Alon and Frank F. Xu and and Zhiruo Wang and Zhengbao Jiang and Graham Neubig},
    year = {2022}
}
```
