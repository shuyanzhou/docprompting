# ANLP Assignment 3: DocPrompting: Generating Code by Retrieving the Docs

## Team ID: 30
Team Members:   
Rucha Kulkarni (rmkulkar)   
Harini Subramanyan (harinisu)   
Asmita Hajra (ahajra)   

--- 

This repository contains the code for Assignment 3 of the course Advanced Natural Language Processing. 

In this repository, we replicate results from the research paper titled DocPrompting -

Shuyan Zhou, Uri Alon, Frank F. Xu, Zhiruo Wang, Zhengbao Jiang, Graham Neubig, ["DocPrompting: Generating Code by Retrieving the Docs"](https://arxiv.org/pdf/2207.05987.pdf),
ICLR'2023 (**Spotlight**) 

---


Since publicly available source-code libraries are continuously growing and changing, this paper 
introduces DocPrompting: a natural-language-to-code generation approach that explicitly leverages documentation by
1. retrieving the relevant documentation pieces given an NL intent, 
and
2. generating code based on the NL intent and the retrieved documentation. 


---

## Experiment setup

1. To replicate the experiments outlined in the paper, our initial step involved identifying the specific areas to focus on for A4. 
2. We decided that, as a component of A4, our emphasis would be directed towards Python programs, and consequently, we opted to utilize the CoNaLa dataset as our benchmark.
3. Additionally, since our objective in A4 is to evaluate the performance of state-of-the-art models, as elaborated in Section 8, we decided to replicate the results of CodeT5, as the other models from the paper are either out-of-date or do not promise better results.
   
---

## Process to replicate results

>The following instructions are for reproducing the results in our report.

## Preparation

Download data for `CoNaLa` and `tldr` from [link](https://drive.google.com/file/d/1CzNlo8-e4XqrgAME5zHEWEKIQMPga0xl/view?usp=sharing)
```bash
# unzip
unzip docprompting_data.zip
# move to the data folder
mv docprompting_data/* data
```

Download trained generator weights from [link](https://drive.google.com/file/d/1NmPMxY1EOWkjM7S8VSKa13DKJmEZ3TqV/view?usp=sharing)
```bash
unzip docprompting_generator_models.zip
# move to the model folder
mv docprompting_generator_models/* models/generator

```
## Retrieval results (Table 2 and 3 of the report)

1. Run inference with the trained model on CoNaLa (Python) with and without normalizing the embeddings.
   
```bash
python retriever/simcse/run_inference.py \
  --model_name "neulab/docprompting-codet5-python-doc-retriever" \
  --source_file data/conala/conala_nl.txt \
  --target_file data/conala/python_manual_firstpara.tok.txt \
  --source_embed_save_file data/conala/.tmp/src_embedding \
  --target_embed_save_file data/conala/.tmp/tgt_embedding \
  --sim_func cls_distance.cosine \
  --num_layers 12 \
  --normalize_embed \
  --save_file data/conala/retrieval_results.json
```

```bash
python retriever/simcse/run_inference.py \
  --model_name "neulab/docprompting-codet5-python-doc-retriever" \
  --source_file data/conala/conala_nl.txt \
  --target_file data/conala/python_manual_firstpara.tok.txt \
  --source_embed_save_file data/conala/.tmp/src_embedding \
  --target_embed_save_file data/conala/.tmp/tgt_embedding \
  --sim_func cls_distance.cosine \
  --num_layers 12 \
  --save_file data/conala/retrieval_results.json
```

The results will be saved to `data/conala/retrieval_results.json`.

---
## Generation results (Table 1 of report)

### FID generation


A training or evaluation file should be converted to the format compatible with FiD. 
An example is [here](./data/conala/example_fid_data.json)
> **Important note**: FiD has a strong dependency on the version of `transformers` (3.0.2).
> Unable to match the version might result in irreproducible results.
> 
1. Run generation.
   
```bash
ds='conala'
python generator/fid/test_reader_simple.py \
    --model_path models/generator/${ds}.fid.codet5.top10/checkpoint/best_dev/best_dev/ \
    --tokenizer_name models/generator/codet5-base \
    --eval_data data/${ds}/fid.cmd_test.codet5.t10.json \
    --per_gpu_batch_size 8 \
    --n_context 10 \
    --name ${ds}.fid.codet5.top10 \
    --checkpoint_dir models/generator  \
    --result_tag test_same \
    --main_port 81692
```
The results will be saved to `models/generator/{name}/test_results_test_same.json`

---
## Pass@k

There are two notebooks titled Generate_preds.ipynb and Test_results.ipynb. These notebooks also create the required .json files needed to run the inference codes.

Run the Generate_preds.ipynb first. It also has details on when to run the retriever inference and generator inference. 
Check above sections for commands for these. 

Then run the Test_results.ipynb to get the pass@k results.

---

## Training

In order to get familiar with the training process of the retrievers and the generators, we also trained the models from scratch. 
We trained CodeT5, and did the end-to-end process.

## Training the retriever

```bash
python retriever/simcse/run_train.py \
    --num_layers 12 \
    --model_name_or_path Salesforce/codet5-base \
    --sim_func cls_distance.cosine \
    --temp 0.05  \
    --train_file data/conala/train_retriever_sup_unsup.json \
    --eval_file data/conala/dev_retriever.json \
    --output_dir models/retriever/docprompting_codet5_python_doc_retriever \
    --eval_src_file data/conala/conala_nl.txt \
    --eval_tgt_file data/conala/python_manual_firstpara.tok.txt \
    --eval_root_folder data/conala \
    --eval_oracle_file data/conala/cmd_dev.oracle_man.full.json \
    --run_name docprompting_codet5_python_doc_retriever \
    --num_train_epochs 10 \
    --per_device_train_batch_size 512 \
    --learning_rate 1e-5 \
    --max_seq_length 32 \
    --evaluation_strategy steps \
    --metric_for_best_model recall@10 \
    --load_best_model_at_end \
    --eval_steps 125 \
    --overwrite_output_dir \
    --do_train \
    --eval_form retrieval
    "$@"
```
---

### Training the generator

```bash
ds='conala'
python generator/fid/train_reader.py \
    --seed 1996 \
    --train_data data/${ds}/fid.cmd_train.codet5.t10.json \
    --eval_data data/${ds}/fid.cmd_dev.codet5.t10.json \
    --model_name models/generator/codet5-base \ # initialize with the codet5-base model \
    --per_gpu_batch_size 4 \
    --n_context 10 \
    --name ${ds}.fid.codet5.top10 \
    --checkpoint_dir models/generator/ \
    --eval_freq 500 \
    --accumulation_steps 2 \
    --main_port 30843 \
    --total_steps 20000 \
```
---

### For StarCoder as Generator

```bash
ds='conala'
python generator/fid/train_reader_starcoder.py
    --seed 1996 \
    --train_data data/${ds}/fid.cmd_train.codet5.t10.json \
    --eval_data data/${ds}/fid.cmd_dev.codet5.t10.json \
    --model_name bigcode/starcoder \
    --per_gpu_batch_size 4 \
    --n_context 10 \
    --name ${ds}.fid.codet5.top10 \
    --checkpoint_dir models/generator/ \
    --eval_freq 500 \
    --accumulation_steps 2 \
    --main_port 30843 \
    --total_steps 20000 \
    --warmup_steps 2000 \
```
Note: StarCoder is a Gated Model, to be able to access and use it, please use the steps below:

a) Accept the license agreement on https://huggingface.co/bigcode/starcoder

b) Get access token for starcoder from https://huggingface.co/settings/tokens

c) Run 'huggingface-cli login' and use token obtained in step b above.

---
## Data
The `data` folder contains the benchmark used.
* CoNaLa

On this dataset, there are
1. Natural language intent (entry `nl`)
2. Oracle code (entry `cmd`) 
  * Python for CoNaLa
3. Oracle docs (entry `oracle_man`) 
  * In the data files, we only provide the manual ids, their contents could be found in the `{dataset}/{dataset}_docs.json`.
4. Other data with different format for different modules


## Citation
```
@inproceedings{zhou23docprompting,
    title = {DocPrompting: Generating Code by Retrieving the Docs},
    author = {Shuyan Zhou and Uri Alon and Frank F. Xu and Zhiruo Wang and Zhengbao Jiang and Graham Neubig},
    booktitle = {International Conference on Learning Representations (ICLR)},
    address = {Kigali, Rwanda},
    month = {May},
    url = {https://arxiv.org/abs/2207.05987},
    year = {2023}
}
```
