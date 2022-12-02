# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import json
import os

import torch
import transformers
import numpy as np
from pathlib import Path
import torch.distributed as dist
from torch.utils.data import DataLoader, SequentialSampler
from dataset_helper.conala.gen_metric import _bleu as conala_bleu
from dataset_helper.tldr.gen_metric import tldr_metrics
from tqdm import tqdm

import src.slurm
import src.util
from src.options import Options
import src.data
import src.model

TQDM_DISABLED = os.environ['TQDM_DISABLED'] if 'TQDM_DISABLED' in os.environ else False

def evaluate(model, dataset, dataloader, tokenizer, opt):
    loss, curr_loss = 0.0, 0.0
    model.eval()
    if hasattr(model, "module"):
        model = model.module
    if opt.write_crossattention_scores:
        model.overwrite_forward_crossattention()
        model.reset_score_storage()
    total = 0

    with torch.no_grad():
        result_d = []

        with open(f"{opt.checkpoint_path}/gold.gold", "w+") as fg, \
                open(f'{opt.checkpoint_path}/pred.pred', 'w+') as fp, \
                open(opt.result_file, 'w+') as fr:
            for i, batch in enumerate(tqdm(dataloader, disable=TQDM_DISABLED)):
                (idx, _, _, context_ids, context_mask) = batch

                if opt.write_crossattention_scores:
                    model.reset_score_storage()

                outputs = model.generate(
                    input_ids=context_ids.cuda(),
                    attention_mask=context_mask.cuda(),
                    max_length=150,
                    lenpen=opt.lenpen,
                    num_beams=opt.num_beams,
                    temperature=opt.temperature,
                    top_p=opt.top_p,
                    num_return_sequences=opt.num_return_sequences,
                )
                if opt.num_return_sequences == 1:
                    for k, o in enumerate(outputs):
                        ans = tokenizer.decode(o, skip_special_tokens=False)
                        gold = dataset.get_example(idx[k])['target']
                        ans = ans.replace("{{", " {{").replace("\n", ' ').replace("\r", "").replace("<pad>", "").replace("<s>", "").replace("</s>", "").strip()
                        ans = " ".join(ans.split())
                        gold = gold.replace("\n", ' ')
                        fg.write(f"{gold}\n")
                        fp.write(f"{ans}\n")
                        cur_result = {'question_id': dataset.get_example(idx[k])['id'], 'gold': gold, 'clean_code': ans}
                        result_d.append(cur_result)
                        total += 1
                        fr.write(json.dumps(cur_result) + "\n")
                else:
                    outputs = outputs.view(-1, opt.num_return_sequences, outputs.size(-1))
                    for k, o in enumerate(outputs):
                        ans_list = []
                        gold = dataset.get_example(idx[k])['target']
                        gold = gold.replace("\n", ' ')
                        for j, oj in enumerate(o):
                            ans = tokenizer.decode(oj, skip_special_tokens=False)
                            ans = ans.replace("{{", " {{").replace("\n", ' ').replace("\r", "").replace("<pad>", "").replace("<s>", "").replace("</s>", "").strip()
                            ans = " ".join(ans.split())
                            ans_list.append(ans)
                        cur_result = {'question_id': dataset.get_example(idx[k])['id'], 'gold': gold, 'clean_code': ans_list}
                        result_d.append(cur_result)
                        total += 1
                        fr.write(json.dumps(cur_result) + "\n")


    if opt.num_return_sequences == 1:
        if opt.eval_metric == 'bleu':
            score = conala_bleu(
                f"{opt.checkpoint_path}/gold.gold",
                f"{opt.checkpoint_path}/pred.pred",
                smooth=False, code_tokenize=True)
            score = {'bleu': score}

        elif opt.eval_metric == 'token_f1':
            score = tldr_metrics(
                f"{opt.checkpoint_path}/gold.gold",
                f"{opt.checkpoint_path}/pred.pred",
            )
        else:
            raise NotImplementedError
    else:
        score = 0

    return score, total

if __name__ == "__main__":
    options = Options()
    options.add_reader_options()
    options.add_eval_options()
    opt = options.parse()
    src.slurm.init_distributed_mode(opt)
    src.slurm.init_signal_handler()
    opt.train_batch_size = opt.per_gpu_batch_size * max(1, opt.world_size)
    opt.checkpoint_path = Path(opt.checkpoint_dir) / opt.name
    opt.result_file = Path(opt.checkpoint_dir) / opt.name / f'test_results_{opt.result_tag}.json'

    dir_path = Path(opt.checkpoint_dir) / opt.name
    directory_exists = dir_path.exists()

    if opt.is_distributed:
        torch.distributed.barrier()
    dir_path.mkdir(parents=True, exist_ok=True)

    logger = src.util.init_logger(opt.is_main, opt.is_distributed, Path(opt.checkpoint_dir) / opt.name / 'run.log')

    if not directory_exists and opt.is_main:
        options.print_options(opt)

    if 'codet5' in opt.tokenizer_name:
        logger.info(f'load the tokenizer from codet5')
        tokenizer = transformers.RobertaTokenizer.from_pretrained(opt.tokenizer_name)
    else:
        logger.info(f'load the tokenizer from t5')
        tokenizer = transformers.T5Tokenizer.from_pretrained(opt.tokenizer_name)

    if opt.dataset == 'tldr':
        special_tokens_dict = {'additional_special_tokens': ['{{', '}}']}
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

    collator_function = src.data.Collator(opt.text_maxlength, tokenizer)
    eval_examples = src.data.load_data(
        opt.eval_data,
        global_rank=opt.global_rank,
        # use the global rank and world size attibutes to split the eval set on multiple gpus
        world_size=opt.world_size
    )
    eval_dataset = src.data.Dataset(
        eval_examples,
        opt.n_context,
    )

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=opt.per_gpu_batch_size,
        num_workers=20,
        collate_fn=collator_function
    )

    model_class = src.model.FiDT5
    model = model_class.from_pretrained(opt.model_path)
    model = model.to(opt.device)

    logger.info("Start eval")
    score, total = evaluate(model, eval_dataset, eval_dataloader, tokenizer, opt)

    logger.info(f'Total number of example {total}')
    logger.info(json.dumps(score, indent=2))

