# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import json
import os
import sys
sys.path.append('/Users/ruchakulkarni/Desktop/docprompting')
import torch
import transformers
import numpy as np
from pathlib import Path
import torch.distributed as dist
from torch.utils.data import DataLoader, SequentialSampler
from dataset_helper.conala.gen_metric import _bleu as conala_bleu
# from dataset_helper.tldr.gen_metric import tldr_metrics
from tqdm import tqdm

import src.slurm
import src.util
from src.options import Options
import src.data
import src.model
from transformers import AutoTokenizer


TQDM_DISABLED = os.environ['TQDM_DISABLED'] if 'TQDM_DISABLED' in os.environ else False


def evaluate(model, dataset, dataloader, tokenizer, opt):
    # loss, curr_loss = 0.0, 0.0
    # model.eval()
    # if hasattr(model, "module"):
    #     model = model.module
    # if opt.write_crossattention_scores:
    #     model.overwrite_forward_crossattention()
    #     model.reset_score_storage()
    # total = 0
    #/home/ubuntu/docprompting/data/conala/fid.cmd_test.codet5.t10_small.json
    with torch.no_grad():
        result_d = []

        with open(f"{opt.checkpoint_path}/gold.gold", "w+") as fg, \
                open(f'{opt.checkpoint_path}/pred.pred', 'w+') as fp, \
                open(opt.result_file, 'w+') as fr, open('data/conala/fid.cmd_test.codet5.t10_small.json') as data_file:
                
               
                data = json.load(data_file)
                # for prompt in data.values():
                for each in data:
                    print("onto next data point")
                # if opt.write_crossattention_scores:
                #     model.reset_score_storage()


                    system = "Provide answers in Python. "
                    system+="Context: "
                    system+= each['ctxs'][0]['text']
                    user = each['question']
                    
                    prompt = system + user
                    # prompt = f"<s><<SYS>>\\n{system}\\n<</SYS>>\\n\\n{user}"
                    print("the final prompt is", prompt)
                    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")

                    # sequences = pipeline(
                    #     prompt,
                    #     do_sample=True,
                    #     top_k=10,
                    #     temperature=0.1,
                    #     top_p=0.95,
                    #     num_return_sequences=1,
                    #     eos_token_id=tokenizer.eos_token_id,
                    #     max_length=200
                    # )
                    output = model.generate(
                    inputs["input_ids"],
                    do_sample=True,
                    top_p=0.95,
                    temperature=0.1,
                    max_length=200
                    )

                    print("reached here")

                    if opt.num_return_sequences == 1:
                        # for k, o in enumerate(outputs):
                        #     # ans = tokenizer.decode(o, skip_special_tokens=False)
                        #     gold = dataset.get_example(idx[k])['target']
                        #     ans = ans.replace("{{", " {{").replace("\n", ' ').replace("\r", "").replace("<pad>", "").replace("<s>", "").replace("</s>", "").strip()
                        #     ans = " ".join(ans.split())
                        #     gold = gold.replace("\n", ' ')
                        #     fg.write(f"{gold}\n")
                        #     fp.write(f"{ans}\n")
                        #     cur_result = {'question_id': dataset.get_example(idx[k])['id'], 'gold': gold, 'clean_code': ans}
                        #     result_d.append(cur_result)
                        #     total += 1
                        #     fr.write(json.dumps(cur_result) + "\n")

                        # for k, o in enumerate(outputs):
                        # for seq in sequences:
                        
                        ans = tokenizer.decode(output[0].to("cpu"))
                        # ans = seq['generated_text']
                        print("the ans is", ans)
                        # gold = dataset.get_example(idx[k])['target']
                        # print("typeeeee",type(prompt))
                        gold = each['target']
                        ans = ans.replace("{{", " {{").replace("\n", ' ').replace("\r", "").replace("<pad>", "").replace("<s>", "").replace("</s>", "").strip()
                        ans = " ".join(ans.split())
                        gold = gold.replace("\n", ' ')
                        fg.write(f"{gold}\n")
                        fp.write(f"{ans}\n")
                        # cur_result = {'question_id': dataset.get_example(idx[k])['id'], 'gold': gold, 'clean_code': ans}
                        cur_result = {'question_id': each['id'], 'gold': gold, 'clean_code': ans}
                        result_d.append(cur_result)
                        # total += 1
                        fr.write(json.dumps(cur_result) + "\n")

                    # else:
                    #     outputs = outputs.view(-1, opt.num_return_sequences, outputs.size(-1))
                    #     for k, o in enumerate(outputs):
                    #         ans_list = []
                    #         gold = dataset.get_example(idx[k])['target']
                    #         gold = gold.replace("\n", ' ')
                    #         for j, oj in enumerate(o):
                    #             ans = tokenizer.decode(oj, skip_special_tokens=False)
                    #             ans = ans.replace("{{", " {{").replace("\n", ' ').replace("\r", "").replace("<pad>", "").replace("<s>", "").replace("</s>", "").strip()
                    #             ans = " ".join(ans.split())
                    #             ans_list.append(ans)
                    #         cur_result = {'question_id': dataset.get_example(idx[k])['id'], 'gold': gold, 'clean_code': ans_list}
                    #         result_d.append(cur_result)
                    #         total += 1
                    #         fr.write(json.dumps(cur_result) + "\n")

    if opt.num_return_sequences == 1:
        print("evaluating BLEU")
        if opt.eval_metric == 'bleu':
            score = conala_bleu(
                f"{opt.checkpoint_path}/gold.gold",
                f"{opt.checkpoint_path}/pred.pred",
                smooth=False, code_tokenize=True)
            score = {'bleu': score}

        # elif opt.eval_metric == 'token_f1':
        #     score = tldr_metrics(
        #         f"{opt.checkpoint_path}/gold.gold",
        #         f"{opt.checkpoint_path}/pred.pred",
        #     )
        else:
            raise NotImplementedError
    else:
        score = 0

    total = 0
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


    model = "bigcode/starcoder"
    tokenizer = AutoTokenizer.from_pretrained(model)

    collator_function = src.data.Collator(opt.text_maxlength, tokenizer)
    opt.eval_data = "data/conala/fid.cmd_test.codet5.t10_small.json"
    print("opt.eval_data", opt.eval_data)
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

    model = transformers.AutoModelForCausalLM.from_pretrained("bigcode/starcoder", load_in_8bit=True,
    torch_dtype=torch.float16,device_map="auto")

    logger.info("Start eval")
    score, total = evaluate(model, eval_dataset, eval_dataloader, tokenizer, opt)
    print("score is", score)
    print("total is", score)
    print("eval_dataset is", eval_dataset)
    print("eval_dataloader is", eval_dataloader)

    logger.info(f'Total number of example {total}')
    logger.info(json.dumps(score, indent=2))




