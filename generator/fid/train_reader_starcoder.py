# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import json
import os
import time
import sys
sys.path.append('/Users/ruchakulkarni/Desktop/docprompting')
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from tqdm import tqdm

from dataset_helper.conala.gen_metric import _bleu as conala_bleu
from dataset_helper.tldr.gen_metric import tldr_metrics
from src.options import Options

import src.slurm
import src.util
import src.evaluation
import src.data
import src.model
import wandb

WANDB_DISABLED= os.environ['WANDB_DISABLED'] if 'WANDB_DISABLED' in os.environ else False
TQDM_DISABLED = os.environ['TQDM_DISABLED'] if 'TQDM_DISABLED' in os.environ else False

def train(model, optimizer, scheduler, step, train_dataset, eval_dataset, opt, collator, best_dev_em, checkpoint_path):

    # if opt.is_main:
    #     try:
    #         tb_logger = torch.utils.tensorboard.SummaryWriter(Path(opt.checkpoint_dir)/opt.name)
    #     except:
    #         tb_logger = None
    #         logger.warning('Tensorboard is not available.')

    torch.manual_seed(opt.global_rank + opt.seed) #different seed for different sampling depending on global_rank
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=opt.per_gpu_batch_size,
        drop_last=True,
        num_workers=10,
        collate_fn=collator
    )

    loss, curr_loss = 0.0, 0.0
    epoch = 1
    model.train()
    while step < opt.total_steps:
        epoch += 1
        opt._epoch = epoch
        for i, batch in enumerate(tqdm(train_dataloader, disable=TQDM_DISABLED)):
            step += 1
            opt._train_step = step
            (idx, labels, _, context_ids, context_mask) = batch

            train_loss = model(
                input_ids=context_ids.cuda(),
                attention_mask=context_mask.cuda(),
                labels=labels.cuda()
            )[0]

            train_loss.backward()

            if step % opt.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt.clip)
                optimizer.step()
                scheduler.step()
                model.zero_grad()

            train_loss = src.util.average_main(train_loss, opt)
            curr_loss += train_loss.item()
            wandb.log({'train_loss': train_loss.item(), 'lr': scheduler.get_last_lr()[0]})

            if step % opt.eval_freq == 0:
                dev_em = evaluate(model, eval_dataset, tokenizer, collator, opt)
                wandb.log({f'eval_{x}': y for x, y in dev_em.items()})
                dev_em = dev_em[opt.eval_metric]
                model.train()
                if opt.is_main:
                    if dev_em > best_dev_em:
                        best_dev_em = dev_em
                        src.util.save(model, optimizer, scheduler, step, best_dev_em,
                                      opt, checkpoint_path, 'best_dev')
                    log = f"{step} / {opt.total_steps} |"
                    log += f"train: {curr_loss/opt.eval_freq:.3f} |"
                    log += f"evaluation {opt.eval_metric}: {dev_em:.02f}|"
                    log += f"lr: {scheduler.get_last_lr()[0]:.5f}"
                    logger.info(log)
                    # if tb_logger is not None:
                    #     tb_logger.add_scalar("Evaluation", dev_em, step)
                    #     tb_logger.add_scalar("Training", curr_loss / (opt.eval_freq), step)
                    curr_loss = 0.


            if opt.is_main and step % opt.save_freq == 0:
                src.util.save(model, optimizer, scheduler, step, best_dev_em,
                              opt, checkpoint_path, f"step-{step}")

            if step > opt.total_steps:
                break

def evaluate_em(model, dataset, tokenizer, collator, opt):
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset,
        sampler=sampler,
        batch_size=opt.per_gpu_batch_size,
        drop_last=False,
        num_workers=10,
        collate_fn=collator
    )
    model.eval()
    total = 0
    exactmatch = []
    model = model.module if hasattr(model, "module") else model
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            (idx, _, _, context_ids, context_mask) = batch

            outputs = model.generate(
                input_ids=context_ids.cuda(),
                attention_mask=context_mask.cuda(),
                max_length=50
            )

            for k, o in enumerate(outputs):
                ans = tokenizer.decode(o, skip_special_tokens=True)
                gold = dataset.get_example(idx[k])['answers']
                score = src.evaluation.ems(ans, gold)
                total += 1
                exactmatch.append(score)

    exactmatch, total = src.util.weighted_average(np.mean(exactmatch), total, opt)
    return exactmatch

def evaluate_customized(model, dataset, tokenizer, collator, opt, result_file=None, is_bleu=False, is_token_f1=False):
    assert is_bleu != is_token_f1
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset,
        sampler=sampler,
        batch_size=opt.per_gpu_batch_size,
        drop_last=False,
        num_workers=10,
        collate_fn=collator
    )
    model.eval()
    model = model.module if hasattr(model, "module") else model

    with torch.no_grad():

        result_file = f"{opt.checkpoint_path}/dev_result_{opt._train_step}.json" if result_file is None else result_file
        result_d = []

        with open(f"{opt.checkpoint_path}/gold.gold", "w+") as fg, open(f'{opt.checkpoint_path}/pred.pred', 'w+') as fp, open(result_file, 'w+') as fr:

            for i, batch in enumerate(tqdm(dataloader, disable=TQDM_DISABLED)):
                (idx, _, _, context_ids, context_mask) = batch

                outputs = model.generate(
                    input_ids=context_ids.cuda(),
                    attention_mask=context_mask.cuda(),
                    max_length=150,
                )

                for k, o in enumerate(outputs):
                    ans = tokenizer.decode(o, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                    gold = dataset.get_example(idx[k])['target']
                    ans = ans.replace("{{", " {{").replace("\n", ' ').replace("\r", "").replace("<pad>", "").replace("<s>", "").replace("</s>", "").strip()
                    ans = " ".join(ans.split())
                    gold = gold.replace("\n", ' ')
                    fg.write(f"{gold}\n")
                    fp.write(f"{ans}\n")
                    cur_result = {'question_id': dataset.get_example(idx[k])['id'], 'gold': gold, 'clean_code': ans}
                    result_d.append(cur_result)

            json.dump(result_d, fr, indent=2)



        if is_bleu:
            score = conala_bleu(
                f"{opt.checkpoint_path}/gold.gold",
                f"{opt.checkpoint_path}/pred.pred",
                smooth=False, code_tokenize=True)
            score = {'bleu': score}
        elif is_token_f1:
            score = tldr_metrics(
                f"{opt.checkpoint_path}/gold.gold",
                f"{opt.checkpoint_path}/pred.pred")

        else:
            raise NotImplementedError

    return score

def evaluate(model, dataset, tokenizer, collator, opt):
    if opt.eval_metric == 'exact_match':
        x = evaluate_em(model, dataset, tokenizer, collator, opt)
        metric = {'exact_match': x}
    elif opt.eval_metric == 'bleu':
        metric = evaluate_customized(model, dataset, tokenizer, collator, opt, is_bleu=True)
    elif opt.eval_metric == 'token_f1':
        metric = evaluate_customized(model, dataset, tokenizer, collator, opt, is_token_f1=True)
        metric['token_f1'] = metric.pop('f1')
    else:
        raise NotImplementedError(f'{opt.eval_metric} has not been implemented yet')
    print(json.dumps(metric, indent=2))
    return metric

if __name__ == "__main__":
    options = Options()
    options.add_reader_options()
    options.add_optim_options()
    opt = options.parse()

    #opt = options.get_options(use_reader=True, use_optim=True)

    torch.manual_seed(opt.seed)
    src.slurm.init_distributed_mode(opt)
    src.slurm.init_signal_handler()

    checkpoint_path = Path(opt.checkpoint_dir)/opt.name
    checkpoint_exists = checkpoint_path.exists()
    if opt.is_distributed:
        torch.distributed.barrier()
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    opt.checkpoint_path = checkpoint_path
    #if not checkpoint_exists and opt.is_main:
    #    options.print_options(opt)
    #checkpoint_path, checkpoint_exists = util.get_checkpoint_path(opt)

    logger = src.util.init_logger(
        opt.is_main,
        opt.is_distributed,
        checkpoint_path / 'run.log'
    )

    logger.info(f"device type: {torch.cuda.get_device_name(0)}, memory: {torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024}G")

    # logger.info(json.dumps(vars(opt), indent=2))

    if WANDB_DISABLED:
        wandb.init(mode='disabled')
    else:
        if opt.is_main:
            # is the master
            wandb.init(project='fid')
            wandb.config.update(opt)
        else:
            wandb.init(mode='disabled')

    # model_name = 't5-' + opt.model_size
    model_name = opt.model_name
    model_class = src.model.FiDT5

    # #load data
    # if 'codet5' in model_name or 'code_t5' in model_name:
    #     logger.info(f'load the tokenizer from codet5')
    #     tokenizer = transformers.RobertaTokenizer.from_pretrained(model_name)
    # else:
    #     logger.info(f'load the tokenizer from t5')
    #     tokenizer = transformers.T5Tokenizer.from_pretrained(model_name)

    ########
    # #load data
    # if 'codet5' in model_name or 'code_t5' in model_name:
    #     logger.info(f'load the tokenizer from codet5')
    #     tokenizer = transformers.RobertaTokenizer.from_pretrained(model_name)
    # else:
    #     logger.info(f'load the tokenizer from t5')
    #     tokenizer = transformers.T5Tokenizer.from_pretrained(model_name)

    # codellama
    #load data
    # if 'codet5' in model_name or 'code_t5' in model_name:
    # bigcode/starcoder
    logger.info(f'load the tokenizer from starcoder')
    # tokenizer = transformers.RobertaTokenizer.from_pretrained(model_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    ########

    if opt.dataset == 'tldr':
        special_tokens_dict = {'additional_special_tokens': ['{{', '}}']}
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

    collator = src.data.Collator(opt.text_maxlength, tokenizer,
                                 answer_maxlength=opt.answer_maxlength)

    # use golbal rank and world size to split the eval set on multiple gpus
    train_examples = src.data.load_data(
        opt.train_data, 
        global_rank=opt.global_rank, 
        world_size=opt.world_size,
    )
    train_dataset = src.data.Dataset(train_examples, opt.n_context)
    # use golbal rank and world size to split the eval set on multiple gpus
    eval_examples = src.data.load_data(
        opt.eval_data,
        global_rank=opt.global_rank,
        world_size=opt.world_size,
    )
    eval_dataset = src.data.Dataset(eval_examples, opt.n_context)

    if not opt.continue_from_checkpoint:
        logger.info("init a model from T5")
        t5 = transformers.T5ForConditionalGeneration.from_pretrained(model_name)
        t5.resize_token_embeddings(len(tokenizer))

        if opt.encoder_weights is not None:
            state_dict = torch.load(f'{opt.encoder_weights}/pytorch_model.bin')
            # rename
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('_model.encoder.'):
                    k = k.replace('_model.encoder.', '')
                new_state_dict[k] = v
            load_model_keys = list(new_state_dict.keys())
            model_keys = list(t5.encoder.state_dict().keys())
            ignored = []
            missed = []
            for k in load_model_keys:
                if k not in model_keys:
                    ignored.append(k)
            for k in model_keys:
                if k not in load_model_keys:
                    missed.append(k)

            logger.info(f'Some weights in the checkpoint are not used when initializing the encoder : {ignored}')
            logger.info(f'Some weights in the encoder were not initialized from the checkpoint : {missed}')
            t5.encoder.load_state_dict(new_state_dict, strict=False)
            logger.info(f'Loaded encoder weights from {opt.encoder_weights}')

        model = src.model.FiDT5(t5.config)
        model.load_t5(t5.state_dict())
        model = model.to(opt.local_rank)
        optimizer, scheduler = src.util.set_optim(opt, model)
        step, best_dev_em = 0, 0.0
    elif opt.model_path == "none" and opt.cont_from_checkpoint:
        load_path = checkpoint_path / 'checkpoint' / 'latest'
        model, optimizer, scheduler, opt_checkpoint, step, best_dev_em = \
            src.util.load(model_class, load_path, opt, reset_params=False)
        logger.info(f"Model loaded from checkpoint {load_path}")
    else: # load from model path
        model, optimizer, scheduler, opt_checkpoint, step, best_dev_em = \
            src.util.load(model_class, opt.model_path, opt, reset_params=True)
        logger.info(f"Model loaded from a model {opt.model_path}")

    model.set_checkpoint(opt.use_checkpoint)

    if opt.is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[opt.local_rank],
            output_device=opt.local_rank,
            find_unused_parameters=False,
        )

    logger.info("Start training")
    train(
        model,
        optimizer,
        scheduler,
        step,
        train_dataset,
        eval_dataset,
        opt,
        collator,
        best_dev_em,
        checkpoint_path
    )
