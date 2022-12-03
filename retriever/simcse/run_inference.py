import json
import os.path
import pickle

import argparse
import shlex

import faiss
import numpy as np
import torch
from tqdm import tqdm
import transformers
from transformers import AutoModel, AutoTokenizer, AutoConfig
from retriever.eval import eval_retrieval_from_file
from model import RetrievalModel
TQDM_DISABLED = os.environ['TQDM_DISABLED'] if 'TQDM_DISABLED' in os.environ else False

class Dummy:
    pass

class CodeT5Retriever:
    def __init__(self, args):
        self.args = args

    def prepare_model(self, model=None, tokenizer=None, config=None):
        if self.args.log_level == 'verbose':
            transformers.logging.set_verbosity_info()
        self.model_name = self.args.model_name

        if model is None:
            self.tokenizer = transformers.RobertaTokenizer.from_pretrained(self.model_name)
            model_arg = Dummy()
            setattr(model_arg, 'sim_func', args.sim_func)
            config = AutoConfig.from_pretrained(self.model_name)
            self.model = RetrievalModel(
                config=config,
                model_type=self.model_name,
                num_layers=args.num_layers,
                tokenizer=tokenizer,
                training_args=None,
                model_args=model_arg)
            self.device = torch.device('cuda') if not self.args.cpu else torch.device('cpu')
            self.model.eval()
            self.model = self.model.to(self.device)
        else: # this is only for evaluation durning training time
            self.model = model
            self.tokenizer = tokenizer
            self.device = self.model.device

    def encode_file(self, text_file, save_file, **kwargs):
        normalize_embed = kwargs.get('normalize_embed', False)
        with open(text_file, "r") as f:
            dataset = []
            for line in f:
                dataset.append(line.strip())
                # print(line)
        print(f"number of sentences in {text_file}: {len(dataset)}")

        def pad_batch(examples):
            sentences = examples
            sent_features = self.tokenizer(
                sentences,
                add_special_tokens=True,
                max_length=self.tokenizer.model_max_length,
                truncation=True
            )
            arr = sent_features['input_ids']
            lens = torch.LongTensor([len(a) for a in arr])
            max_len = lens.max().item()
            padded = torch.ones(len(arr), max_len, dtype=torch.long) * self.tokenizer.pad_token_id
            mask = torch.zeros(len(arr), max_len, dtype=torch.long)
            for i, a in enumerate(arr):
                padded[i, : lens[i]] = torch.tensor(a, dtype=torch.long)
                mask[i, : lens[i]] = 1
            return {'input_ids': padded, 'attention_mask': mask, 'lengths': lens}

        bs = 128
        with torch.no_grad():
            all_embeddings = []
            for i in tqdm(range(0, len(dataset), bs), disable=TQDM_DISABLED):
                batch = dataset[i: i + bs]
                padded_batch = pad_batch(batch)
                for k in padded_batch:
                    if isinstance(padded_batch[k], torch.Tensor):
                        padded_batch[k] = padded_batch[k].to(self.device)
                output = self.model.get_pooling_embedding(**padded_batch, normalize=normalize_embed).detach().cpu().numpy()
                all_embeddings.append(output)

            all_embeddings = np.concatenate(all_embeddings, axis=0)
            print(f"done embedding: {all_embeddings.shape}")

        if not os.path.exists(os.path.dirname(save_file)):
            os.makedirs(os.path.dirname(save_file))

        np.save(save_file, all_embeddings)

    @staticmethod
    def retrieve(source_embed_file, target_embed_file, source_id_file, target_id_file, top_k, save_file):
        print(f'source: {source_embed_file}, target: {target_embed_file}')
        with open(source_id_file, "r") as f:
            source_id_map = {}
            for idx, line in enumerate(f):
                source_id_map[idx] = line.strip()
        with open(target_id_file, "r") as f:
            target_id_map = {}
            for idx, line in enumerate(f):
                target_id_map[idx] = line.strip()

        source_embed = np.load(source_embed_file + ".npy")
        target_embed = np.load(target_embed_file + ".npy")
        assert len(source_id_map) == source_embed.shape[0]
        assert len(target_id_map) == target_embed.shape[0]
        indexer = faiss.IndexFlatIP(target_embed.shape[1])
        indexer.add(target_embed)
        print(source_embed.shape, target_embed.shape)
        D, I = indexer.search(source_embed, top_k)

        results = {}
        for source_idx, (dist, retrieved_index) in enumerate(zip(D, I)):
            source_id = source_id_map[source_idx]
            results[source_id] = {}
            retrieved_target_id = [target_id_map[x] for x in retrieved_index]
            results[source_id]['retrieved'] = retrieved_target_id
            results[source_id]['score'] = dist.tolist()

        with open(save_file, "w+") as f:
            json.dump(results, f, indent=2)

        return results

def config(in_program_call=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--batch_size', type=int, default=48)
    parser.add_argument('--source_file', default='data/conala/conala_nl.txt')
    parser.add_argument('--target_file', default='data/conala/python_manual_firstpara.txt')
    parser.add_argument('--source_embed_save_file', default='data/conala/.tmp/src_embedding')
    parser.add_argument('--target_embed_save_file', default='data/conala/.tmp/tgt_embedding')
    parser.add_argument('--save_file', default='[REPLACE]data/conala/simcse.[MODEL].[SOURCE].[TARGET].[POOLER].t[TOPK].json')
    parser.add_argument('--top_k', type=int, default=200)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--pooler', choices=('cls', 'cls_before_pooler'), default='cls')
    parser.add_argument('--log_level', default='verbose')
    parser.add_argument('--nl_cm_folder', default='data/conala/nl.cm')
    parser.add_argument('--sim_func', default='cls_distance.cosine', choices=('cls_distance.cosine', 'cls_distance.l2', 'bertscore'))
    parser.add_argument('--num_layers', type=int, default=12)
    parser.add_argument('--origin_mode', action='store_true')
    parser.add_argument('--oracle_eval_file', default='data/conala/cmd_dev.oracle_man.full.json')
    parser.add_argument('--eval_hit', action='store_true')
    parser.add_argument('--normalize_embed', action='store_true')



    args = parser.parse_args() if in_program_call is None else parser.parse_args(shlex.split(in_program_call))

    args.source_idx_file = args.source_file.replace(".txt", ".id")
    args.target_idx_file = args.target_file.replace(".txt", ".id")

    if in_program_call is None and args.save_file.startswith("[REPLACE]"):
        args.save_file = args.save_file.replace("[REPLACE]", "")
        args.save_file = args.save_file.replace("[MODEL]", os.path.basename(args.model_name))
        args.save_file = args.save_file.replace("[SOURCE]", os.path.basename(args.source_file).split(".")[0])
        args.save_file = args.save_file.replace("[TARGET]", os.path.basename(args.target_file).split(".")[0])
        args.save_file = args.save_file.replace("[POOLER]", args.pooler)
        args.save_file = args.save_file.replace("[TOPK]", str(args.top_k))
    print(json.dumps(vars(args), indent=2))
    return args

if __name__ == "__main__":
    args = config()

    searcher = CodeT5Retriever(args)
    searcher.prepare_model()
    searcher.encode_file(args.source_file, args.source_embed_save_file, normalize_embed=args.normalize_embed)
    searcher.encode_file(args.target_file, args.target_embed_save_file, normalize_embed=args.normalize_embed)
    searcher.retrieve(args.source_embed_save_file,
                      args.target_embed_save_file, args.source_idx_file,
                      args.target_idx_file, args.top_k, args.save_file)

    flag = 'recall'
    top_n = 10
    m1 = eval_retrieval_from_file(args.oracle_eval_file, args.save_file)


