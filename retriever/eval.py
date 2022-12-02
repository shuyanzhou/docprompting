import json
import copy
from collections import OrderedDict

import numpy as np
import argparse

TOP_K = [1, 3, 5, 8, 10, 12, 15, 20, 30, 50, 100, 200]

def align_src_pred(src_file, pred_file):
    with open(src_file, "r") as fsrc, open(pred_file, "r") as fpred:
        src = json.load(fsrc)
        pred = json.load(fpred)['results']
        # assert len(src) == len(pred), (len(src), len(pred))

    # re-order src
    src_nl = [x['nl'] for x in src]
    _src = []
    _pred = []
    for p in pred:
        if p['nl'] in src_nl:
            _src.append(src[src_nl.index(p['nl'])])
            _pred.append(p)

    src = _src
    pred = _pred

    for s, p in zip(src, pred):
        assert s['nl'] == p['nl'], (s['nl'], p['nl'])

    print(f"unique nl: {len(set(src_nl))}")
    print(f"number of samples (src/pred): {len(src)}/{len(pred)}")
    print("pass nl matching check")

    return src, pred

def calc_metrics(src_file, pred_file):
    src, pred = align_src_pred(src_file, pred_file)

    _src = []
    _pred = []
    for s, p in zip(src, pred):
        cmd_name = s['cmd_name']
        oracle_man = get_oracle(s, cmd_name)
        pred_man = p['pred']
        _src.append(oracle_man)
        _pred.append(pred_man)
    calc_recall(_src, _pred)

    # description only
    _src = []
    for s in src:
        _src.append(s['matching_info']['|main|'])
    calc_recall(_src, _pred)

    _src = []
    _pred = []
    for s, p in zip(src, pred):
        cmd_name = s['cmd_name']
        pred_man = p['pred']
        _src.append(cmd_name)
        _pred.append(pred_man)
    calc_hit(_src, _pred)
    # calc_mean_rank(src, pred)


def calc_mean_rank(src, pred):
    rank = []
    for s, p in zip(src, pred):
        cur_rank = []
        cmd_name = s['cmd_name']
        pred_man = p['pred']
        oracle_man = get_oracle(s, cmd_name)
        for o in oracle_man:
            if o in pred_man:
                cur_rank.append(oracle_man.index(o))
            else:
                cur_rank.append(101)
        if cur_rank:
            rank.append(np.mean(cur_rank))

    print(np.mean(rank))


def calc_hit(src, pred, top_k=None):
    top_k = TOP_K if top_k is None else top_k
    hit_n = {x: 0 for x in top_k}
    assert len(src) == len(pred), (len(src), len(pred))

    for s, p in zip(src, pred):
        cmd_name = s
        pred_man = p

        for tk in hit_n.keys():
            cur_result_vids = pred_man[:tk]
            cur_hit = any([cmd_name in x for x in cur_result_vids])
            hit_n[tk] += cur_hit

    hit_n = {k: v / len(pred) for k, v in hit_n.items()}
    for k in sorted(hit_n.keys()):
        print(f"{hit_n[k] :.3f}", end="\t")
    print()
    return hit_n

def get_oracle(item, cmd_name):
    # oracle = [f"{cmd_name}_{x}" for x in itertools.chain(*item['matching_info'].values())]
    oracle = [f"{cmd_name}_{x}" for x in item['oracle_man']]
    return oracle

def calc_recall(src, pred, print_result=True, top_k=None):
    top_k = TOP_K if top_k is None else top_k
    recall_n = {x: 0 for x in top_k}
    precision_n = {x: 0 for x in top_k}

    for s, p in zip(src, pred):
        # cmd_name = s['cmd_name']
        oracle_man = s
        pred_man = p

        for tk in recall_n.keys():
            cur_result_vids = pred_man[:tk]
            cur_hit = sum([x in cur_result_vids for x in oracle_man])
            # recall_n[tk] += cur_hit / (len(oracle_man) + 1e-10)
            recall_n[tk] += cur_hit / (len(oracle_man)) if len(oracle_man) else 1
            precision_n[tk] += cur_hit / tk
    recall_n = {k: v / len(pred) for k, v in recall_n.items()}
    precision_n = {k: v / len(pred) for k, v in precision_n.items()}

    if print_result:
        for k in sorted(recall_n.keys()):
            print(f"{recall_n[k] :.3f}", end="\t")
        print()
        for k in sorted(precision_n.keys()):
            print(f"{precision_n[k] :.3f}", end="\t")
        print()
        for k in sorted(recall_n.keys()):
            print(f"{2 * precision_n[k] * recall_n[k] / (precision_n[k] + recall_n[k] + 1e-10) :.3f}", end="\t")
        print()

    return {'recall': recall_n, 'precision': precision_n}

def clean_dpr_results(result_file):
    results = {'results': [], 'metrics': {}}
    with open(result_file, "r") as f:
        d = json.load(f)
    for _item in d:
        item = {}
        item['nl'] = _item['question']
        item['pred'] = [x['id'] for x in _item['ctxs']]
        results['results'].append(item)

    with open(result_file + ".clean", "w+") as f:
        json.dump(results, f, indent=2)

def recall_per_manual(src_file, result_file, chunk_length_file, topk):

    def find_sum_in_list(len_list, max_num):
        idx = len(len_list)
        for i in range(len(len_list) + 1):
            if sum(len_list[:i]) >= max_num:
                idx = i - 1
                break
        assert sum(len_list[:idx]) <= max_num
        return idx

    with open(chunk_length_file, "r") as f:
        d = json.load(f)
        man_chunk_length = {k: len(v) for k, v in d.items()}

    src, pred = align_src_pred(src_file, result_file)
    hit_man = 0
    recall = 0
    tot = len(src)
    for s, p in zip(src, pred):
        cmd_name = s['cmd_name']
        oracle_man = get_oracle(s, cmd_name)
        pred_man = p['pred']
        top_k_cmd = p['top_pred_cmd'][:topk]
        if cmd_name in top_k_cmd:
            hit_man += 1
            pred_chunks = pred_man[cmd_name]
            len_list = [man_chunk_length[x] for x in pred_chunks]
            idx = find_sum_in_list(len_list, 1536)
            pred_chunks = pred_chunks[:idx]
            cur_hit = sum([x in pred_chunks for x in oracle_man])
            recall += cur_hit / len(oracle_man)

    print(f"hit rate: {hit_man}/{tot}={hit_man/tot}")
    print(f"recall: {recall}/{hit_man}={recall/hit_man}")


def eval_hit_from_file(data_file, retrieval_file,
                             oracle_entry='oracle_man', retrieval_entry='retrieved'):
    assert 'tldr' in data_file
    with open(data_file, "r") as f:
        d = json.load(f)
    gold = ['_'.join(item[oracle_entry][0].split("_")[:-1]) for item in d]

    with open(retrieval_file, "r") as f:
        r_d = json.load(f)
        # check whether we need to process the retrieved ids
        split_flag = False
        k0 = list(r_d.keys())[0]
        r0 = r_d[k0][retrieval_entry][0]
        if r0.split("_")[-1].isdigit():
            split_flag = True

        for k, item in r_d.items():
            if split_flag:
                r = ['_'.join(x.split("_")[:-1]) for x in item[retrieval_entry]]
            else:
                r = item[retrieval_entry]
            r = list(OrderedDict.fromkeys(r))
            item[retrieval_entry] = r

    pred = [r_d[x['question_id']][retrieval_entry] for x in d]
    print(gold[:3])
    print(pred[0][:3])
    metrics = calc_hit(gold, pred)
    return {'hit': metrics}

def eval_retrieval_from_file(data_file, retrieval_file,
                             oracle_entry='oracle_man', retrieval_entry='retrieved', top_k=None):

    assert 'oracle_man.full' in data_file or 'conala' not in data_file, (data_file)
    # for conala
    with open(data_file, "r") as f:
        d = json.load(f)
    gold = [item[oracle_entry] for item in d]

    with open(retrieval_file, "r") as f:
        r_d = json.load(f)
    pred = [r_d[x['question_id']][retrieval_entry] for x in d]
    metrics = calc_recall(gold, pred, top_k=top_k)
    return metrics

def eval_retrieval_from_loaded(data_file, r_d):
    # for conala
    with open(data_file, "r") as f:
        d = json.load(f)
    gold = [item['oracle_man'] for item in d]
    pred = [r_d[x['question_id']]['retrieved'] for x in d]
    metrics = calc_recall(gold, pred, print_result=False)
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--result-file', default=None)
    parser.add_argument('--src-file', default=None)
    parser.add_argument('--chunk-length-file', default="data/tldr/nl.cm/manual_section.tok.json")
    parser.add_argument('--function', nargs='+', type=int, default=[1])
    args = parser.parse_args()

    for cur_func in args.function:
        if cur_func == 0:
            calc_metrics(args.src_file, args.result_file)
        elif cur_func == 1:
            # convert data
            clean_dpr_results(args.result_file)
        elif cur_func == 2:
            clean_dpr_results(args.result_file)
            args.result_file += ".clean"
            calc_metrics(args.src_file, args.result_file)
        elif cur_func == 3:
            # measure recall for per-doc retrieval
            for k in [1, 10, 30, 50]:
                print(f"top {k}")
                recall_per_manual(args.src_file, args.result_file, args.chunk_length_file, topk=k)
