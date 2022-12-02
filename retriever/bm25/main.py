import itertools
import json
import os.path

from tqdm import tqdm
from collections import defaultdict, OrderedDict
from indexer import ESSearch
from retriever.eval import calc_recall, calc_hit, eval_retrieval_from_file
import argparse

def retrieve_manual(test_file, source, index, search_conf, saved_file, top_k=10):
    indexer = ESSearch(index, source, re_index=False)

    query_type, search_field = search_conf['query'], search_conf['field']
    tag = f"{index}.{query_type}.{search_field}"
    with open(test_file, "r") as f:
        d = json.load(f)
    results = []
    for item in tqdm(d):
        query = item[query_type]
        try:
            r_mans = indexer.get_topk(search_field, query, topk=top_k)

            results.append({**item,
                            f'{tag}.retrieved': [x['library_key'] for x in r_mans],
                            f'{tag}.score': [x.score if not isinstance(x, dict) else x['score'] for x in r_mans]})
        except Exception as e:
            print(repr(e))
            results.append({**item,
                            f'{tag}.retrieved': [],
                            f'{tag}.score': []})

    # metrics = calc_recall(results)
    saved_file = saved_file.replace(".json", f".{tag}.json")
    print(f"save to {saved_file}")
    with open(saved_file, "w+") as f:
        json.dump(results, f, indent=2)

    return saved_file


def doc_base_retrieval(index, source, r1_result_file,
                       retrieval_entry, saved_file, top_k_doc,
                       top_k_result, oracle_only=False):
    indexer = ESSearch(index, source, re_index=False)

    print(f"index: {index}")
    print(f"r1 file: {r1_result_file}")
    print(f"r2 file: {saved_file}")

    with open(r1_result_file, "r") as f:
        r1_result = json.load(f)

    split_flag = False
    r0 = r1_result[0][retrieval_entry][0]
    if r0.split("_")[-1].isdigit():
        split_flag = True

    r2_result = []

    for item in tqdm(r1_result):
        if oracle_only:
            if split_flag:
                pred_cmd = ["_".join(item['cmd_name'].split("_")[:-1])]
            else:
                pred_cmd = [item['cmd_name']]
        else:
            pred_cmd = []
            for pred in item[retrieval_entry]:
                cmd_name = pred
                if split_flag:
                    cmd_name = "_".join(pred.split("_")[:-1])
                pred_cmd.append(cmd_name)
            pred_cmd = list(OrderedDict.fromkeys(pred_cmd))

        for cmd_idx, cmd in enumerate(pred_cmd[:top_k_doc]):
            item_r2 = item.copy()
            item_r2['parent_cmd'] = cmd
            item_r2['question_id'] = f"{item['question_id']}-{cmd_idx}"
            item_r2.pop(retrieval_entry)
            item_r2.pop(retrieval_entry.replace(".retrieved", ".score"))
            try:
                real_query = {
                    "query": {
                        "bool": {
                            "must": [
                                {"term": {'cmd_name.keyword': cmd}},
                                {"match": {'manual': item['nl']}}
                            ]
                        }
                    },
                    "size": top_k_result
                }
                r_mans = indexer.es.search(index=indexer.index, body=real_query)['hits']['hits'][:top_k_result]
                _r_mans = []
                for r in r_mans:
                    i = {'library_key': r['_source']['library_key'], 'score': r['_score']}
                    _r_mans.append(i)
                r_mans = _r_mans


                r2_result.append({**item_r2,
                                retrieval_entry: [x['library_key'] for x in r_mans],
                                retrieval_entry.replace(".retrieved", ".score"): [x.score if not isinstance(x, dict) else x['score'] for x in r_mans]})

            except Exception as e:
                print(repr(e))
                r2_result.append({**item_r2,
                                retrieval_entry: [],
                                retrieval_entry.replace(".retrieved", ".score"): []})


    print(f"size of the results: {len(r2_result)}")
    with open(saved_file, "w+") as f:
        json.dump(r2_result, f, indent=2)


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--retrieval_stage', type=int, choices=(0, 1, 2),
                        help='which retrieval stage to run for tldr'
                        'stage 0: build retrieval index'
                        'stage 1: stage 1 retrieval that retrieves the bash command'
                        'stage 2: stage 2 retrieval that retrieves the paragraphs')
    parser.add_argument('--split', type=str,
                        choices=('cmd_train', 'cmd_dev', 'cmd_test'),
                        default='cmd_dev',
                        help='which data split to run')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = config()
    stage = args.retrieval_stage
    split = args.split
    if stage == 0: # build the index
        index = "bash_man_whole"
        source = "chunk"
        _ = ESSearch(index, source, re_index=True, manual_path='data/tldr/manual_all_raw.json',
                           func_descpt_path=None)

        index = "bash_man_para"
        source = "chunk"
        indexer = ESSearch(index, source, re_index=True, manual_path='data/tldr/manual_section.json',
                            func_descpt_path=None)

    if stage == 1:
        index = 'bash_man_whole' # in the first stage, use the whole bash manual to retrieve the bash commands
        source = "chunk"
        search_config_1 = {'query': 'nl', 'field': 'manual', 'filter_result': False}

        print(split, index)
        data_file = f"./data/tldr/{split}.seed.json"
        save_file = data_file.replace(".seed.json", f".full.json")
        query_type, search_field = search_config_1['query'], search_config_1['field']
        tag = f"{index}.{query_type}.{search_field}"
        real_save_file = save_file.replace(".json", f".{tag}.json")

        if not os.path.exists(real_save_file):
            _ = retrieve_manual(data_file, source, index, search_config_1, save_file, top_k=35)

        with open(real_save_file, 'r') as f:
            d = json.load(f)

        src = []
        pred = []
        for item in d:
            src.append(item['cmd_name'])
            pred.append(item[f'{tag}.retrieved'])

        calc_hit(src, pred, top_k=[1, 3, 5, 10, 15, 20, 30])

    if stage == 2:
        source = "chunk"
        r1_index = 'bash_man_whole'
        r2_index = 'bash_man_para' # in the second stage, use paragraphs to retrieve descriptions of relevant arguments etc
        search_config_1 = {'query': 'nl', 'field': 'manual', 'filter_result': False}

        data_file = f"./data/tldr/{split}.seed.json"
        query_type, search_field = search_config_1['query'], search_config_1['field']
        r1_tag = f"{r1_index}.{query_type}.{search_field}"
        r1_save_file = data_file.replace(".seed.json", f".full.{r1_tag}.json")

        r2_save_file = r1_save_file.replace(".json", f".r2-{r2_index}.json")

        if not os.path.exists(r2_save_file):
            _ = doc_base_retrieval(r2_index, source,
                                   r1_save_file,
                                   f'{r1_tag}.retrieved',
                                   r2_save_file,
                                   top_k_doc=5,
                                   top_k_result=30,
                                   oracle_only=False)

