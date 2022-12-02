import argparse
import glob
import os
import json
import sys
from pathlib import Path


def convert_data(src_file, manual_file, info_file, fid_file, retrieved_manual_list, topk=100, sort_ctx=False):
    with open(manual_file, 'r') as f:
        manual_d = json.load(f)
        for k in list(manual_d.keys()):
            manual_d[f'|{"_".join(k.split("_")[:-1])}|'] = "_".join(k.split("_")[:-1])
        manual_d['|placeholder|'] = 'manual'

    if info_file:
        with open(info_file, 'r') as f:
            manual_info = json.load(f)
            manual_info['|placeholder|'] = {'lib_signature': 'manual'}
    else:
        manual_info = None


    with open(src_file, 'r') as f:
        src_d = json.load(f)

    tgt_d = []
    tot = 0
    for src_item in src_d:
        tgt_item = {}
        tgt_item['id'] = src_item['question_id']
        tgt_item['question'] = src_item['nl']
        tgt_item['target'] = src_item['cmd']
        tgt_item['answers'] = [src_item['cmd']]
        ctxs = []
        _ctxs = set()
        for cur_retrieved in retrieved_manual_list:
            for man in cur_retrieved[src_item['question_id']]['retrieved']:
                if manual_info:
                    title = manual_info[man]['lib_signature'].replace(".", " ")
                else:
                    title = man if man != '|placeholder|' else 'manual'
                if man not in manual_d:
                    text = ""
                    print(f"[WARNING] {man} cannot be found")
                else:
                    text = manual_d[man]
                cur_ctx = {'title': title, 'text': text, 'man_id': man}
                if man not in _ctxs:
                    ctxs.append(cur_ctx)
                    _ctxs.add(man)

        tgt_item['ctxs'] = ctxs[:topk]
        if len(tgt_item['ctxs']) < topk:
            for _ in range(topk - len(tgt_item['ctxs'])):
                tgt_item['ctxs'].append({'title': '', 'text': '', 'man_id': 'fake'})
        if sort_ctx:
            tgt_item['ctxs'] = sorted(tgt_item['ctxs'], key=lambda x: int(x['man_id'].split("_")[-1]) if x['man_id'][-1].isdigit() else 10000)
        tot += len(tgt_item['ctxs'])
        tgt_d.append(tgt_item)


    with open(fid_file, 'w+') as f:
        json.dump(tgt_d, f, indent=2)

    # with open(str(fid_file).replace(".json", ".10.json"), 'w+') as f:
    #     json.dump(tgt_d[:10], f, indent=2)
    print(f"save {len(tgt_d)} data to {os.path.basename(fid_file)}")


def process_manual_list(manual_list):
    _manual_list = []
    for l in manual_list:
        keys = list(l[0].keys())
        r_key = [x for x in keys if 'retrieved' in x][0]
        # s_key = [x for x in keys if 'score' in x][0]
        for item in l:
            item['retrieved'] = item.pop(r_key)
        l = {x['question_id']: x for x in l}
        _manual_list.append(l)
    return _manual_list

def run_conala(args):
    root = Path('data/conala/nl.cm')
    all_splits = []
    if args.have_train:
        all_splits.append('cmd_train')
    if args.have_dev:
        all_splits.append('cmd_dev')
    if args.have_test:
        all_splits.append('cmd_test')

    # # fake
    if args.gen_fake:
        for s in all_splits:
            src_file = root / f'{s}.seed.json'
            manual_file = root / 'manual_all_raw.json'
            info_file = root / 'manual.info.json'
            retrieved_manual_list = []
            with open(root / f'{s}.oracle_manual.es0.code.library.full.json', 'r') as f:
                d = json.load(f)
                d = {x['question_id']: {'retrieved': ['|placeholder|']} for x in d}
                retrieved_manual_list.append(d)
            fid_file = root / f'fid.{s}.nothing.json'
            convert_data(src_file, manual_file, info_file, fid_file, retrieved_manual_list, topk=1)

    if args.gen_mine_fake:
        s = 'cmd_mined'
        src_file = root / f'{s}.seed.json'
        manual_file = root / 'manual_all_raw.json'
        info_file = root / 'manual.info.json'
        retrieved_manual_list = []
        with open(root / f'{s}.oracle_manual.es0.code.library.full.json', 'r') as f:
            d = json.load(f)
            d = {x['question_id']: {'retrieved': ['|placeholder|']} for x in d}
            retrieved_manual_list.append(d)
        fid_file = root / f'fid.{s}.nothing.json'
        convert_data(src_file, manual_file, info_file, fid_file, retrieved_manual_list, topk=1)

        with open(fid_file, 'r') as f:
            d = json.load(f)
        _d = []
        for item in d:
            if '_' in item['id']:
                if len(item['question'].split()) >= 100 or \
                        len(item['target'].split()) >= 100 or \
                        len(item['question']) >= 500 or \
                        len(item['target']) >= 500:
                    continue

                _d.append(item)

        print(len(d), len(_d))
        with open(fid_file, 'w+') as f:
            json.dump(_d, f, indent=2)

    if args.gen_oracle:
        for s in all_splits:
            src_file = root / f'{s}.seed.json'
            manual_file = root / 'manual_all_raw.json'
            info_file = root / 'manual.info.json'
            retrieved_manual_list = []
            with open(root / f'{s}.oracle_manual.es0.code.library.full.json', 'r') as f:
                retrieved_manual_list.append(json.load(f))
            fid_file = root / f'fid.{s}.oracle.json'
            retrieved_manual_list = process_manual_list(retrieved_manual_list)
            convert_data(src_file, manual_file, info_file, fid_file, retrieved_manual_list, topk=3)

    if args.gen_retrieval:
        for s in all_splits:
            for topk in [15, 20, 25, 30][:]:
                src_file = root / f'{s}.seed.json'
                manual_file = root / 'manual_all_raw.json'
                info_file = root / 'manual.info.json'
                retrieved_manual_list = []
                with open(root / f'{s}.{args.retrieval_file_tag}.json', 'r') as f:
                    retrieved_manual_list.append(json.load(f))
                fid_file = root / f'fid.{s}.{args.retrieval_file_tag}.t{topk}.json'
                retrieved_manual_list = process_manual_list(retrieved_manual_list)
                convert_data(src_file, manual_file, info_file, fid_file, retrieved_manual_list, topk=topk)

    if args.gen_oracle_retrieval:
        for s in all_splits:
            for topk in [1, 3, 5, 10][:]:
                src_file = root / f'{s}.seed.json'
                manual_file = root / 'manual_all_raw.json'
                info_file = root / 'manual.info.json'
                retrieved_manual_list = []
                with open(root / f'{s}.oracle_manual.es0.code.library.full.json', 'r') as f:
                    retrieved_manual_list.append(json.load(f))
                with open(root / f'{s}.{args.retrieval_file_tag}.json', 'r') as f:
                    retrieved_manual_list.append(json.load(f))
                fid_file = root / f'fid.{s}.oracle.{args.retrieval_file_tag}.t{topk}.json'
                retrieved_manual_list = process_manual_list(retrieved_manual_list)
                convert_data(src_file, manual_file, info_file, fid_file, retrieved_manual_list, topk=topk)

def run_tldr(args):
    root = Path('data/tldr/nl.cm')
    all_splits = []
    if args.have_train:
        all_splits.append('cmd_train')
    if args.have_dev:
        all_splits.append('cmd_dev')
    if args.have_test:
        all_splits.append('cmd_test')

    manual_file = root / 'manual_section.json'
    # # fake
    if args.gen_fake:
        for s in all_splits:
            src_file = root / f'{s}.seed.json'
            retrieved_manual_list = []
            with open(root / f'{s}.oracle_manual.es1.full.oracle.json', 'r') as f:
                d = json.load(f)
                d = {x['question_id']: {'retrieved': ['|placeholder|']} for x in d}
                retrieved_manual_list.append(d)
            fid_file = root / f'fid.{s}.nothing.json'
            convert_data(src_file, manual_file, None, fid_file, retrieved_manual_list, topk=1, sort_ctx=True)

    if args.gen_oracle:
        for s in all_splits:
            src_file = root / f'{s}.seed.json'
            retrieved_manual_list = []
            with open(root / f'{s}.oracle_manual.es1.full.oracle.json', 'r') as f:
                retrieved_manual_list.append(json.load(f))
            fid_file = root / f'fid.{s}.oracle.json'
            retrieved_manual_list = process_manual_list(retrieved_manual_list)
            convert_data(src_file, manual_file, None, fid_file, retrieved_manual_list, topk=10, sort_ctx=True)

    if args.gen_oracle_cmd:
        assert 'tldr' in str(root)
        for s in all_splits:
            src_file = root / f'{s}.seed.json'
            retrieved_manual_list = []
            with open(root / f'{s}.oracle_manual.es1.full.oracle.json', 'r') as f:
                d = json.load(f)
                d = {x['question_id']: {'retrieved': [f'|{x["cmd_name"]}|']} for x in d}
                retrieved_manual_list.append(d)
            fid_file = root / f'fid.{s}.oracle_cmd.json'
            convert_data(src_file, manual_file, None, fid_file, retrieved_manual_list, topk=1, sort_ctx=True)

    if args.gen_retrieval:
        for s in all_splits:
            for topk in [5][:]:
                for x in range(30):
                    if not (root / f'{s}.{args.retrieval_file_tag}.{x}.json').exists():
                        break
                    src_file = root / f'{s}.seed.json'
                    retrieved_manual_list = []
                    with open(root / f'{s}.{args.retrieval_file_tag}.{x}.json', 'r') as f:
                        retrieved_manual_list.append(json.load(f))
                    fid_file = root / f'fid.{s}.{args.retrieval_file_tag}.t{topk}.{x}.json'
                    retrieved_manual_list = process_manual_list(retrieved_manual_list)
                    convert_data(src_file, manual_file, None, fid_file, retrieved_manual_list, topk=topk, sort_ctx=True)

                # merge data
                all_data = []
                for x in range(30):
                    fid_file = root / f'fid.{s}.{args.retrieval_file_tag}.t{topk}.{x}.json'
                    if not fid_file.exists():
                        break
                    with open(fid_file, 'r') as f:
                        curr_data = json.load(f)
                        all_data.extend(curr_data)
                    # os.remove(fid_file)

                print(f"merged: {len(all_data)}")
                with open(root / f'fid.{s}.{args.retrieval_file_tag}.t{topk}.json', 'w') as f:
                    json.dump(all_data, f, indent=2)


    if args.gen_oracle_retrieval:
        for s in all_splits:
            for topk in [10, 15][:]:
                src_file = root / f'{s}.seed.json'
                retrieved_manual_list = []
                with open(root / f'{s}.oracle_manual.es1.full.oracle.json', 'r') as f:
                    retrieved_manual_list.append(json.load(f))
                with open(root / f'{s}.{args.retrieval_file_tag}.0.json', 'r') as f:
                    retrieved_manual_list.append(json.load(f))
                fid_file = root / f'fid.{s}.oracle.{args.retrieval_file_tag}.t{topk}.json'
                retrieved_manual_list = process_manual_list(retrieved_manual_list)
                convert_data(src_file, manual_file, None, fid_file, retrieved_manual_list, topk=topk, sort_ctx=True)


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gen_retrieval', action='store_true')
    parser.add_argument('--gen_oracle', action='store_true')
    parser.add_argument('--gen_oracle_cmd', action='store_true')
    parser.add_argument('--gen_fake', action='store_true')
    parser.add_argument('--gen_mine_fake', action='store_true')
    parser.add_argument('--gen_oracle_retrieval', action='store_true')
    parser.add_argument('--retrieval_file_tag')
    parser.add_argument('--have_train', action='store_true')
    parser.add_argument('--have_dev', action='store_true')
    parser.add_argument('--have_test', action='store_true')
    parser.add_argument('--conala', action='store_true')
    parser.add_argument('--tldr', action='store_true')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = config()
    if args.conala:
        run_conala(args)
        data = 'conala'
    elif args.tldr:
        run_tldr(args)
        data = 'tldr'
    else:
        raise NotImplementedError
