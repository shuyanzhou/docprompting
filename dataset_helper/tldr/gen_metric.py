import requests
import re
import json
from collections import defaultdict, Counter
from utils.util import dedup_results, get_bag_of_keywords, clean_anonymize_command, get_bag_of_words
from pathlib import Path
# from external.nl2bash.eval import eval_tools
# from external.nl2bash.encoder_decoder import data_utils
# import bashlex
from glob import glob
import os
from sacrebleu.metrics import BLEU, CHRF, TER
import sacrebleu
import numpy as np
import argparse
import editdistance
DEBUG=True
METRIC_LIST = ['p', 'r', 'f1', 'cmd_acc', 'no_cmd_p', 'no_cmd_r', 'no_cmd_f1', 'template_matching', 'sentence_bleu']
DEBUG_INFO = []

def token_prf(tok_gold, tok_pred, match_blank=False):
    if match_blank and len(tok_gold) == 0: # do not generate anything
        if len(tok_pred) == 0:
            m = {'r': 1, 'p': 1, 'f1': 1}
        else:
            m = {'r': 0, 'p': 0, 'f1': 0}
    else:
        tok_gold_dict = Counter(tok_gold)
        tok_pred_dict = Counter(tok_pred)
        tokens = set([*tok_gold_dict] + [*tok_pred_dict])
        hit = 0
        for token in tokens:
            hit += min(tok_gold_dict.get(token, 0), tok_pred_dict.get(token, 0))
        p = hit / (sum(tok_pred_dict.values()) + 1e-10)
        r = hit / (sum(tok_gold_dict.values()) + 1e-10)
        f1 = 2 * p * r / (p + r + 1e-10)
        m = {'r': r, 'p': p, 'f1': f1}
    return m

def measure_bag_of_word(gold, pred):
    # tok_gold = get_bag_of_keywords(gold)
    # tok_pred = get_bag_of_keywords(pred)
    tok_gold = get_bag_of_words(gold)
    tok_pred = get_bag_of_words(pred)
    m = token_prf(tok_gold, tok_pred) # whole sentence
    gold_cmd = tok_gold[0] if len(tok_gold) else "NONE_GOLD"
    pred_cmd = tok_pred[0] if len(tok_pred) else "NONE_PRED"
    m = {**m, 'cmd_acc': int(gold_cmd == pred_cmd)}

    # without cmd
    no_cmd_m = token_prf(tok_gold[1:], tok_pred[1:], match_blank=True)
    no_cmd_m = {f"no_cmd_{k}": v for k, v in no_cmd_m.items()}
    m = {**m, **no_cmd_m}
    return m


def calc_bleu(gold, pred):
    ag = clean_anonymize_command(gold)
    ap = clean_anonymize_command(pred)
    score = sacrebleu.sentence_bleu(ap, [ag], tokenize='none').score
    # print(ap, "|", ag)
    # assert score == _score, (score, _score)
    return  {'sentence_bleu': score}

def calc_template_matching(gold, pred):
    ag = clean_anonymize_command(gold)
    ap = clean_anonymize_command(pred)
    m = {'template_matching': int(ag == ap)}
    ag = ' '.join(ag.split()[1:])
    ap = ' '.join(ap.split()[1:])
    m['no_cmd_template_matching'] = int(ag == ap)
    return m

def calc_edit_distance(gold, pred):
    ag = clean_anonymize_command(gold)
    ap = clean_anonymize_command(pred)
    ag_toks = ag.split()
    ap_toks = ap.split()
    m = {'edit_distance': editdistance.eval(ag_toks, ap_toks)}
    return m


def evaluate_from_json_simple(json_file, reference_file, top_k=10, score_key='per_word_ll', code_entry='clean_code'):
    json_file = Path(json_file) if not isinstance(json_file, Path) else json_file

    with open(json_file, "r") as f:
        all_pred = json.load(f)
        all_pred = dedup_results(all_pred)

    with open(reference_file, "r") as f:
        gold = json.load(f)


    DEBUG_INFO.append(f"number of predictions/gold: {len(all_pred)}/{len(gold)}")

    interval = [1, 3, 5, 10, 20, 30]
    interval = [x for x in interval if x <= top_k]
    top_k_metric = {x: defaultdict(float) for x in interval}
    tot = len(gold)
    missing = []
    _bleu = []
    for sample in gold:
        nl = sample['nl']
        gold_cmd = sample['cmd']
        if nl not in all_pred:
            missing.append(nl)
            continue
        pred = all_pred[nl]
        # sort
        pred_score = defaultdict(lambda: -1e10)
        for pc, score in zip(pred[code_entry], pred[score_key]):
            pred_score[pc] = max(pred_score[pc], score)
        pred_score = sorted(pred_score.items(), key=lambda x: x[1], reverse=True)
        pred_top_k = pred_score[:top_k]

        bow = [measure_bag_of_word(gold_cmd, x[0]) for x in pred_top_k]
        tm = [calc_template_matching(gold_cmd, x[0]) for x in pred_top_k]
        bleu = [calc_bleu(gold_cmd, x[0]) for x in pred_top_k]
        metric = [{**bow[i], **tm[i], **bleu[i]} for i in range(len(bow))]

        for tk in interval:
            _metric = metric[:tk]
            for k in METRIC_LIST:
                v = max([x[k] for x in _metric])
                top_k_metric[tk][k] += v / tot

    # print("\t".join([f"{x: >10}" for x in [''] + metric_list]))
    for k, v in top_k_metric.items():
        print("\t".join(['  ' for _ in range(4)]), end='\t')
        # print('\t'.join([f"{k: >10}"] + [f"{v[m]:10.3f}" for m in metric_list]))
        print('\t'.join([f"{v[m]:10.3f}" for m in METRIC_LIST]))

    m_file = json_file.with_suffix(".metric")
    with open(m_file, "w+") as f:
        json.dump(top_k_metric, f, indent=2)

    DEBUG_INFO.append(f"missing {len(missing)} samples, evaluate {len(gold) - len(missing)} samples")
    return top_k_metric

# class DummyFlag(object):
#     def __init__(self):
#         self.explain = False
#         self.min_vocab_frequency = 1
#         self.channel = 'token'
#         self.normalized = False
#         self.fill_argument_slots = False
#         self.use_copy = False
#
# def evaluate_from_json(json_file, data_dir, sc_path, tg_path, overwrite=True, top_k=10):
#     score_key = 'per_word_ll' # use which score for ranking results
#
#     json_file = Path(json_file) if not isinstance(json_file, Path) else json_file
#     with open(json_file, "r") as f:
#         d = json.load(f)
#     prediction_path = Path(json_file.with_suffix(".prediction.txt"))
#     FLAGS = DummyFlag()
#     FLAGS.data_dir = data_dir
#
#     # source, target = ('nl', 'cm') if not FLAGS.explain else ('cm', 'nl')
#     dataset = data_utils.read_data(FLAGS, sc_path, tg_path, load_features=False,
#                          use_buckets=False, buckets=None,
#                          add_start_token=True, add_end_token=True)
#
#     # group annotation by nl, multiple references
#     attribute = 'source'
#     grouped_dataset = data_utils.group_parallel_data(dataset, attribute=attribute, group=False)
#     temp_list = [x[0] for x in grouped_dataset]
#     predictions = [[] for _ in grouped_dataset]
#     if overwrite or not prediction_path.exists():
#         for x in d:
#             nl = x['nl']
#             gold = x['gold']
#             cmd_name = x['cmd_name']
#             # if group:
#             #     raise NotImplementedError
#             #     # words, _ = tokenizer.basic_tokenizer(nl)
#             #     # temp = ' '.join(words)
#             # else:
#             temp = f"{cmd_name} ||| {nl}"
#             temp_idx = temp_list.index(temp)
#
#             if len(predictions[temp_idx]) != 0 and attribute != 'source':
#                 raise NotImplementedError('only allow merging nl')
#
#             if isinstance(x['code'], list):
#                 if len(x['code'][0]) >= 10 * len(gold):
#                     x['code'] = ['PLACEHOLDER']
#                     x[score_key] = [0]
#                     print(f"Skip {x['nl']} due to length issue")
#                 assert len(x['code']) == len(x[score_key])
#                 predictions[temp_idx] += list(zip(x['code'], x[score_key]))
#             else:
#                 raise ValueError
#                 # predictions[temp_idx].append(x['prediction'])
#
#         # write the prediction in the same order as grouped_dataset
#         with open(prediction_path, "w+") as f:
#             for p in predictions:
#                 p = sorted(p, key=lambda x: x[1], reverse=True)
#                 p = [x[0] for x in p[:top_k]]
#                 # assert p
#                 f.write(f"{'|||'.join(p)}\n")
#
#
#     metrics = eval_tools.automatic_eval(prediction_path, grouped_dataset, top_k=10, FLAGS=FLAGS, verbose=False)
#     m_file = json_file.with_suffix(".metric")
#     with open(m_file, "w+") as f:
#         json.dump(metrics, f, indent=2)

def tldr_metrics(src_file, pred_file):
    src_list = []
    pred_list = []
    with open(src_file, 'r') as f:
        for line in f:
            src_list.append(line.strip())
    with open(pred_file, 'r') as f:
        for line in f:
            pred_list.append(line.strip())
    assert len(src_list) == len(pred_list)

    metric_list = defaultdict(list)
    for src, pred in zip(src_list, pred_list):
        for k, v in calc_template_matching(src, pred).items():
            metric_list[k].append(v)
        for k, v in measure_bag_of_word(src, pred).items():
            metric_list[k].append(v)
        for k, v in calc_edit_distance(src, pred).items():
            metric_list[k].append(v)

    for k, v in metric_list.items():
        metric_list[k] = np.mean(v)


    def clean_for_bleu(s):
        s = s.replace("sudo", "").strip()
        s = s.replace("`", "").replace('"', "").replace("'", "")
        #  '>', '|', '+'
        s = s.replace("|", " ").replace(">", " ").replace("<", " ")
        s = " ".join(s.split())
        s = s.replace("={", " {")
        var_to_pc_holder = defaultdict(lambda: len(var_to_pc_holder))
        for var in re.findall("{{(.*?)}}", s):
            _ = var_to_pc_holder[var]
        for var, id in var_to_pc_holder.items():
            var_str = "{{%s}}" % var
            s = s.replace(var_str, f"${id}")
        # s = re.sub("{{.*?}}", VAR_STR, s)
        # print(s)
        return s

    # calc bleu
    bleu = BLEU(tokenize='none', smooth_method='add-k', smooth_value=1)
    pred_list = [clean_for_bleu(x) for x in pred_list]
    src_list = [clean_for_bleu(x) for x in src_list]
    bleu_score = bleu.corpus_score(pred_list, [src_list]).score
    metric_list['bleu'] = bleu_score


    def to_characters(s):
        # s = s.replace(" ", "")
        # s = ' '.join(list(s))
        return s
    # character level
    bleu = BLEU(tokenize='char')
    pred_list = [to_characters(x) for x in pred_list]
    src_list = [to_characters(x) for x in src_list]
    bleu_score = bleu.corpus_score(pred_list, [src_list]).score
    metric_list['bleu_char'] = bleu_score
    return metric_list

def add_cmd_name(result_file):
    save_file = result_file.replace(".cn1.", '.cn1.processed.')
    if not os.path.exists(save_file):
        with open(result_file, "r") as f:
            d = json.load(f)
            for item in d:
                cmd_name = item['cmd_name']
                item['clean_code'] = [f"{cmd_name} {x}" for x in item['clean_code']]
        with open(save_file, "w+") as f:
            json.dump(d, f, indent=2)
    return save_file

def get_config(result_file):
    if "1.3" in result_file or "13" in result_file:
        model = "neo-1.3B"
    elif "125" in result_file:
        model = "neo-125M"
    elif "gpt-j" in result_file:
        model = "gpt-j-6B"
    elif "codex" in result_file:
        model = "codeX"
    else:
        raise ValueError(f"model not found in {result_file}")

    if any([x in result_file for x in ['raw_models', 'codex']]):
        training = "P"
    else:
        training = "Y"

    if any([x in result_file for x in ['_MN', 'codex_no_examples_man', '.nl_cmd.']]):
        manual = "N"
    elif any([x in result_file for x in ['_MO', 'codex_examples_man', '.man_nl_cmd.']]):
        manual = "O" #at most 2
    elif "_MS" in result_file:
        manual = "O+S"
    else:
        raise ValueError(f"manual config not found in {result_file}")

    if any([x in result_file for x in ['cn0', '_no_cmdname.']]):
        cmd_name = 'N'
    elif any([x in result_file for x in ['cn1', '_with_cmdname.']]):
        cmd_name = 'Y'
    else:
        raise ValueError(f"command name config not found in {result_file}")

    return model, training, manual, cmd_name

def evaluate_lists_of_prediction():
    parser = argparse.ArgumentParser()
    parser.add_argument('--soft_match', default=None)
    parser.add_argument('--score_key', default=None)
    args = parser.parse_args()

    base_path = Path("./data/tldr")
    data_dir = base_path / "nl.cm"

    # f_list = sorted(glob(args.soft_match))
    # score_key = args.score_key
    # score_key = 'per_word_ll'
    score_key = 'fake_per_word_ll'
    f_list = sorted(
        glob("./data/tldr/models/model_13b_TY_MS_v3/*17*.cmd_dev*trimmed_manual_para_cat_descpt*merge*json"))
    # f_list = sorted(glob("./data/tldr/models/model_13b_TY_MN/*cmd_test*.json"))
    # f_list = sorted(glob("./data/tldr/models/rerank/*cmd_dev*.final.json"))
    # f_list += sorted(glob("./data/tldr/models/raw_models/*.json"))
    # f_list = sorted(glob("./data/tldr/models/codex/decode.codex.*.json"))
    print(f_list)
    f_list = [x for x in f_list if '.processed.' not in x]
    print("\t".join([' ' for _ in range(4)]))
    print("\t".join([f"{x: >10}" for x in
                     ['Model', 'Training (N,Y,P)', 'Man (N, Y, OnlyOracle (O), Oracle+Random(O+S))',
                      'Cmd'] + METRIC_LIST]))
    summary_results = [[], []]
    for result_file in f_list:
        DEBUG_INFO.append(result_file)
        config = get_config(result_file)
        config = '\t'.join(config)
        print(config, "\t", result_file)
        l = "random" if "random" in result_file else "cmd"
        s = "test" if "test" in result_file else 'dev'
        reference_file = data_dir / f"{l}_{s}.seed.json"
        if '.cn1.' in result_file and '.processed.' not in result_file:
            # add command name before eval
            result_file = add_cmd_name(result_file)
        m = evaluate_from_json_simple(result_file, reference_file, top_k=30, score_key=score_key,
                                      code_entry='clean_code')

        summary_results[0].append(
            [config, m[1]['template_matching'], m[1]['f1'], m[1]['no_cmd_f1'], m[1]['sentence_bleu']])
        DEBUG_INFO.append("============================")

    print("*****************Summary******************")
    for r in summary_results:
        for item in r:
            print(item[0], '\t', '\t'.join([str(x) for x in item[1:]]))

    for l in DEBUG_INFO:
        print(l)


def debug_tm_bleu():
    bleu = BLEU()
    with open('data/fid/tldr.nothing/test_results_test_same.json', 'r') as f:
        db = []
        for line in f:
            db.append(json.loads(line))
    with open('data/fid/tldr.oracle.r2-bash_man_para.t10/test_results_test_same.json', 'r') as f:
        do = []
        for line in f:
            do.append(json.loads(line))
    tot = 0
    for ib, io in zip(db, do):
        assert ib['question_id'] == io['question_id']
        gold = clean_anonymize_command(ib['gold'])
        pb = clean_anonymize_command(ib['clean_code'])
        po = clean_anonymize_command(io['clean_code'])
        # if po == gold:
        #     print(po, '|', gold, '|', bleu.corpus_score([po], [[gold]]).score)
        print('\t', gold)
        print(bleu.corpus_score([pb],[[gold]]).score, '\t', pb)
        print(bleu.corpus_score([po], [[gold]]).score, '\t', po)
        print('\n')
    print(tot)
if __name__ == "__main__":
    # debug_tm_bleu()
    # exit(0)
    m_list = ['cmd_acc', 'template_matching', 'f1', 'bleu_char', 'edit_distance']

    # codex
    # with open('data/tldr/nl.cm/code-davinci-002.codex.cmd_dev.json', 'r') as f:
    #     d = json.load(f)
    #     with open('gold.gold', 'w+') as fg, open('pred.pred', 'w+') as fp:
    #         for item in d:
    #             fg.write(item['cmd'].strip().replace("\n", "") + '\n')
    #             fp.write(item['codex_response'].replace("\n", "") + '\n')
    #     m = tldr_metrics('gold.gold', 'pred.pred')
    #     for k, v in m.items():
    #         m[k] = f'{v:.4f}'
    #     print('\t'.join([str(m[x]) for x in m_list]))

    # evaluate_lists_of_prediction()

    # gpt
    # tldr.neo13.train_nlcode
    for file_name in sorted(list(glob('data/gpt/tldr.*/decode*cmd_test*.train_nlcode*json'))):
        with open(file_name, 'r') as f:
            d = json.load(f)
        with open('gold.gold', 'w+') as fg, open('pred.pred', 'w+') as fp:
            for item in d:
                fg.write(item['gold'].strip().replace("\n", "") + '\n')
                fp.write(item['clean_code'][0].strip().replace("\n", "") + '\n')

        m = tldr_metrics('gold.gold', 'pred.pred')
        print(f"{file_name}")
        for k, v in m.items():
            m[k] = f'{v:.4f}'
        print('\t'.join([str(m[x]) for x in m_list]))
    # fid
    for file_name in sorted(list(glob('data/fid/*tldr.*/test_results_test*.json'))):
        with open(file_name, 'r') as f:
            data = []
            for line in f:
                data.append(json.loads(line))
        # ids = set()
        # for item in data:
        #     if item['question_id'] in ids:
        #         print(item['question_id'])
        #     ids.add(item['question_id'])
        # split to top-n
        split_data = [[] for _ in range(10)]
        qid_counter = Counter()
        for item in data:
            if item['question_id'] in ['9931', '7895', '3740', '8077', '4737', '7057', '9530']:
                continue
            split_idx = qid_counter[item['question_id']]
            split_data[split_idx].append(item)
            qid_counter[item['question_id']] += 1
        assert all([len(x) in [918, 1845, 0] for x in split_data])

        split_data = split_data[:1]
        for s_idx, cur_split in enumerate(split_data):
            with open('gold.gold', 'w+') as fg, open('pred.pred', 'w+') as fp:
                for item in cur_split:
                    fg.write(item['gold'].strip().replace("\n", "") + '\n')
                    fp.write(item['clean_code'].replace("\n", "") + '\n')
            m = tldr_metrics('gold.gold', 'pred.pred')
            print(f"{file_name}: {s_idx}")
            # print(json.dumps(m, indent=2))
            for k, v in m.items():
                m[k] = f'{v:.4f}'
            print('\t'.join([str(m[x]) for x in m_list]))
            # ff.write('\t'.join([str(m[x]) for x in m_list]) + '\n')

