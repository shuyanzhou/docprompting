#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os.path
import socket
from collections import defaultdict
from typing import Dict, List
import re
import pickle
from utils.constants import VAR_STR

def init_logger(args):
    # setup logger
    logger = logging.getLogger()
    handler = logging.StreamHandler()
    formatter = logging.Formatter(f"%(asctime)s %(module)s - %(funcName)s: [ {socket.gethostname()} | Node {args.node_id} | Rank {args.global_rank} ] %(message)s",
                                  datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    return logger

def dedup_results(saved):
    exc_key = ['nl', 'gold', 'cmd_name']
    for item in saved:
        if 'prediction' in item:
            item['code'] = item['prediction']
            item.pop('prediction')
        if 'score' in item:
            item['sequence_ll'] = item['score']
            item.pop('score')

    _saved = [saved[0]]
    for item in saved[1:]:
        if item['nl'] == _saved[-1]['nl']:
            for k in item.keys():
                if k not in exc_key:
                    _saved[-1][k] += item[k]
        else:
            _saved.append(item)
    _saved = {x['nl']: x for x in _saved}
    return _saved

def clean_command(s):
    s = s.replace("sudo", "").strip()
    s =  s.replace("`", "").replace('"', "").replace("'", "")
    #  '>', '|', '+'
    s = s.replace("|", " ").replace(">", " ").replace("<", " ")
    s = " ".join(s.split())
    return s

def anonymize_command(s):
    s = s.replace("={", " {")
    var_to_pc_holder = defaultdict(lambda: len(var_to_pc_holder))
    for var in re.findall("{{(.*?)}}", s):
        _ = var_to_pc_holder[var]
    for var, id in var_to_pc_holder.items():
        var_str = "{{%s}}" % var
        s = s.replace(var_str, f"{VAR_STR}_{id}")
    # s = re.sub("{{.*?}}", VAR_STR, s)
    return s

def get_bag_of_keywords(cmd):
    cmd = clean_anonymize_command(cmd)
    # try:
    #     tokens = list(bashlex.split(cmd))
    # except NotImplementedError:
    #     tokens = cmd.strip().split()
    tokens = cmd.strip().split()
    tokens = [x for x in tokens if VAR_STR not in x]
    return tokens

def get_bag_of_words(cmd):
    cmd = clean_anonymize_command(cmd)
    # try:
    #     tokens = list(bashlex.split(cmd))
    # except NotImplementedError:
    #     tokens = cmd.strip().split()
    tokens = cmd.strip().split()
    return tokens

def clean_manual(man_string):
    cur_man_line = [x.strip() for x in man_string.strip().split("\n") if len(x.strip().split()) >= 1]
    cur_man_line = [" ".join(x.split()) for x in cur_man_line]
    cur_man_line = " ".join(cur_man_line)
    return cur_man_line

def clean_anonymize_command(s):
    return anonymize_command(clean_command(s))

# used for constraint command_name decoding
class Trie(object):
    def __init__(self, sequences: List[List[int]] = []):
        self.trie_dict = {}
        self.len = 0
        if sequences:
            for sequence in sequences:
                Trie._add_to_trie(sequence, self.trie_dict)
                self.len += 1

        self.append_trie = None
        self.bos_token_id = None

    def append(self, trie, bos_token_id):
        self.append_trie = trie
        self.bos_token_id = bos_token_id

    def add(self, sequence: List[int]):
        Trie._add_to_trie(sequence, self.trie_dict)
        self.len += 1

    def get(self, prefix_sequence: List[int]):
        return Trie._get_from_trie(
            prefix_sequence, self.trie_dict, self.append_trie, self.bos_token_id
        )

    @staticmethod
    def load_from_dict(trie_dict):
        trie = Trie()
        trie.trie_dict = trie_dict
        trie.len = sum(1 for _ in trie)
        return trie

    @staticmethod
    def _add_to_trie(sequence: List[int], trie_dict: Dict):
        if sequence:
            if sequence[0] not in trie_dict:
                trie_dict[sequence[0]] = {}
            Trie._add_to_trie(sequence[1:], trie_dict[sequence[0]])

    @staticmethod
    def _get_from_trie(
        prefix_sequence: List[int],
        trie_dict: Dict,
        append_trie=None,
        bos_token_id: int = None,
    ):
        if len(prefix_sequence) == 0:
            output = list(trie_dict.keys())
            if append_trie and bos_token_id in output:
                output.remove(bos_token_id)
                output += list(append_trie.trie_dict.keys())
            return output
        elif prefix_sequence[0] in trie_dict:
            return Trie._get_from_trie(
                prefix_sequence[1:],
                trie_dict[prefix_sequence[0]],
                append_trie,
                bos_token_id,
            )
        else:
            if append_trie:
                return append_trie.get(prefix_sequence)
            else:
                return []

    def __iter__(self):
        def _traverse(prefix_sequence, trie_dict):
            if trie_dict:
                for next_token in trie_dict:
                    yield from _traverse(
                        prefix_sequence + [next_token], trie_dict[next_token]
                    )
            else:
                yield prefix_sequence

        return _traverse([], self.trie_dict)

    def __len__(self):
        return self.len

    def __getitem__(self, value):
        return self.get(value)

def build_trie():
    from glob import glob
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
    ss = []
    for cmd in glob("./data/tldr/manual_trimmed/*.txt"):
        cmd = os.path.basename(cmd).replace(".txt", "")
        tok_cmd = tokenizer(f" {cmd}")['input_ids']
        ss.append([-1] + tok_cmd + [-2])
    print(f"number of commands: {len(ss)}")
    trie = Trie(ss)

    with open("./data/tldr/nl.cm/cmd_trie.pkl", "wb") as f:
        pickle.dump(trie.trie_dict, f)


def constrain_cmd_name_fn(cmd_trie, tokenizer, batch_idx, prefix_beam):
    sep_token_idx = tokenizer.sep_token_id
    if prefix_beam[-1] == sep_token_idx: # the first token
        next_tok = cmd_trie.get([-1])
    else:
        # get the prefix
        prefix_idx = prefix_beam.index(sep_token_idx)
        prefix = [-1] + prefix_beam[prefix_idx+1:]
        next_tok = cmd_trie.get(prefix)
        # EOS or not a command anymore
        if [-2] in next_tok or next_tok == []:
            next_tok = [x for x in range(len(tokenizer))]

    return next_tok

if __name__ == "__main__":
    cmd = "firejail --net={{eth0}} --ip={{192.168.1.244}} {{/etc/init.d/apache2}} {{start}}"
    print(clean_command(cmd))
    print(anonymize_command(cmd))
    print(anonymize_command(clean_command(cmd)))

    cmd = "toilet {{input_text}} -f {{font_filename}} {{font_filename}}"
    print(clean_command(cmd))
    print(anonymize_command(cmd))
    print(anonymize_command(clean_command(cmd)))
    # print(get_bag_of_keywords(cmd))
    # build_trie()
    # with open("./data/tldr/nl.cm/cmd_trie.pkl", "rb") as f:
    #     d = pickle.load(f)
    #     trie = Trie.load_from_dict(d)
    #     print(len(trie.get([-1])))
    #     print(trie.get([-1, 300]))

    # from transformers import AutoTokenizer
    # tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
    # tokenizer.add_special_tokens({"sep_token": "|nl2code|"})
    # nl = "disable ldap authentication |nl2code|"
    # tok_nl = tokenizer(nl)['input_ids']
    # tt = [335, 499, 2364]
    # while tt:
    #     next_tok = constrain_cmd_name_fn(trie, tokenizer, None, tok_nl)
    #     assert tt[0] in next_tok, (tokenizer.convert_ids_to_tokens(tok_nl), tt[0], tokenizer.convert_ids_to_tokens(next_tok))
    #     tok_nl.append(tt.pop(0))
