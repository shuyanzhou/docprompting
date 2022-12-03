import json
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from tqdm import tqdm
import logging

logging.getLogger('elasticsearch').setLevel(logging.ERROR)

class ESSearch:
    def __init__(self, index: str, source: str,
                 host_address: str='localhost',
                 re_index=False,
                 manual_path=None,
                 func_descpt_path=None):
        self.es = Elasticsearch(timeout=60, host=host_address)
        self.source = source
        self.index = f"{index}.{source}"
        self.manual_path = manual_path
        self.func_descpt_path = func_descpt_path

        if re_index:
            self.es.indices.delete(index=self.index, ignore=[400, 404])
            print(f"delete {self.index}")
            self.es.indices.create(index=self.index)
            # print(self.es.indices.get_alias().keys())
            self.create_index()

        print(f"done init the index {self.index}")
        self.es.indices.refresh(self.index)
        print(self.es.cat.count(self.index, params={"format": "json"}))

    def gendata(self):
        descpt_d = None
        if self.func_descpt_path:
            with open(self.func_descpt_path, "r") as f:
                descpt_d = json.load(f)

        with open(self.manual_path, "r") as f:
            man_d = json.load(f)

        for lib_key, lib_man in tqdm(man_d.items()):
            cmd_name = '_'.join(lib_key.split("_")[:-1]) if lib_key[-1].isdigit() else lib_key
            descpt = descpt_d[lib_key] if descpt_d is not None else ""
            result = {
                '_index': self.index,
                '_type': "_doc",
                'manual': lib_man,
                'func_description': descpt,
                'library_key': lib_key,
                'cmd_name': cmd_name
            }
            yield result

    def create_index(self):
        all_docs = list(self.gendata())
        print(bulk(self.es, all_docs, index=self.index))


    def get_topk(self, search_field, query, topk):
        real_query = {'query': {'match': {search_field: query}},
                      'size': topk + 10}
        r_mans = self.es.search(index=self.index, body=real_query)['hits']['hits'][:topk]
        _r_mans = []
        for r in r_mans:
            i = {'library_key': r['_source']['library_key'], 'score': r['_score']}
            _r_mans.append(i)
        r_mans = _r_mans
        return r_mans


