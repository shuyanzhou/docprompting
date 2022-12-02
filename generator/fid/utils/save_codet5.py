import json
import shutil
import transformers
assert transformers.__version__ == '4.11.3', (transformers.__version__)

tokenizer = transformers.RobertaTokenizer.from_pretrained('Salesforce/codet5-base')
tokenizer.save_pretrained('data/fid/codet5-base')

with open('data/fid/codet5-base/tokenizer_config.json', "w+") as f:
    json.dump({"model_max_length": 512}, f)

with open('data/fid/codet5-base/special_tokens_map.json', 'r') as f:
    d = json.load(f)
    d['additional_special_tokens'] = [x['content'] for x in d['additional_special_tokens']]
    # add_tokens = d.pop('additional_special_tokens')
    # for item in add_tokens:
    #     d[item['content']] = item

shutil.move('data/fid/codet5-base/special_tokens_map.json', 'data/fid/codet5-base/special_tokens_map.json.bck')
with open('data/fid/codet5-base/special_tokens_map.json', 'w+') as f:
    json.dump(d, f, indent=2)


print('save tokenizer')

t5 = transformers.T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-base')
t5.save_pretrained('data/fid/codet5-base')

print('save model')