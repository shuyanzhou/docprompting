import torch
from typing import Dict
from dataclasses import dataclass, field

@dataclass
class OurDataCollatorWithPadding:
    def __init__(self, pad_token_id, idf_dict):
        self.pad_token_id = pad_token_id
        self.idf_dict = idf_dict

    def padding_func(self, arr, pad_token, dtype=torch.long):
        lens = torch.LongTensor([len(a) for a in arr])
        max_len = lens.max().item()
        padded = torch.ones(len(arr), max_len, dtype=dtype) * pad_token
        mask = torch.zeros(len(arr), max_len, dtype=torch.long)
        for i, a in enumerate(arr):
            padded[i, : lens[i]] = torch.tensor(a, dtype=dtype)
            mask[i, : lens[i]] = 1
        return padded, lens, mask

    def negative_sample_mask(self, target_sent, dtype=torch.bool):
        mask = torch.ones((len(target_sent), len(target_sent)), dtype=dtype)
        for i in range(len(target_sent)):
            s1 = target_sent[i]
            for j in range(len(target_sent)):
                s2 = target_sent[j]
                if i != j and s1 == s2:
                    mask[i, j] = 0
        return mask

    def __call__(self, batch) -> Dict[str, torch.Tensor]:
        bs = len(batch)
        assert bs
        num_sent = len(batch[0]['input_ids'])
        pad_token = self.pad_token_id

        flat_input_ids = []
        for sample in batch:
            for i in range(num_sent):
                flat_input_ids.append(sample['input_ids'][i])

        flat_idf_weights = []
        for input_ids in flat_input_ids:
            cur_idf_weights = [self.idf_dict[id] for id in input_ids]
            flat_idf_weights.append(cur_idf_weights)

        padded, lens, mask = self.padding_func(flat_input_ids, pad_token, dtype=torch.long)
        padded_idf, _, _ = self.padding_func(flat_idf_weights, 0, dtype=torch.float)
        assert padded.shape == padded_idf.shape

        target_sent = []
        for sample in batch:
            target_sent.append(sample['plain_text'][1])
        negative_sample_mask = self.negative_sample_mask(target_sent)

        # padded = padded.to(device=device)
        # mask = mask.to(device=device)
        # lens = lens.to(device=device)
        # return padded, padded_idf, lens, mask
        return {'input_ids': padded, 'attention_mask': mask, 'negative_sample_mask': negative_sample_mask,
                'lengths': lens, 'input_idf': padded_idf, 'num_sent': num_sent}

def tok_sentences(tokenizer, sentences, has_hard_neg, total, max_length=None):
    sent_features = tokenizer(
        sentences,
        add_special_tokens=True,
        # add_prefix_space=True,
        max_length=tokenizer.model_max_length if max_length is None else max_length,
        truncation=True
    )

    features = {}
    if has_hard_neg:
        for key in sent_features:
            features[key] = [[sent_features[key][i], sent_features[key][i + total], sent_features[key][i + total * 2]]
                             for i in range(total)]

    else:
        for key in sent_features:
            features[key] = [[sent_features[key][i], sent_features[key][i + total]] for i in range(total)]
        # get the plain text
        features['plain_text'] = []
        for i in range(total):
            features['plain_text'].append([sentences[i], sentences[i + total]])

    return features

