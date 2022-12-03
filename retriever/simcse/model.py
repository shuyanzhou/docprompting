import os
import time
import torch
import pandas as pd
import warnings

from torch import nn
from transformers import PreTrainedModel
from collections import defaultdict
from transformers.modeling_outputs import SequenceClassifierOutput
torch.set_printoptions(profile="full")

from utils import (
    get_model,
    get_tokenizer,
    get_idf_dict,
    bert_cos_score_idf,
    model2layers,
    get_hash,
    greedy_cos_idf_for_train,
    greedy_cos_idf
)


class RetrievalModel(PreTrainedModel):
    """
    Adapt the implementation of BertScore to calculate the similarity between a query and a doc
    with either CLS mean pooling distance or BERTScore F1.
    """

    def __init__(
        self,
            config: object,
            model_type: object = None,
            num_layers: object = None,
            batch_size: object = 64,
            nthreads: object = 4,
            all_layers: object = False,
            idf: object = False,
            idf_sents: object = None,
            device: object = None,
            lang: object = None,
            rescale_with_baseline: object = False,
            baseline_path: object = None,
            use_fast_tokenizer: object = False,
            tokenizer: object = None,
            training_args: object = None,
            model_args: object = None
    ) -> object:
        super().__init__(config)
        """
        Args:
            - :param: `model_type` (str): contexual embedding model specification, default using the suggested
                      model for the target langauge; has to specify at least one of
                      `model_type` or `lang`
            - :param: `num_layers` (int): the layer of representation to use.
                      default using the number of layer tuned on WMT16 correlation data
            - :param: `verbose` (bool): turn on intermediate status update
            - :param: `idf` (bool): a booling to specify whether to use idf or not (this should be True even if `idf_sents` is given)
            - :param: `idf_sents` (List of str): list of sentences used to compute the idf weights
            - :param: `device` (str): on which the contextual embedding model will be allocated on.
                      If this argument is None, the model lives on cuda:0 if cuda is available.
            - :param: `batch_size` (int): bert score processing batch size
            - :param: `nthreads` (int): number of threads
            - :param: `lang` (str): language of the sentences; has to specify
                      at least one of `model_type` or `lang`. `lang` needs to be
                      specified when `rescale_with_baseline` is True.
            - :param: `return_hash` (bool): return hash code of the setting
            - :param: `rescale_with_baseline` (bool): rescale bertscore with pre-computed baseline
            - :param: `baseline_path` (str): customized baseline file
            - :param: `use_fast_tokenizer` (bool): `use_fast` parameter passed to HF tokenizer
        """

        assert lang is not None or model_type is not None, "Either lang or model_type should be specified"
        if rescale_with_baseline:
            assert lang is not None, "Need to specify Language when rescaling with baseline"
        self.training_args = training_args
        self.model_args = model_args
        self._lang = lang
        self._rescale_with_baseline = rescale_with_baseline
        self._idf = idf
        self.batch_size = batch_size
        self.nthreads = nthreads
        self.all_layers = all_layers

        assert model_type is not None
        self._model_type = model_type

        if num_layers is None:
            self._num_layers = model2layers[self.model_type]
        else:
            self._num_layers = num_layers


        self._use_fast_tokenizer = use_fast_tokenizer
        self._tokenizer = get_tokenizer(self.model_type, self._use_fast_tokenizer) if tokenizer is None else tokenizer
        self._model = get_model(self.model_type, self.num_layers, self.all_layers)

        self._idf_dict = None
        if idf_sents is not None:
            self.compute_idf(idf_sents)

        self._baseline_vals = None
        self.baseline_path = baseline_path
        self.use_custom_baseline = self.baseline_path is not None
        if self.baseline_path is None:
            self.baseline_path = os.path.join(
                os.path.dirname(__file__), f"rescale_baseline/{self.lang}/{self.model_type}.tsv"
            )

    @property
    def lang(self):
        return self._lang

    @property
    def idf(self):
        return self._idf

    @property
    def model_type(self):
        return self._model_type

    @property
    def num_layers(self):
        return self._num_layers

    @property
    def rescale_with_baseline(self):
        return self._rescale_with_baseline

    @property
    def baseline_vals(self):
        if self._baseline_vals is None:
            if os.path.isfile(self.baseline_path):
                if not self.all_layers:
                    self._baseline_vals = torch.from_numpy(
                        pd.read_csv(self.baseline_path).iloc[self.num_layers].to_numpy()
                    )[1:].float()
                else:
                    self._baseline_vals = (
                        torch.from_numpy(pd.read_csv(self.baseline_path).to_numpy())[:, 1:].unsqueeze(1).float()
                    )
            else:
                raise ValueError(f"Baseline not Found for {self.model_type} on {self.lang} at {self.baseline_path}")

        return self._baseline_vals

    @property
    def use_fast_tokenizer(self):
        return self._use_fast_tokenizer

    @property
    def hash(self):
        return get_hash(
            self.model_type, self.num_layers, self.idf, self.rescale_with_baseline, self.use_custom_baseline, self.use_fast_tokenizer
        )

    def compute_idf(self, sents):
        """
        Args:

        """
        if self._idf_dict is not None:
            warnings.warn("Overwriting the previous importance weights.")

        self._idf_dict = get_idf_dict(sents, self._tokenizer, nthreads=self.nthreads)

    def get_pooling_embedding(self, input_ids, attention_mask, lengths, pooling="mean", normalize=False):
        out = self._model(input_ids, attention_mask=attention_mask, output_hidden_states=self.all_layers)
        if self.all_layers:
            emb = torch.stack(out[-1], dim=2)
        else:
            emb = out[0]
        if pooling == "mean":
            emb.masked_fill_(~attention_mask.bool().unsqueeze(-1), 2)
            max_len = max(lengths)
            base = torch.arange(max_len, dtype=torch.long).expand(len(lengths), max_len).to(lengths.device)
            pad_mask = base < lengths.unsqueeze(1)
            emb = (emb * pad_mask.unsqueeze(-1)).sum(dim=1) / pad_mask.sum(-1).unsqueeze(-1)
            if normalize:
                emb = emb / emb.norm(dim=1, keepdim=True)
        else:
            raise NotImplementedError
        return emb


    def calc_pair_tok_embedding(self, input_ids, attention_mask, lengths=None, input_idf=None, num_sent=None):

        batch_size = int(input_ids.shape[0] / num_sent)
        # calc embeddings
        out = self._model(input_ids, attention_mask=attention_mask, output_hidden_states=self.all_layers)
        if self.all_layers:
            emb = torch.stack(out[-1], dim=2)
        else:
            emb = out[0]

        def split_to_pair(m):
            dim_size = len(m.shape)
            if dim_size == 1:
                m = m.view(batch_size, num_sent)
            elif dim_size == 2:
                m = m.view(batch_size, num_sent, -1)
            elif dim_size == 3:
                max_len = m.size(-2)
                m = m.view(batch_size, num_sent, max_len, -1)
            else:
                raise ValueError('dimension should be only 2 or 3')
            return m[:, 0], m[:, 1]

        def length_to_mask(lens, max_len):
            base = torch.arange(max_len, dtype=torch.long).expand(len(lens), max_len).to(lens.device)
            return base < lens.unsqueeze(1)

        ref_emb, hyp_emb = split_to_pair(emb)
        ref_att_mask, hyp_attn_mask = split_to_pair(attention_mask)
        ref_len, hyp_len = split_to_pair(lengths)
        ref_idf, hyp_idf = split_to_pair(input_idf)

        # pad to calculate greedy cos idf
        # emb_pad: padding with value 2
        ref_emb.masked_fill_(~ref_att_mask.bool().unsqueeze(-1), 2)
        hyp_emb.masked_fill_(~hyp_attn_mask.bool().unsqueeze(-1), 2)
        # idf_pad: padding with value 0 (already satisfied)
        # pad_mask: length mask
        max_len = max(max(ref_len), max(hyp_len))
        ref_pad_mask = length_to_mask(ref_len, max_len)
        hyp_pad_mask = length_to_mask(hyp_len, max_len)

        return ref_emb, ref_pad_mask, ref_idf, hyp_emb, hyp_pad_mask, hyp_idf

    def forward(self, input_ids=None, attention_mask=None, negative_sample_mask=None,
                lengths=None, input_idf=None,
                num_sent=None, pairwise_similarity=False):
        ref_emb, ref_pad_mask, ref_idf, \
        hyp_emb, hyp_pad_mask, hyp_idf = self.calc_pair_tok_embedding(input_ids, attention_mask,
                                                             lengths, input_idf, num_sent)

        if 'cls_distance' in self.model_args.sim_func:
            # ref_emb [B, max_len, embed_size]
            # ref_pad_mask [B, max_len]
            # m_ref_emb [B, embed_size]
            m_ref_emb = (ref_emb * ref_pad_mask.unsqueeze(-1)).sum(dim=1) / ref_pad_mask.sum(-1).unsqueeze(-1)
            m_hyp_emb = (hyp_emb * hyp_pad_mask.unsqueeze(-1)).sum(dim=1) / hyp_pad_mask.sum(-1).unsqueeze(-1)

            if 'cosine' in self.model_args.sim_func:
                cos_sim = nn.CosineSimilarity(dim=-1)
                sim_score = cos_sim(m_ref_emb.unsqueeze(1), m_hyp_emb.unsqueeze(0))
            elif 'l2' in self.model_args.sim_func:
                sim_score = torch.matmul(m_ref_emb, m_hyp_emb.transpose(0, 1))
            else:
                raise NotImplementedError

            if pairwise_similarity: # pairwise score only
                return torch.diagonal(sim_score, 0)
            else:
                loss_fct = nn.CrossEntropyLoss(reduction='mean')
                labels = torch.arange(sim_score.size(0)).long().to(sim_score.device)
                # mask the conflict negative examples
                sim_score.masked_fill_(~negative_sample_mask, -1e10)
                sim_score = sim_score / self.model_args.temp
                loss = loss_fct(sim_score, labels)
                return SequenceClassifierOutput(loss=loss)

        elif self.model_args.sim_func == 'bertscore':
            if pairwise_similarity:
                _, _, sim_score = greedy_cos_idf(ref_emb, ref_pad_mask, ref_idf,
                                      hyp_emb, hyp_pad_mask, hyp_idf,
                                      self.all_layers)
                return sim_score
            else:
                _, _, sim_score = greedy_cos_idf_for_train(ref_emb, ref_pad_mask, ref_idf,
                                          hyp_emb, hyp_pad_mask, hyp_idf,
                                          self.all_layers)
                labels = torch.arange(sim_score.size(0)).long().to(sim_score.device)
                # print(sim_score)
                if self.model_args.bert_score_loss == 'hinge':
                    loss_fct = nn.MultiMarginLoss(margin=self.model_args.hinge_margin, reduction='none')
                    sim_score.masked_fill_(~negative_sample_mask, -1e10)
                    loss = loss_fct(sim_score, labels)
                    loss = loss * sim_score.shape[1] / negative_sample_mask.long().sum(-1) # recover x.size(0)
                    loss = torch.mean(loss)
                else:
                    loss_fct = nn.CrossEntropyLoss(reduction='mean')
                    sim_score.masked_fill_(~negative_sample_mask, -1e10)
                    sim_score = sim_score / self.model_args.temp
                    loss = loss_fct(sim_score, labels)
                # loss_fct = nn.CrossEntropyLoss()
                return SequenceClassifierOutput(loss = loss)

        else:
            raise NotImplementedError



    def score(self, cands, refs, verbose=False, batch_size=64, return_hash=False):
        """
        Args:
            - :param: `cands` (list of str): candidate sentences
            - :param: `refs` (list of str or list of list of str): reference sentences

        Return:
            - :param: `(P, R, F)`: each is of shape (N); N = number of input
                      candidate reference pairs. if returning hashcode, the
                      output will be ((P, R, F), hashcode). If a candidate have
                      multiple references, the returned score of this candidate is
                      the *best* score among all references.
        """

        ref_group_boundaries = None
        if not isinstance(refs[0], str):
            ref_group_boundaries = []
            ori_cands, ori_refs = cands, refs
            cands, refs = [], []
            count = 0
            for cand, ref_group in zip(ori_cands, ori_refs):
                cands += [cand] * len(ref_group)
                refs += ref_group
                ref_group_boundaries.append((count, count + len(ref_group)))
                count += len(ref_group)

        if verbose:
            print("calculating scores...")
            start = time.perf_counter()

        if self.idf:
            assert self._idf_dict, "IDF weights are not computed"
            idf_dict = self._idf_dict
        else:
            idf_dict = defaultdict(lambda: 1.0)
            idf_dict[self._tokenizer.sep_token_id] = 0
            idf_dict[self._tokenizer.cls_token_id] = 0

        all_preds = bert_cos_score_idf(
            self._model,
            refs,
            cands,
            self._tokenizer,
            idf_dict,
            verbose=verbose,
            device=self.device,
            batch_size=batch_size,
            all_layers=self.all_layers,
        ).cpu()

        if ref_group_boundaries is not None:
            max_preds = []
            for start, end in ref_group_boundaries:
                max_preds.append(all_preds[start:end].max(dim=0)[0])
            all_preds = torch.stack(max_preds, dim=0)

        if self.rescale_with_baseline:
            all_preds = (all_preds - self.baseline_vals) / (1 - self.baseline_vals)

        out = all_preds[..., 0], all_preds[..., 1], all_preds[..., 2]  # P, R, F

        if verbose:
            time_diff = time.perf_counter() - start
            print(f"done in {time_diff:.2f} seconds, {len(refs) / time_diff:.2f} sentences/sec")

        if return_hash:
            out = tuple([out, self.hash])

        return out


    def __repr__(self):
        return f"{self.__class__.__name__}(hash={self.hash}, batch_size={self.batch_size}, nthreads={self.nthreads})"

    def __str__(self):
        return self.__repr__()
