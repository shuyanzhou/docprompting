import logging
from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict, Tuple
import torch
from transformers import (
    MODEL_FOR_MASKED_LM_MAPPING,
    TrainingArguments,
)
from transformers.file_utils import cached_property, torch_required, is_torch_available, is_torch_tpu_available

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    # Huggingface's original arguments
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
                    "Don't set if you want to train a model from scratch."
        },
    )

    mlp_weight_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "mlp weight path"
        },
    )

    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                    "with private models)."
        },
    )

    pooler_type: str = field(
        default="cls",
        metadata={
            "help": "What kind of pooler to use (cls, cls_before_pooler, avg, avg_top2, avg_first_last)."
        }
    )

    mlp_only_train: bool = field(
        default=False,
        metadata={
            "help": "Use MLP only during training"
        }
    )

    sim_func: str = field(
        default='cls_distance',
        metadata={"help": "the similarity function",
                  "choices": ['cls_distance.cosine', 'cls_distance.l2', 'bertscore']}
    )


    bert_score_loss: str = field(
        default='softmax',
        metadata={'help': 'loss function for bertscore sim function',
                  'choices': ['softmax', 'hinge']}
    )

    hinge_margin: float = field(
        default=1.0
    )

    hard_negative_weight: float = field(
        default=0,
        metadata={
            "help": "The **logit** of weight for hard negatives (only effective if hard negatives are used)."
        }
    )

    # SimCSE's arguments
    temp: float = field(
        default=0.05,
        metadata={
            "help": "Temperature for softmax."
        }
    )

    do_mlm: bool = field(
        default=False,
        metadata={
            "help": "Whether to use MLM auxiliary objective."
        }
    )

    mlm_weight: float = field(
        default=0.1,
        metadata={
            "help": "Weight for MLM auxiliary objective (only effective if --do_mlm)."
        }
    )

    def __post_init__(self):
        if self.sim_func == 'cls_distance.l2':
            self.temp = 1



@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    # Huggingface's original arguments.
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )

    # SimCSE's arguments
    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "The training data file (.txt or .csv)."}
    )

    eval_file: Optional[str] = field(
        default=None,
        metadata={"help": "The eval data file (.txt or .csv)."}
    )


    max_seq_length: Optional[int] = field(
        default=32,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
                    "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    mlm_probability: float = field(
        default=0.15,
        metadata={"help": "Ratio of tokens to mask for MLM (only effective if --do_mlm)"}
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."

x = [1, 2, 3]
from matplotlib.pyplot import plot
plot(x, "go", label="temperature")

@dataclass
class OurTrainingArguments(TrainingArguments):
    # Evaluation
    ## By default, we evaluate STS (dev) during training (for selecting best checkpoints) and evaluate
    ## both STS and transfer tasks (dev) at the end of training. Using --eval_transfer will allow evaluating
    ## both STS and transfer tasks (dev) during training.
    eval_transfer: bool = field(
        default=False,
        metadata={"help": "Evaluate transfer task dev sets (in validation)."}
    )

    customized_eval: bool = field(
        default=True,
        metadata={"help": "Evaluate on the original set, if True, evaluate on user's own data"}
    )

    customized_eval_used_split: Optional[str] = field(
        default='dev'
    )

    tmp_tag: Optional[str] = field(
        default='tmp',
        metadata={'help': 'tag to save tmp models in case of overwriting'}
    )

    report_to: Optional[str] = field(
        default='wandb'
    )

    logging_steps: int = field(
        default=1
    )

    logging_dir: Optional[str] = field(
        default='logs'
    )

    disable_tqdm: bool = field(
        default=True
    )

    eval_form: str = field(
        default='reranking',
        metadata={'choices': ['reranking', 'retrieval']}
    )

    eval_retriever: str = field(
        default='t5',
        metadata={'choices': ['mlm', 't5']},
    )

    eval_src_file: str = field(
        default='conala_nl.txt'
    )

    eval_tgt_file: str = field(
        default='python_manual_firstpara.tok.txt'
    )

    eval_root_folder: str = field(
        default='data/conala',
        metadata={'help': 'root folder of validation dataset'}
    )


    eval_oracle_file: str = field(
        default='cmd_dev.oracle_man.full.json'
    )

    # eval_max_length: int = field(
    #     default=None,
    #     metadata={'help': 'the length for dev set, None will call the max length of the tokenizer'}
    # )


    @cached_property
    @torch_required
    def _setup_devices(self) -> "torch.device":
        logger.info("PyTorch: setting up devices")
        if self.no_cuda:
            device = torch.device("cpu")
            self._n_gpu = 0
        elif is_torch_tpu_available():
            device = xm.xla_device()
            self._n_gpu = 0
        elif self.local_rank == -1:
            # if n_gpu is > 1 we'll use nn.DataParallel.
            # If you only want to use a specific subset of GPUs use `CUDA_VISIBLE_DEVICES=0`
            # Explicitly set CUDA to the first (index 0) CUDA device, otherwise `set_device` will
            # trigger an error that a device index is missing. Index 0 takes into account the
            # GPUs available in the environment, so `CUDA_VISIBLE_DEVICES=1,2` with `cuda:0`
            # will use the first GPU in that env, i.e. GPU#1
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            # Sometimes the line in the postinit has not been run before we end up here, so just checking we're not at
            # the default value.
            self._n_gpu = torch.cuda.device_count()
        else:
            # Here, we'll use torch.distributed.
            # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
            #
            # deepspeed performs its own DDP internally, and requires the program to be started with:
            # deepspeed  ./program.py
            # rather than:
            # python -m torch.distributed.launch --nproc_per_node=2 ./program.py
            if self.deepspeed:
                from .integrations import is_deepspeed_available

                if not is_deepspeed_available():
                    raise ImportError("--deepspeed requires deepspeed: `pip install deepspeed`.")
                import deepspeed

                deepspeed.init_distributed()
            else:
                torch.distributed.init_process_group(backend="nccl")
            device = torch.device("cuda", self.local_rank)
            self._n_gpu = 1

        if device.type == "cuda":
            torch.cuda.set_device(device)

        return device

@dataclass
class RetrieverArguments:
    """
    model_type=model_args.model_name_or_path,
    num_layers=bertscore_args.bertscore_layer_num,
    all_layers=bertscore_args.all_layers,
    idf = bertscore_args.idf,
    idf_sents= bertscore_args.idf_sents,
    rescale_with_baseline=bertscore_args.rescale_with_baseline,
    baseline_path=bertscore_args.baseline_path
    """
    num_layers: int = field(
        default=11
    )
    all_layers: bool = field(
        default=False
    )
    idf: bool = field(
        default=False
    )
    rescale_with_baseline: bool = field(
        default=False
    )
    baseline_path: str = field(
        default=None
    )