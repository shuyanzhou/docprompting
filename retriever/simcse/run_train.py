import logging
import os
import sys
from dataclasses import dataclass, field
from collections import defaultdict
from datasets import load_dataset
import wandb
import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
)
from transformers.trainer_utils import is_main_process
from model import RetrievalModel
from trainers import CLTrainer
from data_utils import OurDataCollatorWithPadding, tok_sentences
from arguments import ModelArguments, DataTrainingArguments, OurTrainingArguments, RetrieverArguments
logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, OurTrainingArguments, RetrieverArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, bertscore_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, bertscore_args = parser.parse_args_into_dataclasses()

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty."
            "Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
    )

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)
    training_args.eval_file = data_args.eval_file

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column. You can easily tweak this
    # behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.

    assert 'json' in data_args.train_file
    data_files = {'train': data_args.train_file}
    datasets = load_dataset('json', data_files=data_files)
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    # tokenizer_kwargs = {
    #     "cache_dir": model_args.cache_dir,
    #     "use_fast": model_args.use_fast_tokenizer,
    #     "revision": model_args.model_revision,
    #     "use_auth_token": True if model_args.use_auth_token else None,
    # }
    assert model_args.model_name_or_path
    if 'codet5' in model_args.model_name_or_path:
        tokenizer = transformers.RobertaTokenizerFast.from_pretrained(model_args.model_name_or_path, add_prefix_space=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    assert model_args.model_name_or_path
    # assert training_args.sim_func == 'bertscore'
    model = RetrievalModel(
                       config=config,
                       model_type=model_args.model_name_or_path,
                       num_layers=bertscore_args.num_layers,
                       all_layers=bertscore_args.all_layers,
                       idf=bertscore_args.idf,
                       rescale_with_baseline=bertscore_args.rescale_with_baseline,
                       baseline_path=bertscore_args.baseline_path,
                       tokenizer=tokenizer,
                       training_args = training_args,
                       model_args=model_args)


    # load idf dict
    if bertscore_args.idf:
        raise NotImplementedError
        # assert _idf_dict, "IDF weights are not computed"
        # idf_dict = _idf_dict
    else:
        idf_dict = defaultdict(lambda: 1.0)
        idf_dict[tokenizer.sep_token_id] = 0
        idf_dict[tokenizer.cls_token_id] = 0

    def prepare_features(examples):
        total = len(examples['text1'])
        for idx in range(total):
            if examples['text1'][idx] == '':
                examples['text1'][idx] = " "
            if examples['text2'][idx] == '':
                examples['text2'][idx] = " "

        sentences = examples['text1'] + examples['text2']
        features = tok_sentences(tokenizer, sentences, has_hard_neg=False, total=total, max_length=data_args.max_seq_length)
        return features


    if training_args.do_train:
        train_dataset = datasets['train'].map(
            prepare_features,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    # Data collator
    data_collator = OurDataCollatorWithPadding(tokenizer.pad_token_id, idf_dict)

    training_args.remove_unused_columns = False
    trainer = CLTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    trainer.model_args = model_args
    trainer.epoch_metric = {}
    trainer.metric_for_best_model = training_args.metric_for_best_model
    training_args.do_eval = False

    # Training
    if training_args.do_train:
        model_path = (
            model_args.model_name_or_path
            if (model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path))
            else None
        )

        trainer.train(model_path=model_path)

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        results = trainer.evaluate(eval_senteval_transfer=True)

        output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
        if trainer.is_world_process_zero():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key, value in sorted(results.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
