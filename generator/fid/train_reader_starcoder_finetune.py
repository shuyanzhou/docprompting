from datetime import datetime
import os
import sys

import torch
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
train_dataset = load_dataset('json', data_files='/home/ubuntu/docprompting/data/conala/train_llama_small.json', split='train')
eval_dataset = load_dataset('json', data_files='/home/ubuntu/docprompting/data/conala/dev_llama_small.json', split='train')

base_model = "bigcode/starcoder"
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    load_in_8bit=True,
    torch_dtype=torch.float16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("bigcode/starcoder")
eval_prompt = """Give answers in Python.

### Context:
pandas.core.groupby.DataFrameGroupBy.count   DataFrameGroupBy.count()[source]\n \nCompute count of group, excluding missing values.  Returns \n Series or DataFrame\n\nCount of values within each group.      See also  Series.groupby\n\nApply a function groupby to a Series.  DataFrame.groupby\n\nApply a function groupby to each row or column of a DataFrame.

### Input:
Get pandas GroupBy object with sum over the rows with same column names within  dataframe `df`
"""
# {'question': 'Name the comptroller for office of prohibition', 'context': 'CREATE TABLE table_22607062_1 (comptroller VARCHAR, ticket___office VARCHAR)', 'answer': 'SELECT comptroller FROM table_22607062_1 WHERE ticket___office = "Prohibition"'}
model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")

model.eval()
with torch.no_grad():
    print(tokenizer.decode(model.generate(**model_input, max_new_tokens=100)[0], skip_special_tokens=True))

tokenizer.add_eos_token = True
tokenizer.pad_token_id = 0
tokenizer.padding_side = "left"

def tokenize(prompt):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=512,
        padding=False,
        return_tensors=None,
    )

    result["labels"] = result["input_ids"].copy()

    return result

def generate_and_tokenize_prompt(data_point):
    full_prompt =f"""Provide answers in Python.

    ### Context:
    {data_point["context"]}

    ### Input:
    {data_point["question"]}

    ### Response:
    {data_point["answer"]}
    """
    return tokenize(full_prompt)

tokenized_train_dataset = train_dataset.map(generate_and_tokenize_prompt)
tokenized_val_dataset = eval_dataset.map(generate_and_tokenize_prompt)

model.train() 
model = prepare_model_for_int8_training(model)

config = LoraConfig(
r=16,
lora_alpha=32,
lora_dropout=0.05,
bias="none",
task_type="CAUSAL_LM",
target_modules=["c_proj", "c_attn", "q_attn"],
)
model = get_peft_model(model, config)

if torch.cuda.device_count() > 1:
    # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
    model.is_parallelizable = True
    model.model_parallel = True

batch_size = 1
per_device_train_batch_size = 1
gradient_accumulation_steps = batch_size // per_device_train_batch_size
output_dir = "sql-starcoder"


training_args = TrainingArguments(
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        # warmup_steps=100,
        # max_steps=400,
        # max_steps=1,
        num_train_epochs=2,
        learning_rate=3e-4,
        fp16=True,
        logging_steps=1,
        optim="adamw_torch",
        evaluation_strategy="steps", # if val_set_size > 0 else "no",
        save_strategy="epoch",
        # eval_steps=1,
        # eval_steps=20,
        # save_steps=20,
        # save_steps=0,
        output_dir=output_dir,
        # save_total_limit=3,
        load_best_model_at_end=False,
        # ddp_find_unused_parameters=False if ddp else None,
        group_by_length=True, # group sequences of roughly the same length together to speed up training
        report_to="wandb", # if use_wandb else "none",
        run_name=f"starcoder-{datetime.now().strftime('%Y-%m-%d-%H-%M')}", # if use_wandb else None,
    )


trainer = Trainer(
    model=model,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    args=training_args,
    data_collator=DataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    ),
)


model.config.use_cache = False

old_state_dict = model.state_dict
model.state_dict = (lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())).__get__(
    model, type(model)
)

trainer.train()



import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer

base_model = "bigcode/starcoder"
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    load_in_8bit=True,
    torch_dtype=torch.float16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("bigcode/starcoder")

print("final op after fine tune")

output_dir="sql-starcoder/checkpoint-3"

from peft import PeftModel
model = PeftModel.from_pretrained(model, output_dir)

eval_prompt = """Provide answers in Python.

### Context:
items()  \nReturn a new view of the dictionary\u2019s items ((key, value) pairs). See the documentation of view objects.

### Input:
divide the values with same keys of two dictionary `d1` and `d2`

### Response:
"""

model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")

model.eval()
with torch.no_grad():
    print(tokenizer.decode(model.generate(**model_input, max_new_tokens=100)[0], skip_special_tokens=True))






