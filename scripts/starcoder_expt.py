from transformers import AutoTokenizer
import transformers
import torch
from tqdm import tqdm

# Hugging face repo name
model = "bigcode/starcoder"
tokenizer = AutoTokenizer.from_pretrained(model)

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto" # if you have GPU
)

# system = ""
# user = ""
# prompt = f"<s><<SYS>>\n{system}\n<</SYS>>\n\n{user}"

sequences = pipeline(
    'Provide answers in Python. divide values associated with each key in dictionary `d1` from values associated with the same key in dictionary `d2`. Context: dict items: Return a new view of the dictionary’s items ((key, value) pairs). See the documentation of view objects.',
    do_sample=True,
    top_k=10,
    top_p = 0.9,
    temperature = 0.2,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=1000, # can increase the length of sequence
)
for seq in tqdm(sequences):
    print(f"Result: {seq['generated_text']}")
