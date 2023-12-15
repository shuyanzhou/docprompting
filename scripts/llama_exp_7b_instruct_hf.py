from transformers import AutoTokenizer
import transformers
import torch

# tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-Python-hf")
tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-Instruct-hf")
model = "codellama/CodeLlama-7b-Instruct-hf"

pipeline = transformers.pipeline(
     "text-generation",
     model=model,
     torch_dtype=torch.float16,
     device_map="auto",
)

system = "Provide answers in Python."
user = "get logical xor of a and b"

prompt = f"<s><<SYS>>\n{system}\n<</SYS>>\n\n{user}"

sequences = pipeline(
     prompt,
     do_sample=True,
     top_k=10,
     temperature=0.1,
     top_p=0.95,
     num_return_sequences=1,
     eos_token_id=tokenizer.eos_token_id,
     max_length=200
)

for seq in sequences:
    print(f"Result: {seq['generated_text']}")