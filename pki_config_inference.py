from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
import torch
import os
import gc

offload_dir = "offload"
os.makedirs(offload_dir, exist_ok=True)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)

gc.collect()
torch.cuda.empty_cache()

model = AutoModelForCausalLM.from_pretrained(
    "qlora-merged-temp",
    device_map="auto",
    quantization_config=bnb_config,
    offload_folder=offload_dir,
    offload_buffers=True,
)

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")

inputs = tokenizer("What does jobsScheduler.impl.UnpublishExpiredJob.class do in CS.cfg?", return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=150)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
