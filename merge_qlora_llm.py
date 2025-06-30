from transformers import BitsAndBytesConfig
from peft import AutoPeftModelForCausalLM
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",       # or "fp4"
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16 # or torch.float16
)

model = AutoPeftModelForCausalLM.from_pretrained(
    "qlora-output/checkpoint-15",
    device_map="auto",
    torch_dtype=torch.float16,
    quantization_config=bnb_config,
)

model.save_pretrained("qlora-merged-temp", safe_serialization=True)
