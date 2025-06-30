import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig
from trl import SFTTrainer
import argparse


# Parse CLI args
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-Instruct-v0.1")
parser.add_argument("--dataset_path", type=str, default="config_context_dataset.jsonl")
parser.add_argument("--output_dir", type=str, default="./qlora-output")
args = parser.parse_args()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Load and verify dataset
df = pd.read_json(args.dataset_path, lines=True)
assert "messages" in df.columns, "Each row must contain a 'messages' field"

dataset = Dataset.from_pandas(df)

# Flatten 'messages' list to a single string using chat template
dataset = dataset.map(
    lambda x: {"text": tokenizer.apply_chat_template(x["messages"], tokenize=False)},
    remove_columns=["messages"]
)

# Tokenize the rendered chat text
dataset = dataset.map(
    lambda x: tokenizer(x["text"], padding=True, truncation=True, max_length=512),
    batched=True,
    remove_columns=["text"]
)

# Load 4-bit quantized model
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    llm_int8_enable_fp32_cpu_offload=True,
)

model = AutoModelForCausalLM.from_pretrained(
    args.model_name,
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

# model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)

# Optional: check parallelizability
if hasattr(model, "is_parallelizable"):
    print("Model is parallelizable:", model.is_parallelizable)
else:
    print("Model parallelizability unknown")

# Configure LoRA
peft_config = LoraConfig(
    r=64,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, peft_config)

# Training args
training_args = TrainingArguments(
    output_dir=args.output_dir,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=2,
    bf16=False,
    remove_unused_columns=False,
    report_to="none"
)

# SFT Trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=512,
    packing=False
)

trainer.train()
