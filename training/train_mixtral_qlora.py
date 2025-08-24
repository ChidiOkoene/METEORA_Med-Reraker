"""
train_mistral_qlora.py
LoRA / QLoRA SFT script tuned for a single-GPU local machine (RTX 4090).
Target base model: Mistral-7B (recommended for local fine-tuning).
This script assumes you have a JSONL file with a "text" field (prompt+response),
e.g., rationales_sft.jsonl (lines are JSON objects containing "text").

Install requirements:
  pip install transformers accelerate peft bitsandbytes datasets safetensors

Run (example):
  accelerate launch --num_processes 1 --num_machines 1 train_mistral_qlora.py

Notes:
- If your GPU has limited VRAM (16GB), use gradient_accumulation_steps to increase effective batch size.
- This uses bitsandbytes 4-bit loading (QLoRA). If you encounter issues, remove load_in_4bit flags.
"""

import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

MODEL = "mistralai/mistral-7b-v0.1"  # change if needed; this is Mistral-7B
DATA_PATH = "rationales_sft.jsonl"  # path to your JSONL with 'text' field
OUTPUT_DIR = "./mistral_qlora_out"

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=False)
    # Ensure tokenizer has pad token
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token":"<|pad|>"})
    # Load model in 4-bit (QLoRA) if possible
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    model.resize_token_embeddings(len(tokenizer))

    # Prepare for kbit training and attach LoRA
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj","k_proj","v_proj","o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    print("Model and LoRA prepared. Loading dataset...")
    ds = load_dataset("json", data_files=DATA_PATH, split="train")

    def tokenize_fn(ex):
        return tokenizer(ex["text"], truncation=True, max_length=1024)

    tokenized = ds.map(tokenize_fn, batched=True, remove_columns=ds.column_names)
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=50,
        save_strategy="epoch",
        remove_unused_columns=False,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
    )
    trainer.train()
    # Save only the LoRA adapters (small)
    model.save_pretrained(OUTPUT_DIR)
    print("Done. LoRA adapters saved to", OUTPUT_DIR)

if __name__ == "__main__":
    main()
