#!/usr/bin/env python3
r"""
training/train_mistral_7b_qlora.py

Downloads Mistral-7B-Instruct-v0.3 model to D: drive and trains LoRA adapters using QLoRA.
All offloading is directed to D: drive to avoid C: drive space issues.
"""

import os, sys, logging, subprocess, math, random
from pathlib import Path

import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from huggingface_hub import login, snapshot_download

# ---------------- UPDATED PATHS FOR MISTRAL-7B v0.3 ----------------
LOCAL_MODEL_PATH = r"D:\models\Mistral-7B-Instruct-v0.3"
TRAIN_FILE = r"C:\projects\meteora_project\data\rationales_sft_train.jsonl"
VAL_FILE   = r"C:\projects\meteora_project\data\rationales_sft_val.jsonl"
OUTPUT_DIR = r"D:\projects\meteora_project\outputs\lora_out_mistral_7b_v3"
OFFLOAD_DIR = r"D:\offload"  # Directory for any offloading operations

# Hyperparameters (adjusted for 7B model)
MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "1024"))
PER_DEVICE_BATCH_SIZE = int(os.environ.get("PER_DEVICE_BATCH_SIZE", "2"))  # Increased for 7B
GRAD_ACCUM_STEPS = int(os.environ.get("GRAD_ACCUM_STEPS", "4"))  # Reduced for 7B
NUM_EPOCHS = int(os.environ.get("NUM_EPOCHS", "3"))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "2e-4"))

LORA_R = int(os.environ.get("LORA_R", "16"))
LORA_ALPHA = int(os.environ.get("LORA_ALPHA", "32"))
LORA_DROPOUT = float(os.environ.get("LORA_DROPOUT", "0.05"))

SEED = int(os.environ.get("SEED", "42"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("train_mistral_7b_qlora")

def download_model_if_needed():
    """Download the model if it doesn't exist locally"""
    if not os.path.exists(LOCAL_MODEL_PATH):
        logger.info(f"Model not found at {LOCAL_MODEL_PATH}. Downloading...")
        os.makedirs(LOCAL_MODEL_PATH, exist_ok=True)
        
        # Ensure we're logged in to Hugging Face
        try:
            login()
        except Exception as e:
            logger.warning(f"Login failed: {e}")
        
        # Download the model
        try:
            snapshot_download(
                "mistralai/Mistral-7B-Instruct-v0.3",  # Updated to v0.3
                local_dir=LOCAL_MODEL_PATH,
                local_dir_use_symlinks=False,
                resume_download=True
            )
            logger.info("Model downloaded successfully.")
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            raise
    else:
        logger.info(f"Model found at {LOCAL_MODEL_PATH}")

def detect_gpu_total_memory_gb():
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,nounits,noheader"],
            capture_output=True, text=True, check=True, timeout=5
        )
        lines = r.stdout.strip().splitlines()
        if not lines:
            return None
        val = int(lines[0].strip())
        return max(1, math.floor(val / 1024))
    except Exception:
        return None

def choose_memory_mapping_for_infer(default_allowance_gb=14, cpu_offload_gb=400):
    detected = detect_gpu_total_memory_gb()
    if detected:
        logger.info(f"Detected GPU VRAM: {detected} GB")
        allowed = max(12, detected - 1)
        allowed = min(allowed, default_allowance_gb)
        gpu_mem_str = f"{allowed}GB"
    else:
        logger.warning("Could not detect GPU VRAM; using default allowance.")
        gpu_mem_str = f"{default_allowance_gb}GB"
    mem = {0: gpu_mem_str, "cpu": f"{cpu_offload_gb}GB"}
    logger.info(f"Max memory mapping for device-map inference: {mem}")
    return mem

def tokenize_function(examples, tokenizer, max_length):
    texts = examples.get("text") or examples.get("query") or ""
    if isinstance(texts, list):
        # Filter out any tokens that exceed the model's vocabulary
        tokenized = tokenizer(texts, truncation=True, max_length=max_length, padding="max_length")
        # Ensure all token IDs are within valid range
        for i in range(len(tokenized["input_ids"])):
            tokenized["input_ids"][i] = [min(token, tokenizer.vocab_size - 1) for token in tokenized["input_ids"][i]]
        return tokenized
    else:
        tokenized = tokenizer([texts], truncation=True, max_length=max_length, padding="max_length")
        tokenized["input_ids"][0] = [min(token, tokenizer.vocab_size - 1) for token in tokenized["input_ids"][0]]
        return tokenized

def check_vocab_alignment(tokenizer, model):
    """Check if tokenizer and model vocabulary are aligned"""
    logger.info(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    logger.info(f"Model vocab size: {model.config.vocab_size}")
    
    if tokenizer.vocab_size != model.config.vocab_size:
        logger.warning(f"Vocabulary size mismatch! Tokenizer: {tokenizer.vocab_size}, Model: {model.config.vocab_size}")
        return False
    return True

def check_token_ids(dataset, tokenizer, model, sample_size=100):
    """Check for token IDs that exceed the model's vocabulary size"""
    logger.info("Checking for out-of-vocabulary tokens...")
    
    # Sample some examples to check
    sample_indices = random.sample(range(len(dataset)), min(sample_size, len(dataset)))
    max_token_id = 0
    
    for idx in sample_indices:
        input_ids = dataset[idx]["input_ids"]
        current_max = max(input_ids)
        max_token_id = max(max_token_id, current_max)
    
    logger.info(f"Max token ID found in sample: {max_token_id}")
    
    if max_token_id >= model.config.vocab_size:
        logger.error(f"Found token ID {max_token_id} which exceeds model vocab size {model.config.vocab_size}")
        return False
    
    logger.info("All token IDs are within vocabulary bounds")
    return True

def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Ensure output and offload directories exist on D:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(OFFLOAD_DIR, exist_ok=True)
    logger.info(f"Outputs will be written to: {OUTPUT_DIR}")
    logger.info(f"Offloading will use: {OFFLOAD_DIR}")

    # Download model if needed
    download_model_if_needed()

    # Load datasets
    logger.info("Loading datasets from:")
    logger.info(f"  TRAIN: {TRAIN_FILE}")
    logger.info(f"  VAL:   {VAL_FILE}")
    train_ds = load_dataset("json", data_files=TRAIN_FILE)["train"]
    val_ds = None
    if Path(VAL_FILE).exists():
        val_ds = load_dataset("json", data_files=VAL_FILE)["train"]

    # Load tokenizer
    logger.info(f"Loading tokenizer from local path: {LOCAL_MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH, use_fast=False)
    logger.info("Tokenizer loaded successfully.")

    # Use EOS token as PAD token to avoid adding new tokens
    if tokenizer.pad_token is None:
        logger.info("Setting pad token to eos token.")
        tokenizer.pad_token = tokenizer.eos_token

    # bitsandbytes config (4-bit, NF4)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    # Load model with offloading to D: drive
    logger.info("Loading model with offloading to D: drive...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            LOCAL_MODEL_PATH,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
            offload_folder=OFFLOAD_DIR,  # Offload to D: drive
        )
        logger.info("Model loaded successfully with auto device mapping.")
    except Exception as e:
        logger.error(f"Failed to load model with auto device mapping: {e}")
        logger.info("Trying with CPU fallback...")
        model = AutoModelForCausalLM.from_pretrained(
            LOCAL_MODEL_PATH,
            quantization_config=bnb_config,
            device_map={"": "cpu"},
            trust_remote_code=True,
            torch_dtype=torch.float16,
            offload_folder=OFFLOAD_DIR,  # Offload to D: drive
        )
        logger.info("Model loaded on CPU with offloading to D: drive.")

    # Check vocabulary alignment
    if not check_vocab_alignment(tokenizer, model):
        logger.warning("Vocabulary mismatch detected. Resizing model embeddings...")
        model.resize_token_embeddings(len(tokenizer))
        logger.info(f"Resized model embeddings to match tokenizer: {len(tokenizer)}")

    # Set model config pad_token_id to match tokenizer
    if tokenizer.pad_token is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    # tokenize datasets
    logger.info("Tokenizing training dataset...")
    tokenized_train = train_ds.map(
        lambda x: tokenize_function(x, tokenizer, MAX_LENGTH), 
        batched=True, 
        remove_columns=train_ds.column_names
    )
    tokenized_val = None
    if val_ds is not None:
        tokenized_val = val_ds.map(
            lambda x: tokenize_function(x, tokenizer, MAX_LENGTH), 
            batched=True, 
            remove_columns=val_ds.column_names
        )

    # Check for out-of-vocabulary tokens
    if not check_token_ids(tokenized_train, tokenizer, model):
        logger.error("Found tokens that exceed model vocabulary. This will cause CUDA errors.")
        logger.error("Please check your tokenizer and dataset for consistency.")
        sys.exit(1)

    # smoke test forward
    try:
        device = next(model.parameters()).device
        logger.info("Running a short smoke-forward on device: %s", device)
        sample_text = "Smoke test."
        enc = tokenizer(sample_text, return_tensors="pt", truncation=True, max_length=64)
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        model.eval()
        with torch.no_grad():
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
        logger.info("Smoke test forward OK.")
    except Exception as smoke_err:
        logger.exception(
            "Smoke test forward failed. If this is a CUDA device-side assert, re-run with CUDA_LAUNCH_BLOCKING=1 to get a synchronous traceback."
        )
        raise smoke_err

    # prepare model for k-bit training and attach LoRA
    logger.info("Preparing model for k-bit training...")
    model = prepare_model_for_kbit_training(model)

    logger.info("Attaching LoRA adapters...")
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"LoRA attached — trainable params: {trainable_params:,}")

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # training args
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=True,
        logging_steps=50,
        save_total_limit=3,
        save_strategy="epoch",
        optim="paged_adamw_32bit",
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        report_to="none",
    )

    logger.info("Starting Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()

    # save LoRA adapters / training artifacts on D:
    logger.info("Saving LoRA adapters to %s", OUTPUT_DIR)
    model.save_pretrained(OUTPUT_DIR)
    logger.info("Training complete. Adapters saved to %s", OUTPUT_DIR)

if __name__ == "__main__":
    main()