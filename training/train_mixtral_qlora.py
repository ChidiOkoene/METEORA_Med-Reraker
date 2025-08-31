#!/usr/bin/env python3
r"""
training/train_mixtral_qlora.py

Downloads Mixtral model to D: drive and trains LoRA adapters using QLoRA.
- Local model path: D:\models\Mixtral-8x7B-Instruct-v0.1
- Train/Val files: C:\projects\meteora_project\data\rationales_sft_train.jsonl and _val.jsonl
- Outputs saved to D:\projects\meteora_project\outputs\lora_out

Run:
  accelerate launch --mixed_precision=fp16 training/train_mixtral_qlora.py
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

# accelerate utilities
from accelerate import init_empty_weights
from accelerate.utils import infer_auto_device_map

# Import huggingface_hub for authentication and downloading
from huggingface_hub import login, snapshot_download

# ---------------- USER PATHS & HYPERPARAMS ----------------
LOCAL_MODEL_PATH = r"D:\models\Mixtral-8x7B-Instruct-v0.1"
TRAIN_FILE = r"C:\projects\meteora_project\data\rationales_sft_train.jsonl"
VAL_FILE   = r"C:\projects\meteora_project\data\rationales_sft_val.jsonl"
OUTPUT_DIR = r"D:\projects\meteora_project\outputs\lora_out"

MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "1024"))
PER_DEVICE_BATCH_SIZE = int(os.environ.get("PER_DEVICE_BATCH_SIZE", "1"))
GRAD_ACCUM_STEPS = int(os.environ.get("GRAD_ACCUM_STEPS", "8"))
NUM_EPOCHS = int(os.environ.get("NUM_EPOCHS", "3"))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "2e-4"))

LORA_R = int(os.environ.get("LORA_R", "16"))
LORA_ALPHA = int(os.environ.get("LORA_ALPHA", "32"))
LORA_DROPOUT = float(os.environ.get("LORA_DROPOUT", "0.05"))

SEED = int(os.environ.get("SEED", "42"))

# If you want to skip device-map inference and use device_map="auto", set this env var to "true"
FORCE_SKIP_INFER = os.environ.get("FORCE_SKIP_INFER", "false").lower() in ("1", "true", "yes")
# -----------------------------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("train_mixtral_qlora")

def download_model_if_needed():
    """Download the model if it doesn't exist locally"""
    if not os.path.exists(LOCAL_MODEL_PATH):
        logger.info(f"Model not found at {LOCAL_MODEL_PATH}. Downloading...")
        os.makedirs(LOCAL_MODEL_PATH, exist_ok=True)
        
        # Ensure we're logged in to Hugging Face
        try:
            login()  # This will use cached credentials or prompt for login
        except Exception as e:
            logger.warning(f"Login failed: {e}. Trying to download without auth (may fail for gated models)")
        
        # Download the model
        try:
            snapshot_download(
                "mistralai/Mixtral-8x7B-Instruct-v0.1",
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

def choose_memory_mapping_for_infer(default_allowance_gb=22, cpu_offload_gb=400):
    detected = detect_gpu_total_memory_gb()
    if detected:
        logger.info(f"Detected GPU VRAM: {detected} GB")
        allowed = max(8, detected - 1)
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
        return tokenizer(texts, truncation=True, max_length=max_length, padding="max_length")
    return tokenizer([texts], truncation=True, max_length=max_length, padding="max_length")

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

    # Ensure output dir exists on D:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger.info(f"Outputs will be written to: {OUTPUT_DIR}")

    # Download model if needed
    download_model_if_needed()

    # load datasets
    logger.info("Loading datasets from:")
    logger.info(f"  TRAIN: {TRAIN_FILE}")
    logger.info(f"  VAL:   {VAL_FILE}")
    train_ds = load_dataset("json", data_files=TRAIN_FILE)["train"]
    val_ds = None
    if Path(VAL_FILE).exists():
        val_ds = load_dataset("json", data_files=VAL_FILE)["train"]

    # Load tokenizer from local path
    logger.info(f"Loading tokenizer from local path: {LOCAL_MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH, use_fast=False)
    logger.info("Tokenizer loaded successfully.")

    if tokenizer.pad_token is None:
        logger.info("Adding pad token to tokenizer.")
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    # bitsandbytes config (4-bit, NF4)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = None

    # device-map inference loading path (preferred)
    if not FORCE_SKIP_INFER:
        try:
            logger.info("Attempting device-map inference to maximize GPU placement...")
            config = AutoConfig.from_pretrained(LOCAL_MODEL_PATH, trust_remote_code=True)
            max_memory = choose_memory_mapping_for_infer(default_allowance_gb=22, cpu_offload_gb=400)

            logger.info("Building empty-weight skeleton for inference (no weights allocated)...")
            with init_empty_weights():
                skeleton = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

            no_split_module_classes = ["MoeLayer", "MoE", "MixtureOfExperts"]
            logger.info("Inferring device_map (may take a moment)...")
            device_map = infer_auto_device_map(
                skeleton,
                dtype=torch.float16,
                max_memory=max_memory,
                no_split_module_classes=no_split_module_classes,
            )
            logger.info(f"Inferred device_map summary: {device_map}")

            logger.info("Loading model from local files according to inferred device_map...")
            model = AutoModelForCausalLM.from_pretrained(
                LOCAL_MODEL_PATH,
                quantization_config=bnb_config,
                device_map=device_map,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            )
            logger.info("Model loaded with inferred device_map.")
        except Exception as e:
            logger.warning("Device-map inference path failed: %s", e)
            model = None

    # fallback: device_map="auto" -> CPU-only
    if model is None:
        try:
            logger.info("Attempting simpler load with device_map='auto'...")
            detected = detect_gpu_total_memory_gb() or 24
            allowed = max(8, detected - 1)
            max_memory = {0: f"{allowed}GB", "cpu": "400GB"}
            model = AutoModelForCausalLM.from_pretrained(
                LOCAL_MODEL_PATH,
                quantization_config=bnb_config,
                device_map="auto",
                max_memory=max_memory,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            )
            logger.info("Model loaded with device_map='auto'.")
        except Exception as e_auto:
            logger.warning("Auto load failed: %s", e_auto)
            logger.info("Falling back to CPU-only load. This will be slow.")
            model = AutoModelForCausalLM.from_pretrained(
                LOCAL_MODEL_PATH,
                quantization_config=bnb_config,
                device_map={"": "cpu"},
                trust_remote_code=True,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            )
            logger.info("Model loaded on CPU.")

    if model is None:
        raise RuntimeError("Failed to load model from local path. Check LOCAL_MODEL_PATH and local files.")

    # Check vocabulary alignment
    if not check_vocab_alignment(tokenizer, model):
        logger.warning("Vocabulary mismatch detected. Resizing model embeddings...")
        model.resize_token_embeddings(len(tokenizer))
        logger.info(f"Resized model embeddings to match tokenizer: {len(tokenizer)}")

    # resize embeddings if tokenizer changed vocab
    try:
        old_emb = model.get_input_embeddings().weight.size(0)
        new_emb = len(tokenizer)
        if new_emb != old_emb:
            logger.info(f"Resizing embeddings: {old_emb} -> {new_emb}")
            model.resize_token_embeddings(new_emb)
        else:
            logger.info("Token embeddings already match tokenizer.")
    except Exception as e:
        logger.exception("Exception when resizing embeddings (continuing): %s", e)

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