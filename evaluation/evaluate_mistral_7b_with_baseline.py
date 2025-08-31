#!/usr/bin/env python3
"""
evaluate_mistral_7b_with_baseline.py

Evaluates and compares the fine-tuned Mistral-7B-Instruct-v0.3 model with LoRA adapters
against the baseline (original) model on the validation dataset.
"""

import os
import json
import torch
import numpy as np
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    DataCollatorForLanguageModeling
)
from peft import PeftModel
from datasets import load_dataset
from tqdm import tqdm
import time

# Paths
MODEL_PATH = r"D:\models\Mistral-7B-Instruct-v0.3"
LORA_ADAPTER_PATH = r"D:\projects\meteora_project\outputs\lora_out_mistral_7b_v3"
VAL_FILE = r"C:\projects\meteora_project\data\rationales_sft_test.jsonl"
RESULTS_PATH = r"D:\projects\meteora_project\outputs\evaluation_comparison_results.json"

# Evaluation parameters
MAX_LENGTH = 1024
BATCH_SIZE = 1
NUM_SAMPLES = None  # Number of samples to evaluate (use None for full dataset)

def load_model_and_tokenizer(use_lora=True):
    """Load the model and tokenizer, with or without LoRA adapters"""
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    
    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    
    if use_lora:
        print("Loading LoRA adapters...")
        model = PeftModel.from_pretrained(model, LORA_ADAPTER_PATH)
    
    return model, tokenizer

def calculate_perplexity(model, tokenizer, texts, model_name="model"):
    """Calculate perplexity for given texts"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    losses = []
    
    start_time = time.time()
    
    with torch.no_grad():
        for text in tqdm(texts, desc=f"Calculating perplexity ({model_name})"):
            inputs = tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=MAX_LENGTH
            )
            
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            
            total_loss += loss.item() * inputs["input_ids"].numel()
            total_tokens += inputs["input_ids"].numel()
            losses.append(loss.item())
    
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    evaluation_time = time.time() - start_time
    
    return {
        "perplexity": perplexity,
        "average_loss": avg_loss,
        "loss_std": np.std(losses),
        "evaluation_time_seconds": evaluation_time,
        "samples_per_second": len(texts) / evaluation_time
    }

def generate_examples(model, tokenizer, prompts, num_examples=5, model_name="model"):
    """Generate example outputs from the model"""
    model.eval()
    examples = []
    
    for i, prompt in enumerate(prompts[:num_examples]):
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        examples.append({
            "prompt": prompt,
            "generated_text": generated_text
        })
    
    return examples

def main():
    print("Starting evaluation with baseline comparison...")
    
    # Load validation data
    print("Loading validation data...")
    val_ds = load_dataset("json", data_files=VAL_FILE)["train"]
    
    # Sample if specified
    if NUM_SAMPLES and NUM_SAMPLES < len(val_ds):
        val_ds = val_ds.select(range(NUM_SAMPLES))
    
    # Extract texts for evaluation
    texts = []
    for example in val_ds:
        text = example.get("text") or example.get("query") or ""
        if text:
            texts.append(text)
    
    print(f"Evaluating on {len(texts)} samples...")
    
    # Evaluate baseline model
    print("\n=== EVALUATING BASELINE MODEL ===")
    baseline_model, tokenizer = load_model_and_tokenizer(use_lora=False)
    baseline_results = calculate_perplexity(baseline_model, tokenizer, texts, "baseline")
    baseline_examples = generate_examples(baseline_model, tokenizer, texts, num_examples=5, model_name="baseline")
    
    # Free up memory
    del baseline_model
    torch.cuda.empty_cache()
    
    # Evaluate fine-tuned model
    print("\n=== EVALUATING FINE-TUNED MODEL ===")
    finetuned_model, tokenizer = load_model_and_tokenizer(use_lora=True)
    finetuned_results = calculate_perplexity(finetuned_model, tokenizer, texts, "fine-tuned")
    finetuned_examples = generate_examples(finetuned_model, tokenizer, texts, num_examples=5, model_name="fine-tuned")
    
    # Calculate improvement percentages
    perplexity_improvement = ((baseline_results["perplexity"] - finetuned_results["perplexity"]) / baseline_results["perplexity"]) * 100
    loss_improvement = ((baseline_results["average_loss"] - finetuned_results["average_loss"]) / baseline_results["average_loss"]) * 100
    
    # Prepare results
    results = {
        "model_info": {
            "base_model": "Mistral-7B-Instruct-v0.3",
            "lora_adapters_path": LORA_ADAPTER_PATH,
            "validation_set_size": len(texts),
        },
        "baseline_model": baseline_results,
        "fine_tuned_model": finetuned_results,
        "comparison": {
            "perplexity_improvement_percent": perplexity_improvement,
            "loss_improvement_percent": loss_improvement,
            "relative_speed": finetuned_results["samples_per_second"] / baseline_results["samples_per_second"]
        },
        "example_outputs": {
            "baseline": baseline_examples,
            "fine_tuned": finetuned_examples
        },
        "evaluation_parameters": {
            "max_length": MAX_LENGTH,
            "batch_size": BATCH_SIZE,
            "num_samples_evaluated": len(texts)
        }
    }
    
    # Save results
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n=== EVALUATION RESULTS ===")
    print(f"Baseline Perplexity: {baseline_results['perplexity']:.4f}")
    print(f"Fine-tuned Perplexity: {finetuned_results['perplexity']:.4f}")
    print(f"Perplexity Improvement: {perplexity_improvement:.2f}%")
    print(f"\nBaseline Average Loss: {baseline_results['average_loss']:.4f}")
    print(f"Fine-tuned Average Loss: {finetuned_results['average_loss']:.4f}")
    print(f"Loss Improvement: {loss_improvement:.2f}%")
    print(f"\nBaseline Speed: {baseline_results['samples_per_second']:.2f} samples/sec")
    print(f"Fine-tuned Speed: {finetuned_results['samples_per_second']:.2f} samples/sec")
    print(f"Relative Speed: {results['comparison']['relative_speed']:.2f}x")
    print(f"\nResults saved to: {RESULTS_PATH}")
    
    print("\n=== EXAMPLE OUTPUTS ===")
    for i in range(min(3, len(baseline_examples))):
        print(f"\nExample {i + 1}:")
        print(f"Prompt: {baseline_examples[i]['prompt'][:100]}...")
        print(f"Baseline: {baseline_examples[i]['generated_text'][:150]}...")
        print(f"Fine-tuned: {finetuned_examples[i]['generated_text'][:150]}...")
    
    return results

if __name__ == "__main__":
    main()