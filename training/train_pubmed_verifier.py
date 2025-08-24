"""
train_pubmed_verifier.py
Train a PubMedBERT-based verifier (sequence classification) locally.
Assumes verifier.jsonl exists with fields: 'rationale', 'passage', 'label' (0/1).

Install requirements:
  pip install transformers datasets accelerate

Run:
  python train_pubmed_verifier.py
"""

import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

MODEL = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"  # PubMedBERT
DATA_PATH = "verifier.jsonl"  # jsonl with rationale, passage, label (0/1)
OUTPUT_DIR = "./pubmed_verifier_out"

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2)
    ds = load_dataset("json", data_files=DATA_PATH, split="train")

    def preprocess(ex):
        text = "Rationale: " + ex["rationale"].strip() + "\nPassage: " + ex["passage"].strip()
        return tokenizer(text, truncation=True, max_length=512)

    tokenized = ds.map(preprocess, batched=True)
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format(type="torch", columns=["input_ids","attention_mask","labels"])

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=8,
        num_train_epochs=3,
        fp16=True if torch.cuda.is_available() else False,
        logging_steps=50,
        save_strategy="epoch",
        learning_rate=2e-5,
        report_to=[],
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=tokenized)
    trainer.train()
    model.save_pretrained(OUTPUT_DIR)
    print("Verifier saved to", OUTPUT_DIR)

if __name__ == "__main__":
    main()
