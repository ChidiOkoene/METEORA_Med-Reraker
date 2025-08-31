TRAINING HELPER - README

Files:
- validate_and_split.py   -> validate, dedupe, phi-check & split your rationales_sft.jsonl
- train_qlora.py          -> QLoRA/LoRA training script for Mixtral-8x7B (local RTX 4090)

Suggested workflow:
1) Install dependencies (conda recommended):
   pip install -U pip
   pip install transformers accelerate datasets peft bitsandbytes safetensors evaluate

2) Validate & split your data:
   python /mnt/data/validate_and_split.py
   -> produces rationales_sft_clean.jsonl and train/val/test splits in /mnt/data/

3) Edit training hyperparameters in train_qlora.py if needed (batch size, epochs, r, alpha).

4) Run training (recommended with accelerate):
   accelerate launch /mnt/data/train_qlora.py

Notes & tips:
- If you run out of GPU memory, reduce PER_DEVICE_BATCH_SIZE to 1 and increase GRAD_ACCUM_STEPS.
- For Mixtral-8x7B QLoRA, bitsandbytes 4-bit loading is used. If you encounter bnb errors, try removing 4-bit flags and run on larger machine.
- After training, adapters are saved in ./lora_out. Use peft.PeftModel.from_pretrained to load adapters for inference.

Safety & review:
- Ensure clinician review of generated outputs before using in production.
- Check PHI flags CSV (/mnt/data/rationales_sft_phi_flags.csv) and remove/clean flagged items.
