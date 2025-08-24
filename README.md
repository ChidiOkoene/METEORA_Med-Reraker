# METEORA Reranker - Project Workspace

This repository contains a METEORA-style reranker prototype and training artifacts.
Directory layout (root: `/mnt/data/meteora_project`):

- `src/` - core source code
  - `reranker.py` - refactored METEORA reranker implementation
  - `wrappers.py` - LLM and verifier wrapper helpers
- `training/` - training scripts for rationale generator and verifier
  - `train_mixtral_qlora.py`
  - `train_pubmed_verifier.py`
- `data/` - sample datasets and starter files
  - `rationales_sft_sample.jsonl` - sample SFT dataset (prompt+response)
  - `verifier_sample.jsonl` - small verifier training sample
- `configs/` - configs and templates (Label Studio config, accelerate config)
- `scripts/` - helper scripts and demos
  - `run_reranker.py` - demo to run reranker with heuristic fallbacks
- `docs/` - annotation guidelines and docs
- `requirements.txt` - Python dependencies
- `LICENSE` - MIT license stub

## Quick start (local)
1. Create Python environment and install dependencies from `requirements.txt`.
2. Review `configs/local_setup_instructions.txt` for local setup notes (CUDA/PyTorch).
3. Prepare your dataset in `data/` and update paths in training scripts.
4. Run verifier training: `python training/train_pubmed_verifier.py`
5. Run SFT (LoRA) training: `accelerate launch training/train_mixtral_qlora.py`
6. Use `scripts/run_reranker.py` to test the reranker (demo uses heuristic LLM fallback).

## Notes
- For full training of Mixtral-8x7B or Llama 70B, use cloud instances (A100-80GB or multi-GPU) and follow the QLoRA + FSDP recipes.
- The project includes simple fallback heuristics; replace them with your fine-tuned adapters via the wrappers for production.
