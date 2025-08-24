"""
rationale_wrapper.py - helper wrappers to load LoRA-adapted Mistral model and PubMed verifier.
Provides RationaleLLMWrapper.generate_rationales(query, n) and VerifierWrapper.verify(rationale, passage).
"""

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
import torch

class RationaleLLMWrapper:
    def __init__(self, model_dir_or_id: str):
        # model_dir_or_id can be local path with LoRA adapters saved by peft
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir_or_id, use_fast=False)
        self.model = AutoModelForCausalLM.from_pretrained(model_dir_or_id, device_map="auto", torch_dtype=torch.float16)
        self.model.eval()

    def generate_rationales(self, query: str, n: int = 5, max_new_tokens: int = 192):
        prompt = f\"\"\"### Instruction:
Given the user query below, generate {n} concise rationales (1-2 sentences each) describing what evidence a correct passage should contain.

### Query:
{query}

### Response:
\"\"\"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        txt = self.tokenizer.decode(out[0], skip_special_tokens=True)
        # crude split - take lines after the 'Response:' marker
        parts = txt.split("### Response:")
        body = parts[-1] if parts else txt
        lines = [l.strip() for l in body.splitlines() if l.strip()]
        return lines[:n]

class VerifierWrapper:
    def __init__(self, model_dir_or_id: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir_or_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir_or_id, device_map="auto")
        self.model.eval()

    def verify(self, rationale: str, passage: str):
        text = f\"Rationale: {rationale}\\nPassage: {passage}\"
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.model.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0].cpu().tolist()
        score = float(probs[1])
        return {"ok": score > 0.5, "score": score, "reason": "model_prob"}
