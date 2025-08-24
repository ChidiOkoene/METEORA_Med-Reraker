# run_reranker.py - demo script to run the METEORA reranker locally (uses files in src/)
import json
from pathlib import Path
root = Path(__file__).resolve().parent.parent
import sys
sys.path.append(str(root / "src"))

from reranker import MeteoraReranker
from wrappers import RationaleLLMWrapper, VerifierWrapper

# Replace these with your local model ids/paths if you have adapters
# For demo we'll use the heuristic fallback in reranker (llm=None)
llm = None
verifier = None

# Load sample data
data_file = root / "data" / "rationales_sft_sample.jsonl"
docs = []
with data_file.open() as f:
    for line in f:
        obj = json.loads(line)
        docs.append({"id": obj['id'], "text": obj['query']})

reranker = MeteoraReranker(llm=llm)
out = reranker.rerank("What evidence supports pulmonary embolism?", docs)
print(json.dumps(out, indent=2))
