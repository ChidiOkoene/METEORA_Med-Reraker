#!/usr/bin/env python3
import os
import json
import time
import re
import datetime
import traceback
from pathlib import Path
from openai import OpenAI  # OpenAI-compatible SDK

# --------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------
TAXONOMY_FILE = "data/medical_vet_taxonomy_for_meteora.json"
OUTPUT_FILE = "data/rationales_sft.jsonl"
MODEL_NAME = "mistralai/mistral-7b-instruct-v0.3"
N_EXAMPLES_PER_NODE = 20
START_ID = 1

DEBUG_LOG = "debug_responses.log"  # single debug file

PROMPT_TEMPLATE = """You are generating supervised fine-tuning (SFT) training data 
for a RAG reranker system called METEORA.  
Output ONLY valid JSON objects in JSONL format (one JSON object per line).  
Do not output any other text, explanations, or markdown formatting.

### Format Specification:
Each JSON object must contain:
- "id": unique identifier, formatted "qXXX" where XXX is a 3-digit number.  
- "query": a realistic clinical or veterinary query relevant to the given node(make it sound as veterinary as possible).  
- "rationales": an array of exactly 3 rationales (1–2 sentences each).  
- "text": a concatenated instruction–query–response block as in the example.  

### Example:
{{
    "id": "q001",
  "query": "Explain the renin-angiotensin-aldosterone system (RAAS) in simple terms.",
  "rationales": [
    "Look for an explanation of the system's trigger, starting with low blood pressure or low sodium detected by the kidneys.",
    "Look for a description of the key steps: renin release, conversion of angiotensinogen to angiotensin I, and then to angiotensin II by ACE.",
    "Look for the main effects of angiotensin II: vasoconstriction and stimulation of aldosterone release to increase sodium/water reabsorption."
    ],
  "text": "### Instruction:\\nGiven the user query below, generate 3 concise rationales (1–2 sentences each) describing what evidence a correct passage should contain.\\n\\n### Query:\\nExplain the renin-angiotensin-aldosterone system (RAAS) in simple terms.\\n\\n### Response:\\nLook for an explanation of the system's trigger, starting with low blood pressure or low sodium detected by the kidneys.\\nLook for a description of the key steps: renin release, conversion of angiotensinogen to angiotensin I, and then to angiotensin II by ACE.\\nLook for the main effects of angiotensin II: vasoconstriction and stimulation of aldosterone release to increase sodium/water reabsorption."
}}

### Task:
For node: {node_name}  
Generate exactly {n} JSON objects in this format, outputting only the JSON objects with no additional text.  
Ensure queries cover a wide range of animals, scenarios, severity levels, diagnostic and management angles, and include variation in phrasing, never repeat a query.  
The "3 rationales" should highlight different supporting evidence (clinical signs, labs, imaging, risk factors, differential points, etc.) depending on the query.
"""

# --------------------------------------------------------------------
# HELPERS
# --------------------------------------------------------------------
def load_taxonomy(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_node_paths(taxonomy_list):
    return [node["path"] for node in taxonomy_list if "path" in node]

def append_debug_log(header: str, body: str):
    ts = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S%z")
    with open(DEBUG_LOG, "a", encoding="utf-8") as fh:
        fh.write(f"\n\n=== {ts} === {header}\n")
        fh.write(body)
        fh.write("\n=== end ===\n")

def clean_text(text):
    """
    Remove invalid control characters from text
    """
    # Remove control characters except for tab, newline, and carriage return
    return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)

def extract_json_objects_from_text(text):
    """
    Extract JSON objects from text, handling both single objects and arrays
    """
    text = clean_text(text)
    objects = []
    
    # Try to parse as a JSON array first
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
        elif isinstance(parsed, dict):
            return [parsed]
    except json.JSONDecodeError:
        pass
    
    # Try to find individual JSON objects
    json_pattern = r'\{[^{}]*\}'
    matches = re.finditer(json_pattern, text, re.DOTALL)
    
    for match in matches:
        try:
            json_obj = json.loads(match.group())
            objects.append(json_obj)
        except json.JSONDecodeError:
            continue
            
    return objects

def extract_candidate_texts_from_response(response):
    """
    Extract the actual content from LM Studio responses
    """
    cands = []
    
    # Handle LM Studio response format
    if hasattr(response, 'choices') and response.choices:
        for choice in response.choices:
            if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                cands.append(choice.message.content)
    
    # Fallback to other possible formats
    if not cands and hasattr(response, 'output_text'):
        cands.append(response.output_text)
        
    if not cands and hasattr(response, 'output'):
        cands.append(str(response.output))
        
    # Final fallback
    if not cands:
        cands.append(str(response))
        
    return cands

def get_next_global_id_from_file(path, start_id):
    if not os.path.exists(path):
        return start_id
    max_id = start_id - 1
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
                idv = o.get("id") or o.get("qid") or ""
                m = re.search(r"q(\d+)", str(idv))
                if m:
                    v = int(m.group(1))
                    if v > max_id:
                        max_id = v
            except Exception:
                continue
    return max_id + 1

# --------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------
def main():
    client = OpenAI(api_key="lm-studio", base_url="http://127.0.0.1:1234/v1")
    print("=== Starting generation ===")
    start_time = time.time()
    print(f"Starting at {start_time}")
    print(f"Model: {MODEL_NAME}")

    taxonomy = load_taxonomy(TAXONOMY_FILE)
    nodes = get_node_paths(taxonomy)

    # TEMP: for debugging, you can uncomment the next line to restrict nodes
    # nodes = nodes[:3]

    global_id = get_next_global_id_from_file(OUTPUT_FILE, START_ID)

    out_path = Path(OUTPUT_FILE)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("a", encoding="utf-8") as out_f:
        for node_name in nodes:
            print(f"\n=== Generating for node: {node_name} ===")
            system_msg = ("You are a strict data generator. Output ONLY the requested JSON objects "
                          "in exact JSONL format (one JSON object per line). Do NOT output any explanation, numbering, "
                          "markdown formatting, or extra commentary. If you cannot produce the requested JSON, return an empty array [].")
            prompt = PROMPT_TEMPLATE.format(node_name=node_name, n=N_EXAMPLES_PER_NODE)
            print("Prompt length:", len(prompt))

            # We'll accumulate candidate_texts here:
            candidate_texts = []
            raw_response_repr = None
            response_error = None

            # Use chat.completions.create for LM Studio
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                    max_tokens=3000,
                )
                raw_response_repr = repr(response)
                candidate_texts = extract_candidate_texts_from_response(response)
                
                # Debug output
                print(f"Number of candidate texts: {len(candidate_texts)}")
                for i, text in enumerate(candidate_texts):
                    print(f"Candidate {i} (first 200 chars): {text[:200]}...")
                    
            except Exception as e:
                exc_info = traceback.format_exc()
                append_debug_log(
                    f"Node: {node_name} - exception in API call",
                    f"exception:\n{exc_info}\n"
                )
                candidate_texts = [f"API Error: {str(e)}"]
                print(f"API Error: {str(e)}")

            # Write unified debug entry for this node
            dbg_body = (
                f"node: {node_name}\n"
                f"prompt_length: {len(prompt)}\n"
                f"response_error: {response_error}\n\n"
                f"raw_response_repr:\n{(raw_response_repr or 'None')}\n\n"
                f"candidate_texts (first 3000 chars each):\n"
            )
            for idx, ct in enumerate(candidate_texts, 1):
                dbg_body += f"\n--- candidate {idx} ---\n{ct[:2500]}\n"
            append_debug_log(f"Node debug: {node_name}", dbg_body)

            # Attempt to parse candidate_texts to JSON objects
            parsed_objects = []
            for txt in candidate_texts:
                objs = extract_json_objects_from_text(txt)
                if objs:
                    parsed_objects.extend(objs)
                    break

            print(f"Number of parsed objects: {len(parsed_objects)}")
            
            if not parsed_objects:
                print(f"⚠️ No JSON objects parsed for node: {node_name}")
                time.sleep(1)
                continue

            # Validate and write parsed objects
            wrote = 0
            for obj in parsed_objects:
                if not isinstance(obj, dict):
                    continue
                if "query" not in obj or "rationales" not in obj:
                    print("⚠️ Parsed object missing required keys, skipping:", list(obj.keys()))
                    continue
                    
                # Generate text field if missing
                if "text" not in obj:
                    if isinstance(obj["rationales"], list):
                        response_text = "\n".join(obj["rationales"])
                    else:
                        response_text = str(obj["rationales"])
                    obj["text"] = f"### Instruction:\nGiven the user query below, generate 3 concise rationales (1–2 sentences each) describing what evidence a correct passage should contain.\n\n### Query:\n{obj['query']}\n\n### Response:\n{response_text}"
                
                obj["id"] = f"q{global_id:04d}"
                out_f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                global_id += 1
                wrote += 1

            print(f"Wrote {wrote} objects for node: {node_name}")
            time.sleep(1)

    print(f"\n✅ Dataset generation complete → {OUTPUT_FILE}")

if __name__ == "__main__":
    main()