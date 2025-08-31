#!/usr/bin/env python3
"""validate_and_split.py

Validate, deduplicate, and split a rationales SFT JSONL file.
Input: /mnt/data/rationales_sft.jsonl
Outputs:
  - /mnt/data/rationales_sft_clean.jsonl
  - /mnt/data/rationales_sft_train.jsonl
  - /mnt/data/rationales_sft_val.jsonl
  - /mnt/data/rationales_sft_test.jsonl
  - /mnt/data/rationales_sft_validation_report.json
  - /mnt/data/rationales_sft_phi_flags.csv

Run:
  python validate_and_split.py
"""

import json, re, hashlib, random, csv
from pathlib import Path

INPUT = Path("data/rationales_sft.jsonl")
CLEAN = Path("data/rationales_sft_clean.jsonl")
TRAIN = Path("data/rationales_sft_train.jsonl")
VAL = Path("data/rationales_sft_val.jsonl")
TEST = Path("data/rationales_sft_test.jsonl")
REPORT = Path("data/rationales_sft_validation_report.json")
PHI_CSV = Path("data/rationales_sft_phi_flags.csv")

SEED = 42
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1
MAX_WORDS_RATIONALE = 80    # max words per rationale to accept
MIN_WORDS_RATIONALE = 4     # min words per rationale to accept
MIN_RATIONALES = 3
EXPECTED_FIELDS = ["id","query","rationales","text"]

# PHI regexes (simple heuristics)
RE_EMAIL = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
RE_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
RE_PHONE = re.compile(r"\b\+?\d[\d\s\-()]{7,}\d\b")
RE_DATE = re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b")
RE_MRN = re.compile(r"MRN[:\s]*\d{4,}")
RE_NAME_SIMPLE = re.compile(r"\b([A-Z][a-z]+\s[A-Z][a-z]+)\b")  # naive

def hash_entry(obj):
    # hash query + rationales text to deduplicate
    key = obj.get("query","") + " ||| " + " ||| ".join(obj.get("rationales",[]))
    return hashlib.sha256(key.encode("utf8")).hexdigest()

def phi_check(text):
    flags = []
    if RE_EMAIL.search(text): flags.append("email")
    if RE_SSN.search(text): flags.append("ssn")
    if RE_PHONE.search(text): flags.append("phone")
    if RE_DATE.search(text): flags.append("date")
    if RE_MRN.search(text): flags.append("mrn")
    # naive name check - may produce false positives
    if RE_NAME_SIMPLE.search(text):
        flags.append("name_like")
    return flags

def quality_check(obj):
    issues = []
    for f in EXPECTED_FIELDS:
        if f not in obj:
            issues.append(f"missing_field:{f}")
    # rationales list
    rats = obj.get("rationales",[])
    if not isinstance(rats, list) or len(rats) < MIN_RATIONALES:
        issues.append("bad_rationales_count")
    else:
        for i,r in enumerate(rats):
            words = len(r.split())
            if words < MIN_WORDS_RATIONALE:
                issues.append(f"rationale_{i}_too_short")
            if words > MAX_WORDS_RATIONALE:
                issues.append(f"rationale_{i}_too_long")
    # text length
    if not obj.get("text") or len(obj.get("text")) < 20:
        issues.append("text_too_short")
    return issues

def main():
    if not INPUT.exists():
        print("Input file not found:", INPUT)
        return

    seen = set()
    clean = []
    phi_rows = []

    total = 0
    malformed = 0
    for line in INPUT.open(encoding="utf8"):
        total += 1
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception as e:
            malformed += 1
            continue

        # Basic quality checks
        issues = quality_check(obj)
        if issues:
            obj["_qc_issues"] = issues
            phi_rows.append({"id": obj.get("id",""), "issues": ";".join(issues), "query": obj.get("query","")})
            continue

        # PHI scan across fields (query + rationales + text)
        combined = obj.get("query","") + "\n" + "\n".join(obj.get("rationales",[])) + "\n" + obj.get("text","")
        flags = phi_check(combined)
        if flags:
            phi_rows.append({"id": obj.get("id",""), "issues": ";".join(flags), "query": obj.get("query","")})

        # Deduplicate
        h = hash_entry(obj)
        if h in seen:
            continue
        seen.add(h)
        clean.append(obj)

    # Shuffle and split
    random.seed(SEED)
    random.shuffle(clean)
    n = len(clean)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    train_items = clean[:n_train]
    val_items = clean[n_train:n_train+n_val]
    test_items = clean[n_train+n_val:]

    # Write outputs
    for path, items in [(CLEAN, clean),(TRAIN, train_items),(VAL, val_items),(TEST, test_items)]:
        with path.open("w", encoding="utf8") as f:
            for o in items:
                f.write(json.dumps(o, ensure_ascii=False) + "\n")

    # PHI CSV
    with PHI_CSV.open("w", newline='', encoding="utf8") as csvf:
        writer = csv.DictWriter(csvf, fieldnames=["id","issues","query"])
        writer.writeheader()
        for r in phi_rows:
            writer.writerow(r)

    report = {
        "total_input": total,
        "malformed": malformed,
        "clean_count": len(clean),
        "train_count": len(train_items),
        "val_count": len(val_items),
        "test_count": len(test_items),
        "deduplicated": total - len(clean) - malformed
    }
    with REPORT.open("w", encoding="utf8") as rf:
        json.dump(report, rf, indent=2)

    print("Validation complete. Report:")
    print(json.dumps(report, indent=2))
    print("Clean file:", CLEAN)
    print("Train/Val/Test written to:", TRAIN, VAL, TEST)
    print("PHI flags (sample):", PHI_CSV)

if __name__ == '__main__':
    main()
