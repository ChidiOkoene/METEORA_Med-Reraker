PROMPT = """You are generating supervised fine-tuning (SFT) training data for a RAG reranker system called METEORA.  
Each output must be a single valid JSON object, one per line (JSONL format).  
Do not output anything else besides JSON objects.

### Format Specification:
Each JSON object must contain:
- "id": unique identifier, formatted "qXXX" where XXX is a 3-digit number
- "query": a realistic clinical/veterinary query with varied phrasing and scenarios
- "rationales": an array of exactly 3 rationales (1–2 sentences each)
- "text": concatenated instruction-query-response block

### Query Type Examples:
1. Exam questions: "Which of the following", "What is the most likely", "The best next step"
2. Clinical decisions: "How would you manage", "When to initiate", "Criteria for"
3. Guideline applications: "According to [guideline]", "Based on [standard]"
4. Diagnostic reasoning: "Interpret these findings", "Explain the significance of"
5. Treatment selection: "Compare options for", "Select appropriate therapy for"

### Rationale Guidelines:
- For exam questions: focus on key distinctions, classic presentations, and exclusion criteria
- For clinical decisions: emphasize risk-benefit analysis, timing considerations, and contraindications  
- For guidelines: highlight specific criteria, recommendation strengths, and supporting evidence
- Include variation in evidence types: clinical signs, labs, imaging, risk factors, timelines

### Example Outputs:
{{
  "id": "q001",
  "query": "A 58-year-old male presents with persistent cough and weight loss. Chest X-ray shows a right hilar mass. Which of the following is the most appropriate next diagnostic step?",
  "rationales": [
    "Look for mention of bronchoscopy as primary method for obtaining tissue diagnosis in suspected lung cancer.",
    "Seek confirmation that CT scan should be performed before invasive procedures to characterize the mass and guide biopsy.",
    "Check for exclusion of alternative first steps like sputum cytology which has lower sensitivity for central lesions."
  ],
  "text": "### Instruction:\nGiven the user query below, generate 3 concise rationales (1–2 sentences each) describing what evidence a correct passage should contain.\n\n### Query:\nA 58-year-old male presents with persistent cough and weight loss. Chest X-ray shows a right hilar mass. Which of the following is the most appropriate next diagnostic step?\n\n### Response:\nLook for mention of bronchoscopy as primary method for obtaining tissue diagnosis in suspected lung cancer.\nSeek confirmation that CT scan should be performed before invasive procedures to characterize the mass and guide biopsy.\nCheck for exclusion of alternative first steps like sputum cytology which has lower sensitivity for central lesions."
}}

For node: {node_name}  
Generate {n} JSON objects covering at least 3 different query types.
Ensure queries vary in:
- Severity levels (emergency to chronic)
- Clinical specialties (cardiology, oncology, etc.)
- Question formats (multiple-choice, open-ended, scenario-based)
- Temporal aspects (acute presentation vs long-term management)
"""





main_prompt = """You are generating supervised fine-tuning (SFT) training data 
for a RAG reranker system called METEORA.  
Each output must be a single valid JSON object, one per line (JSONL format).  
Do not output anything else besides JSON objects.  

### Format Specification:
Each JSON object must contain:
- "id": unique identifier, formatted "qXXX" where XXX is a 3-digit number.  
- "query": a realistic clinical or veterinary query relevant to the given node.  
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
  "text": "### Instruction:\nGiven the user query below, generate 3 concise rationales (1–2 sentences each) describing what evidence a correct passage should contain.\n\n### Query:\nExplain the renin-angiotensin-aldosterone system (RAAS) in simple terms.\n\n### Response:\nLook for an explanation of the system's trigger, starting with low blood pressure or low sodium detected by the kidneys.\nLook for a description of the key steps: renin release, conversion of angiotensinogen to angiotensin I, and then to angiotensin II by ACE.\nLook for the main effects of angiotensin II: vasoconstriction and stimulation of aldosterone release to increase sodium/water reabsorption."
}}

### Task:
For node: {node_name}  
Generate {n} JSON objects in this format.  
Ensure queries cover a wide range of scenarios, severity levels, diagnostic and management angles, and include variation in phrasing.  
The "rationales" should highlight different supporting evidence (clinical signs, labs, imaging, risk factors, differential points, etc.) depending on the query.
"""





