```markdown
# Vehicle Specification Extraction (Text-only RAG Pipeline)

This project extracts vehicle specifications (e.g., torque values, capacities, part numbers) from a service manual PDF using a **text-only** pipeline.

It uses:
- **PyMuPDF** to extract selectable text from the PDF (including text inside tables)
- **Chunking + embeddings** to enable semantic retrieval
- **Retrieval-Augmented Generation (RAG)** + an LLM to produce structured outputs
- Final outputs are saved as **JSON + CSV**

> Images/diagrams are intentionally ignored. This is a text-only solution.

---

## Directory Structure (Where to save what)

Create this folder structure:

```

vehicle-spec-extractor/
data/
manual.pdf
src/
step1_extract_pages.py
step2_make_chunks.py
step3_build_embeddings.py
step4_retrieve.py
step5_rag_llm_extract.py
outputs/
(auto-generated)
README.md

---

## Setup (Windows PowerShell)

### 1) Create + activate virtual environment
From inside the project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
````

### 2) Install dependencies

```powershell
pip install pymupdf sentence-transformers numpy requests
```

---

## Environment Variables (Required for Step 5)

Step 5 calls an OpenAI-compatible Chat Completions endpoint.

### Recommended (PowerShell - current session)

```powershell
$env:OPENAI_API_KEY="YOUR_KEY"
$env:OPENAI_MODEL="gpt-4o-mini"
$env:OPENAI_BASE_URL="https://api.openai.com"
```

### Quick check (make sure Python can see it)

```powershell
python -c "import os; print('KEY?', bool(os.getenv('OPENAI_API_KEY'))); print('MODEL', os.getenv('OPENAI_MODEL'))"
```

> If you see HTTP 429 during Step 5, it usually means **rate limit** or **insufficient quota/billing** on the API key/project.

---

## How to Run (End-to-end)

Run these commands from inside `vehicle-spec-extractor/`:

### Step 1 — Extract page text (PyMuPDF)

```powershell
python src/step1_extract_pages.py --pdf data/manual.pdf --out outputs/pages_raw.json
```

Output:

* `outputs/pages_raw.json`

---

### Step 2 — Clean + chunk (keeps table line breaks)

```powershell
python src/step2_make_chunks.py --infile outputs/pages_raw.json --pages_out outputs/pages_cleaned.json --chunks_out outputs/chunks.json
```

Outputs:

* `outputs/pages_cleaned.json`
* `outputs/chunks.json`

---

### Step 3 — Build embeddings

```powershell
python src/step3_build_embeddings.py --chunks outputs/chunks.json --out outputs/embeddings.npy
```

Outputs:

* `outputs/embeddings.npy`
* `outputs/embedding_info.json`

---

### Step 4 — Retrieval test (no LLM)

Use this to confirm the manual actually contains the answer in retrieved chunks.

```powershell
python src/step4_retrieve.py --query "Torque for brake caliper bolts" --top_k 10
```

Output:

* `outputs/retrieved_chunks.json`

---

### Step 5 — RAG + LLM extraction (final structured output)

```powershell
python src/step5_rag_llm_extract.py --query "Torque for brake caliper bolts" --top_k 10
```

Final outputs:

* `outputs/results.json`
* `outputs/results.csv`

---

## Output Format (Step 5)

Each extracted record follows this schema:

* `component` (string)
* `spec_type` (string) — e.g., Torque, Capacity, PartNumber
* `value` (string)
* `unit` (string)
* `source_page` (number)

---

## Notes / Limitations

* This solution is **text-only**: diagrams/images are ignored.
* Tables are extracted as text. Row/column formatting may not always be perfect.
* If the PDF is scanned or image-only, OCR would be needed (not implemented here).

---

