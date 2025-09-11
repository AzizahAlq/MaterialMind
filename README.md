# MaterialMind
MaterialMind is an LLM-powered materials selection assistant that retrieves relevant pages from your corpus, ranks candidate materials to your constraints, and explains trade-offs with page-level citations. Runs locally (Ollama + Flask) now; optional public/cloud deployment later.
MaterialMind

Decisions, not summaries. MaterialMind is an LLM-powered materials-selection assistant that turns engineering requirements into an evidence-backed, ranked shortlist with page-level citations. It runs locally (Flask + Ollama) and searches your own PDF corpus via a lightweight RAG index.

What MaterialMind does

• Retrieves → Ranks → Explains: finds the most relevant pages, scores candidate materials against your constraints, and explains trade-offs with citations.
• Local and private: your PDFs never leave your machine.
• Robust ingestion: handles real-world PDFs (papers, standards, reports, textbooks); incremental updates.
• Friendly UI: clean dark theme, background tiles, live weight controls (accepts 40/30/20/10 or 0.4/0.3/0.2/0.1).
• Exact math: backend normalizes weights to sum to exactly 1.0000.

Repository layout

materialmind/ (your corpus lives here)
├─ sources/ (drop your PDFs here)
└─ index/ (vector DB; auto-created)

rag_mini.py (RAG: index/search/ask/answer/export)
app_user.py (Flask app for end users)
templates/ (base.html, index.html, results.html)
static/ (styles.css and images: 11.png, 22.jpg)

Note: We work with PDFs you drop into materialmind/sources (papers, standards, reports). You don’t need vendor datasheets; if you have them as PDFs, drop them in like any other PDF.

Prerequisites

• Python 3.10+
• pip and a virtual environment
• Ollama (for model-generated answers). Example model: qwen2.5:7b-instruct
• Optional: ocrmypdf + tesseract (only for scanned PDFs with no selectable text)

Installation

Create and activate a virtual environment.

macOS / Linux:
python3 -m venv .venv
source .venv/bin/activate

Windows (PowerShell):
python -m venv .venv
.venv\Scripts\Activate.ps1

Install dependencies.

Apple Silicon (macOS):
pip install -U fastembed onnxruntime-silicon chromadb pypdf pymupdf markdown filelock flask flask-cors

Linux/Windows:
pip install -U fastembed onnxruntime chromadb pypdf pymupdf markdown filelock flask flask-cors

Install and start Ollama; pull a local model.

macOS (example):
brew install ollama
ollama serve &
ollama pull qwen2.5:7b-instruct

Confirm paths.

rag_mini.py uses a folder named “materialmind” next to the script:
BASE_DIR = Path(file).resolve().parent / "materialmind"

Ensure the folder exists:
materialmind/sources (create this and drop PDFs here)

Add your PDFs

Drop research papers, standards, reports, and textbooks into:
materialmind/sources

Tips:
• Prefer publisher PDFs (selectable text).
• If a PDF is scanned (no text), OCR it:
ocrmypdf input.pdf output_ocr.pdf
then place output_ocr.pdf in materialmind/sources

Build or update the index

First time (full index):
python rag_mini.py --rebuild

Later (only changed/new PDFs):
python rag_mini.py --update

Optional backup:
python rag_mini.py --backup

Optional export to JSONL (for inspection):
python rag_mini.py --export-json ./materialmind_dump.jsonl

Quick retrieval check (no model required)

Example:
python rag_mini.py --ask "Which alloys resist pitting in seawater better than 316L?"

You will see top-k snippets with file:page citations from your PDFs.

Run the UI

Start the Flask app:
python app_user.py

Open:
http://127.0.0.1:5000/

Usage:
• Fill environment/temperature/constraints.
• Set weights as percentages (e.g., 40/30/20/10) or fractions (0.4/0.3/0.2/0.1). The form enables the button only when the sum equals 100% (or 1.0).
• Click “Get ranked shortlist” to see the ranked table, material cards, and page-level citations.

How it works

• Retrieval: FastEmbed (ONNX) creates embeddings; Chroma stores and retrieves chunks with file + page metadata.
• Decision layer: your constraints and weights guide the model to produce a structured shortlist JSON (name, score, reasons, trade-offs, citations).
• Local LLM: by default qwen2.5:7b-instruct via Ollama; you can swap models freely.

CLI reference (rag_mini.py)

--rebuild rebuild the entire index from PDFs in materialmind/sources
--update incremental index of only changed/new PDFs
--backup copy the current index to a timestamped folder
--export-json PATH dump records to JSONL (optional)
--ask "question" retrieval only with citations
--answer "question" --model NAME --k N --show
retrieval plus local LLM answer (Ollama required)

Customization

Change model (Ollama):
ollama pull mistral:7b-instruct
then set “Model” in the UI to mistral:7b-instruct

Tune retrieval (rag_mini.py):
CHUNK_CHARS, CHUNK_OVERLAP, DEFAULT_TOPK

Swap embedding model (rag_mini.py):
EMB_MODEL = "BAAI/bge-small-en-v1.5" (good default). Other FastEmbed models also work.

Theme colors:
Edit CSS variables in static/styles.css (:root { … }).

Troubleshooting

“Get ranked shortlist” button disabled:
Enter weights as 40/30/20/10 or 0.4/0.3/0.2/0.1 until the sum reads 100%.

Ollama not found / model errors:
Install/start/pull as above. Retrieval still works without Ollama; only model-generated answers require it.

Chroma lock / mutex errors:
Close other processes using the index. If stuck:
python rag_mini.py --backup
rm -rf materialmind/index/chroma_v3
python rag_mini.py --rebuild

No text in a PDF:
OCR it with ocrmypdf, then re-index.

Port already in use:
Change the host/port in app_user.py.

Privacy

Everything runs locally by default. Your PDFs are indexed on disk and never uploaded. Remove PDFs from materialmind/sources and run --update to de-index.

Roadmap

Team/on-prem sharing, rule/property packs (e.g., PREN/oxidation windows/standards checks), uploads & tagging UI, LoRA adapter for stricter JSON, evaluation harness for grounding and JSON validity.

License

MIT (or your chosen permissive license). Add a LICENSE file to the repository.

One-minute demo

Drop a few corrosion or seawater PDFs into materialmind/sources.

Build: python rag_mini.py --rebuild

Run UI: python app_user.py

Query seawater at 20–25 °C, UTS ≥ 600 MPa, weights 50/30/10/10.

Show ranked shortlist, open a citation (file:page).

Add another PDF, run --update, rerun the query, and show the new citation.
