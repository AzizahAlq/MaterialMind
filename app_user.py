#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MaterialMind – End-user Flask app
- Simple form for constraints
- RAG retrieval + structured LLM answer
- Renders ranked shortlist + “cards” + citations
"""

from pathlib import Path
import re, json, textwrap, subprocess
from typing import List, Tuple
from flask import Flask, request, render_template, redirect, url_for, flash
from flask_cors import CORS
from filelock import FileLock

# Import your RAG helpers
from rag_mini import (
    search, ensure_dirs, DATA_DIR, DEFAULT_TOPK, DEFAULT_MODEL
)

app = Flask(__name__)
app.secret_key = "change-me"  # set a real secret for production
CORS(app)

BASE_DIR = DATA_DIR.parent
LOCK_PATH = BASE_DIR / ".rag_lock"

# --- LLM caller (Ollama) ---
def call_ollama(model: str, prompt: str) -> str:
    try:
        out = subprocess.run(
            ["ollama", "run", model, prompt],
            check=True, capture_output=True, text=True
        )
        return out.stdout.strip()
    except FileNotFoundError:
        return ("[Error] Ollama not found. Install: brew install ollama\n"
                "Run: ollama serve &\n"
                f"Pull: ollama pull {model}")
    except subprocess.CalledProcessError as e:
        return f"[Error] ollama run failed: {e.stderr.strip() or e.stdout.strip()}"

# --- Prompting ---
SYSTEM_RULES = """You are MaterialMind, a materials-selection assistant.
Return two things:
1) A JSON block with a ranked shortlist, using this exact schema:
{
  "candidates": [
    {
      "name": "string",
      "score": 0.0,            // 0..1 overall score
      "reasons": ["string", ...],
      "tradeoffs": ["string", ...],
      "citations": ["[1]", "[2]"]   // map to the citations list
    }
  ]
}
2) After the JSON, a short narrative (3–6 bullet points) explaining the trade-offs.
Rules:
- Use only the provided context; ask for missing constraints if critical.
- Prefer pitting/crevice resistance signals for seawater questions (e.g., PREN).
- Be conservative if data is weak; never fabricate.
- Keep units correct and clearly stated.
"""

ANSWER_TEMPLATE = """{rules}

User constraints:
- Environment: {environment}
- Temperature: {temperature}
- Min UTS (MPa): {min_uts}
- Max density (g/cm^3): {max_density}
- Budget: {budget}
- Process: {process}
- Weights: corrosion={w_corrosion}, strength={w_strength}, cost={w_cost}, availability={w_availability}

Question:
{question}

Context snippets (numbered):
{context}

Citations:
{citations}

Now, first output the JSON only (no preamble). Then a short narrative.
"""

def format_context(hits: List[Tuple[str, str]]) -> Tuple[str, str]:
    blocks, cites = [], []
    for i, (text, cite) in enumerate(hits, 1):
        snippet = textwrap.shorten(text.replace("\n", " "), width=450, placeholder=" …")
        blocks.append(f"[{i}] {snippet}")
        cites.append(f"[{i}] {cite}")
    return "\n".join(blocks), "\n".join(cites)

def extract_json_block(text: str):
    # Find the first {...} JSON block (optionally wrapped in ```json ... ```)
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.S)
    if m:
        s = m.group(1)
    else:
        m2 = re.search(r"(\{.*\})", text, flags=re.S)
        s = m2.group(1) if m2 else None
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        # try to trim trailing text after last closing }
        last = s.rfind("}")
        if last != -1:
            try:
                return json.loads(s[:last+1])
            except Exception:
                return None
        return None

# --- Routes ---
@app.get("/")
def index():
    return render_template("index.html", default_model=DEFAULT_MODEL, default_k=DEFAULT_TOPK)

@app.post("/recommend")
def recommend():
    # Read form inputs
    environment  = request.form.get("environment", "").strip() or "seawater"
    temperature  = request.form.get("temperature", "").strip() or "20–25 °C"
    min_uts      = request.form.get("min_uts", "").strip() or "0"
    max_density  = request.form.get("max_density", "").strip() or "100"
    budget       = request.form.get("budget", "").strip() or "open"
    process      = request.form.get("process", "").strip() or "any"
    w_corrosion  = float(request.form.get("w_corrosion", "0.4"))
    w_strength   = float(request.form.get("w_strength", "0.3"))
    w_cost       = float(request.form.get("w_cost", "0.2"))
    w_avail      = float(request.form.get("w_availability", "0.1"))
    model        = request.form.get("model", DEFAULT_MODEL).strip()
    k            = int(request.form.get("k", DEFAULT_TOPK))

    # Build a query for retrieval
    question = f"For {environment} at {temperature}, shortlist materials that meet UTS ≥ {min_uts} MPa and density ≤ {max_density} g/cm^3. Consider budget={budget} and process={process}. Rank by corrosion resistance, strength, cost, and availability."

    # Retrieve context
    hits = search(question, k=k)
    if not hits:
        flash("No context found. Please add sources and rebuild/update the index.", "error")
        return redirect(url_for("index"))

    ctx, cites = format_context(hits)

    # Build the LLM prompt
    prompt = ANSWER_TEMPLATE.format(
        rules=SYSTEM_RULES, environment=environment, temperature=temperature,
        min_uts=min_uts, max_density=max_density, budget=budget, process=process,
        w_corrosion=w_corrosion, w_strength=w_strength, w_cost=w_cost, w_availability=w_avail,
        question=question, context=ctx, citations=cites
    )

    # Single-writer guard if you add write ops; here reads are fine
    with FileLock(str(LOCK_PATH), timeout=2):
        raw = call_ollama(model, prompt)

    # Parse candidates JSON
    parsed = extract_json_block(raw) if raw else None
    candidates = (parsed or {}).get("candidates", []) if parsed else []

    return render_template(
        "results.html",
        raw_output=raw,
        candidates=candidates,
        citations=cites.splitlines(),
        environment=environment,
        temperature=temperature
    )

if __name__ == "__main__":
    # Run: python app_user.py (http://127.0.0.1:5000/)
    # Prod: gunicorn -w 1 -b 0.0.0.0:5000 app_user:app
    ensure_dirs()
    app.run(host="127.0.0.1", port=5000, debug=False)
