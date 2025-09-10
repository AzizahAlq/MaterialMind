#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AlloyMind RAG (Torch-free embeddings via FastEmbed to avoid macOS mutex locks)

Usage:
  python rag_mini.py --rebuild
  python rag_mini.py --update
  python rag_mini.py --backup
  python rag_mini.py --ask "question"
  python rag_mini.py --answer "question" --model qwen2.5:7b-instruct --k 6 --show
  python rag_mini.py --export-json "/path/to/alloymind_dump.jsonl"
"""

import os, re, uuid, argparse, subprocess, textwrap, logging, warnings
import hashlib, json, shutil, datetime
from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Any

# ---------- PATHS ----------
BASE_DIR = Path("/Users/azizahalq/Desktop/AlloyMind_Ingestion_Kit/AlloyMind")
DATA_DIR = BASE_DIR / "sources"
DB_DIR   = BASE_DIR / "index" / "chroma_v3"   # fresh DB path

# ---------- CONFIG ----------
EMB_MODEL = "BAAI/bge-small-en-v1.5"   # fallback if FastEmbed unavailable
CHUNK_CHARS = 1200
CHUNK_OVERLAP = 150
DEFAULT_TOPK = 5
DEFAULT_MODEL = "qwen2.5:7b-instruct"
MANIFEST_PATH = BASE_DIR / "index" / "manifest.json"

# Silence noisy PDF logs
logging.getLogger("pypdf").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module="pypdf")

# ---- Lazy imports (heavy) ----
def _lazy_imports():
    global chromadb
    import chromadb

# ---- Embeddings: FastEmbed first, ST fallback (Torch) ----
_EMBED_FAST = None
_EMBED_ST = None

def init_embedder():
    global _EMBED_FAST, _EMBED_ST
    if _EMBED_FAST or _EMBED_ST:
        return
    try:
        from fastembed import TextEmbedding  # ONNX, no Torch
        _EMBED_FAST = TextEmbedding(model_name=EMB_MODEL)  # downloads once
        print(f"[EMB] FastEmbed ready: {EMB_MODEL}")
    except Exception as e:
        print(f"[WARN] FastEmbed not available ({e}). Falling back to SentenceTransformers (may use Torch).")
        try:
            from sentence_transformers import SentenceTransformer
            _EMBED_ST = SentenceTransformer(EMB_MODEL)
            print(f"[EMB] SentenceTransformers ready: {EMB_MODEL}")
        except Exception as ee:
            raise RuntimeError(f"No embedding backend available: {ee}")

def embed_texts(texts: List[str]) -> List[List[float]]:
    init_embedder()
    if _EMBED_FAST is not None:
        # FastEmbed returns generator of vectors
        return [vec for vec in _EMBED_FAST.embed(texts)]
    else:
        # ST returns numpy array
        arr = _EMBED_ST.encode(texts, normalize_embeddings=True)
        return arr.tolist()

# ---- FS helpers ----
def ensure_dirs():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DB_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

# --------- LOADERS ----------
def normalize_spaces(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def load_text_from_pdf(path: Path) -> Iterable[Tuple[str, int]]:
    # 1) PyMuPDF
    try:
        import fitz  # pymupdf
        doc = fitz.open(str(path))
        empty_pages = 0
        for i, page in enumerate(doc):
            txt = page.get_text("text").strip()
            if txt:
                yield normalize_spaces(txt), i + 1
            else:
                empty_pages += 1
        doc.close()
        if empty_pages == i + 1:
            print(f"[HINT] '{path.name}' may be scanned (no text). Try OCR (ocrmypdf).")
        return
    except Exception:
        pass
    # 2) pypdf fallback
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        empty_pages = 0
        for i, page in enumerate(reader.pages):
            try:
                raw = page.extract_text() or ""
            except Exception:
                raw = ""
            txt = normalize_spaces(raw)
            if txt:
                yield txt, i + 1
            else:
                empty_pages += 1
        if empty_pages == i + 1:
            print(f"[HINT] '{path.name}' may be scanned (no extractable text). Try OCR.")
    except Exception as e:
        print(f"[WARN] Failed to read {path.name}: {e}")

def load_text_from_md_txt(path: Path) -> str:
    try:
        raw = path.read_text(errors="ignore")
    except Exception:
        raw = ""
    return normalize_spaces(raw)

def chunk(text: str, max_chars: int = CHUNK_CHARS, overlap: int = CHUNK_OVERLAP) -> Iterable[str]:
    n = len(text)
    if n <= max_chars:
        if n > 0: yield text
        return
    i = 0
    while i < n:
        j = min(i + max_chars, n)
        yield text[i:j]
        i = j - overlap if j < n else j

def iter_documents() -> Iterable[Dict[str, Any]]:
    for f in DATA_DIR.rglob("*"):
        if not f.is_file(): continue
        ext = f.suffix.lower()
        rel = f.relative_to(BASE_DIR).as_posix()
        if ext == ".pdf":
            any_text = False
            for page_text, page in load_text_from_pdf(f):
                any_text = True
                for c in chunk(page_text):
                    yield {"id": str(uuid.uuid4()), "text": c, "meta": {"source": rel, "page": page}}
            if not any_text:
                yield {"id": str(uuid.uuid4()), "text": f"[NO-TEXT] {f.name}", "meta": {"source": rel, "page": None}}
        elif ext in (".md", ".txt"):
            text = load_text_from_md_txt(f)
            for c in chunk(text):
                yield {"id": str(uuid.uuid4()), "text": c, "meta": {"source": rel, "page": None}}

# --------- DB ----------
def get_collection(reset: bool = False):
    _lazy_imports()
    client = chromadb.PersistentClient(path=str(DB_DIR))
    if reset:
        try: client.delete_collection("alloymind")
        except Exception: pass
    # NOTE: no embedding_function here; we provide vectors ourselves
    return client.get_or_create_collection(name="alloymind")

def add_batch(col, ids, docs, metas):
    embs = embed_texts(docs)
    col.add(ids=ids, documents=docs, metadatas=metas, embeddings=embs)

def build_index(batch_size: int = 256) -> int:
    ensure_dirs()
    col = get_collection(reset=True)
    ids, docs, metas = [], [], []
    total = 0
    for doc in iter_documents():
        if doc["text"].startswith("[NO-TEXT]"):
            print(f"[INFO] Skipping unextractable file: {doc['meta']['source']}")
            continue
        ids.append(doc["id"]); docs.append(doc["text"]); metas.append(doc["meta"])
        if len(ids) >= batch_size:
            add_batch(col, ids, docs, metas)
            total += len(ids); ids, docs, metas = [], [], []
    if ids:
        add_batch(col, ids, docs, metas)
        total += len(ids)
    return total

def search(query: str, k: int = DEFAULT_TOPK) -> List[Tuple[str, str]]:
    col = get_collection(reset=False)
    # Embed query with the same backend
    qvec = embed_texts([query])[0]
    res = col.query(query_embeddings=[qvec], n_results=k, include=["documents", "metadatas"])
    hits = []
    for doc, meta in zip(res.get("documents", [[]])[0], res.get("metadatas", [[]])[0]):
        src = meta.get("source", "unknown")
        page = meta.get("page")
        cite = f"{src}" + (f":p.{page}" if page else "")
        hits.append((doc, cite))
    return hits

# --------- BACKUP / INCREMENTAL / EXPORT ----------
def backup_index():
    ensure_dirs()
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    src = DB_DIR
    dst = BASE_DIR / "index" / f"chroma_backup_{ts}"
    if not src.exists():
        print(f"[INFO] Nothing to backup, {src} does not exist."); return
    print(f"[BACKUP] Copying {src} -> {dst} ...")
    shutil.copytree(src, dst)
    if MANIFEST_PATH.exists():
        shutil.copy2(MANIFEST_PATH, MANIFEST_PATH.with_name(f"manifest_{ts}.json"))
    print(f"[BACKUP] Done. Saved to: {dst}")

def file_sig(path: Path):
    h = hashlib.sha1()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1<<20), b""): h.update(chunk)
    except Exception: return None
    stat = path.stat()
    return {"sha1": h.hexdigest(), "size": stat.st_size, "mtime": int(stat.st_mtime)}

def load_manifest():
    if MANIFEST_PATH.exists():
        try: return json.loads(MANIFEST_PATH.read_text())
        except Exception: return {}
    return {}

def save_manifest(m): MANIFEST_PATH.write_text(json.dumps(m, indent=2))

def update_index() -> None:
    ensure_dirs()
    col = get_collection(reset=False)
    manifest = load_manifest()
    current = {f.relative_to(BASE_DIR).as_posix(): f for f in DATA_DIR.rglob("*") if f.is_file()}

    # remove deleted
    for src in list(manifest.keys()):
        if src not in current:
            col.delete(where={"source": src})
            manifest.pop(src, None)
            print(f"[DEL] {src}")

    # add/refresh changed
    for src, path in current.items():
        sig = file_sig(path)
        if sig is None: continue
        if manifest.get(src) == sig: continue
        col.delete(where={"source": src})
        added = 0
        ext = path.suffix.lower()
        if ext == ".pdf":
            any_text = False
            for page_text, page in load_text_from_pdf(path):
                any_text = True
                for c in chunk(page_text):
                    add_batch(col, [str(uuid.uuid4())], [c], [{"source": src, "page": page}])
                    added += 1
            if not any_text: print(f"[INFO] Skipping unextractable file: {src}")
        elif ext in (".md", ".txt"):
            text = load_text_from_md_txt(path)
            for c in chunk(text):
                add_batch(col, [str(uuid.uuid4())], [c], [{"source": src, "page": None}])
                added += 1
        manifest[src] = sig
        print(f"[UPD] {src} (+{added} chunks)")
    save_manifest(manifest)
    print("[UPDATE] Done.")

def export_jsonl(out_path: Path, include_embeddings: bool=False):
    ensure_dirs()
    _lazy_imports()
    col = get_collection(reset=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    batch = 1000
    with open(out_path, "w", encoding="utf-8") as f:
        offset = 0
        while True:
            res = col.get(limit=batch, offset=offset, include=["documents","metadatas"]+(["embeddings"] if include_embeddings else []))
            ids = res.get("ids") or []
            if not ids: break
            for i in range(len(ids)):
                rec = {"id": ids[i], "document": res["documents"][i], "metadata": res["metadatas"][i]}
                if include_embeddings: rec["embedding"] = res["embeddings"][i]
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            offset += batch
    print(f"[EXPORT] Wrote JSONL to {out_path}")

# --------- GENERATION ----------
PROMPT_TEMPLATE = """You are AlloyMind Assistant. Use the provided context snippets to answer the user question.
Rules:
- Cite sources inline like [1], [2], etc., mapping to the "Citations" list I will provide.
- If the context is insufficient for a claim, say what’s missing and ask for that constraint.
- Keep units consistent and correct. If units are uncertain, state assumptions.
- Be concise but explain trade-offs clearly.

Question:
{question}

Context:
{context}

Citations:
{citations}

Answer:"""

def format_context(hits: List[Tuple[str, str]]) -> Tuple[str, str]:
    blocks, cites = [], []
    for i, (text, cite) in enumerate(hits, 1):
        snippet = textwrap.shorten(text.replace("\n", " "), width=450, placeholder=" ...")
        blocks.append(f"[{i}] {snippet}")
        cites.append(f"[{i}] {cite}")
    return "\n".join(blocks), "\n".join(cites)

def call_ollama(model: str, prompt: str) -> str:
    try:
        out = subprocess.run(["ollama", "run", model, prompt], check=True, capture_output=True, text=True)
        return out.stdout.strip()
    except FileNotFoundError:
        return ("[Error] Ollama not found. Install: brew install ollama\n"
                "Then run: ollama serve &\n"
                f"And pull a model: ollama pull {model}")
    except subprocess.CalledProcessError as e:
        return f"[Error] ollama run failed: {e.stderr.strip() or e.stdout.strip()}"

def answer_with_context(question: str, model: str, k: int, show_context: bool = False) -> str:
    hits = search(question, k=k)
    if not hits:
        return "No context found. Add PDFs/notes under 'sources/' and run with --rebuild or --update."
    ctx, cites = format_context(hits)
    prompt = PROMPT_TEMPLATE.format(question=question, context=ctx, citations=cites)
    if show_context:
        print("\n--- RETRIEVED CONTEXT ---"); print(ctx)
        print("\n--- CITATIONS ---"); print(cites)
        print("\n--- MODEL OUTPUT ---")
    return call_ollama(model, prompt)

# --------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="AlloyMind RAG (Torch-free embeddings)")
    ap.add_argument("--rebuild", action="store_true", help="Rebuild index from PDFs/notes in 'sources/'")
    ap.add_argument("--update", action="store_true", help="Incrementally index only new/changed files")
    ap.add_argument("--backup", action="store_true", help="Backup the current vector index folder")
    ap.add_argument("--export-json", type=str, help="Export collection to JSONL at this path")
    ap.add_argument("--ask", type=str, help="Retrieve top-k chunks with citations (no LLM generation)")
    ap.add_argument("--answer", type=str, help="Retrieve + compose answer with Ollama")
    ap.add_argument("--k", type=int, default=DEFAULT_TOPK, help="Top-k chunks")
    ap.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Ollama model for --answer")
    ap.add_argument("--show", action="store_true", help="Print retrieved context")
    args = ap.parse_args()

    # Single-process guard
    try:
        from filelock import FileLock
        lock_path = BASE_DIR / ".rag_lock"
        with FileLock(str(lock_path), timeout=1):
            run_cli(args)
    except Exception:
        run_cli(args)

def run_cli(args):
    ensure_dirs()

    if args.backup:
        backup_index()

    if args.rebuild:
        total = build_index()
        print(f"Indexed {total} chunks from {DATA_DIR}")
        if not (args.ask or args.answer or args.update or args.export_json):
            print("\nIf you saw [HINT] messages, run OCR on those PDFs:")
            print("  brew install ocrmypdf tesseract")
            print("  ocrmypdf input.pdf output_ocr.pdf")
            return

    if args.update:
        update_index()
        if not (args.ask or args.answer or args.export_json):
            return

    if args.export_json:
        export_jsonl(Path(args.export_json), include_embeddings=False)
        if not (args.ask or args.answer):
            return

    if args.ask:
        hits = search(args.ask, k=args.k)
        if not hits:
            print("No results. Put files in 'sources/' and run --rebuild or --update.")
            return
        for i, (text, cite) in enumerate(hits, 1):
            print(f"[{i}] {cite}\n{textwrap.shorten(text.replace(chr(10), ' '), width=600, placeholder=' ...')}\n")
        if not args.answer:
            return

    if args.answer:
        output = answer_with_context(args.answer, model=args.model, k=args.k, show_context=args.show)
        print(output)

    if not (args.rebuild or args.update or args.backup or args.export_json or args.ask or args.answer):
        print(__doc__)

if __name__ == "__main__":
    main()
