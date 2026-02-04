import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm
import cohere

load_dotenv()

RAW_DIR = Path("data/raw")
TOP_K = int(os.getenv("TOP_K", "6"))
CHUNK_CHARS = int(os.getenv("CHUNK_CHARS", "1200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")
CHAT_MODEL = os.getenv("COHERE_CHAT_MODEL", "command-r-plus")
EMBED_MODEL = os.getenv("COHERE_EMBED_MODEL", "embed-english-v3.0")

if not COHERE_API_KEY:
    raise RuntimeError("Missing COHERE_API_KEY. Set it in your .env file.")

client = cohere.Client(COHERE_API_KEY)

@dataclass
class Chunk:
    chunk_id: str
    title: str
    source_path: str
    text: str

def list_docs() -> List[Path]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    exts = {".txt", ".md"}
    return sorted([p for p in RAW_DIR.rglob("*") if p.is_file() and p.suffix.lower() in exts])

def chunk_text(text: str, title: str, source_path: str) -> List[Chunk]:
    text = text.replace("\r\n", "\n").strip()
    out: List[Chunk] = []
    if not text:
        return out

    i, idx = 0, 0
    while i < len(text):
        j = min(len(text), i + CHUNK_CHARS)
        piece = text[i:j].strip()
        if piece:
            out.append(
                Chunk(
                    chunk_id=f"{Path(source_path).stem}__{idx}",
                    title=title,
                    source_path=source_path,
                    text=piece,
                )
            )
            idx += 1
        if j == len(text):
            break
        i = max(0, j - CHUNK_OVERLAP)
    return out

def embed_texts(texts: List[str], input_type: str) -> np.ndarray:
    resp = client.embed(texts=texts, model=EMBED_MODEL, input_type=input_type)
    vecs = np.array(resp.embeddings, dtype=np.float32)
    # L2 normalize for cosine similarity
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    return vecs / norms

def retrieve(query: str, chunks: List[Chunk], chunk_vecs: np.ndarray, k: int) -> List[Tuple[Chunk, float]]:
    qvec = embed_texts([query], input_type="search_query")[0]
    scores = chunk_vecs @ qvec  # cosine similarity (dot product of normalized vectors)
    top_idx = np.argsort(-scores)[:k]
    return [(chunks[i], float(scores[i])) for i in top_idx]

def build_prompt(question: str, retrieved: List[Tuple[Chunk, float]]) -> str:
    ctx_blocks = []
    for ch, score in retrieved:
        ctx_blocks.append(
            f"CHUNK_ID: {ch.chunk_id}\n"
            f"TITLE: {ch.title}\n"
            f"SOURCE: {ch.source_path}\n"
            f"TEXT:\n{ch.text}\n"
        )
    context = "\n---\n".join(ctx_blocks)

    return f"""
You are a policy assistant. Answer the user's question using ONLY the provided excerpts.

Rules:
- If the excerpts do not contain enough information, say: "I don't have enough information in the provided policies to answer that."
- Do NOT use outside knowledge.
- Every factual/policy statement MUST cite at least one chunk_id in square brackets, like [chunk_id].
- Keep the answer concise.

User question:
{question}

Policy excerpts:
{context}
""".strip()

def main():
    docs = list_docs()
    if not docs:
        print("No docs found. Put 3–5 .txt/.md files into data/raw/ and re-run.")
        return

    # 1) Load + chunk
    chunks: List[Chunk] = []
    for p in docs:
        text = p.read_text(encoding="utf-8", errors="ignore")
        chunks.extend(chunk_text(text=text, title=p.stem, source_path=str(p)))

    print(f"Loaded {len(docs)} docs → {len(chunks)} chunks")

    # 2) Embed chunks
    texts = [c.text for c in chunks]
    vecs_list = []
    batch = 64
    for i in tqdm(range(0, len(texts), batch), desc="Embedding chunks"):
        vecs_list.append(embed_texts(texts[i:i+batch], input_type="search_document"))
    chunk_vecs = np.vstack(vecs_list)

    # 3) Interactive loop
    while True:
        q = input("\nAsk a policy question (or 'exit'): ").strip()
        if q.lower() in {"exit", "quit"}:
            break

        retrieved = retrieve(q, chunks, chunk_vecs, TOP_K)
        prompt = build_prompt(q, retrieved)

        resp = client.chat(model=CHAT_MODEL, message=prompt, temperature=0.2)
        answer = (resp.text or "").strip()

        print("\nANSWER:\n", answer)
        print("\nCITATIONS (retrieved excerpts):")
        for ch, score in retrieved:
            quote = ch.text.replace("\n", " ")
            quote = quote[:240] + ("…" if len(quote) > 240 else "")
            print(f"- [{ch.chunk_id}] {ch.title} (score={score:.3f})")
            print(f"  {quote}\n")

if __name__ == "__main__":
    main()
