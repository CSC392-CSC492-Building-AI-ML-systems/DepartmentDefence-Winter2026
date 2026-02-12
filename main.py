from rag.app_config import CHAT_MODEL, TOP_K
from rag.corpus import list_docs
from rag.embedding_client import create_client
from rag.pipeline import embed_chunks, load_chunks_from_docs
from rag.prompting import build_prompt
from rag.retrieval import retrieve


def main() -> None:
    docs = list_docs()
    if not docs:
        print("No docs found. Put .txt/.md files into data/ (or set RAW_DIR) and re-run.")
        return

    chunks = load_chunks_from_docs(docs)
    if not chunks:
        print("No text chunks were created from RAW_DIR. Check file contents and retry.")
        return

    print(f"Loaded {len(docs)} docs → {len(chunks)} chunks")

    client = create_client()
    chunk_vecs = embed_chunks(client, chunks)
    if chunk_vecs.size == 0:
        print("No embeddings were created. Check chunking/input data and retry.")
        return

    while True:
        q = input("\nAsk a policy question (or 'exit'): ").strip()
        if q.lower() in {"exit", "quit"}:
            break

        retrieved = retrieve(client, q, chunks, chunk_vecs, TOP_K)
        prompt = build_prompt(q, retrieved)

        resp = client.chat(model=CHAT_MODEL, message=prompt, temperature=0.2)
        answer = (resp.text or "").strip()

        print("\nANSWER:\n", answer)
        print("\nCITATIONS (retrieved excerpts):")
        for ch, score in retrieved:
            quote = ch.text.replace("\n", " ")
            print(f"- [{ch.chunk_id}] {ch.title} (score={score:.3f})")
            print(f"  {quote}\n")


if __name__ == "__main__":
    main()
