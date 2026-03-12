from rag.app_config import (
    CHAT_MAX_INPUT_TOKENS,
    CHAT_MAX_OUTPUT_TOKENS,
    CHAT_MODEL,
    CHAT_PREAMBLE,
    ENABLE_CONTRADICTION_ANALYSIS,
    MAX_HISTORY_TURNS,
    TOP_K,
)
from rag.contradiction import analyze_conflicts, build_conflict_prompt_section
from rag.corpus import list_docs
from rag.embedding_client import create_client
from rag.pipeline import embed_chunks, load_chunks_from_docs
from rag.prompting import pack_retrieved_documents
from rag.query_rewrite import generate_query_expansions
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

    print(f"Loaded {len(docs)} docs -> {len(chunks)} chunks")

    client = create_client()
    chunk_vecs = embed_chunks(client, chunks)
    if chunk_vecs.size == 0:
        print("No embeddings were created. Check chunking/input data and retry.")
        return

    chat_history: list[dict] = []

    while True:
        try:
            q = input("\nAsk a policy question (or 'exit'): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not q:
            print("Please enter a non-empty question (or 'exit').")
            continue
        if q.lower() in {"exit", "quit"}:
            break

        try:
            query_expansions = generate_query_expansions(
                client=client,
                question=q,
                chat_history=chat_history,
            )
            retrieved = retrieve(
                client=client,
                query=q,
                chunks=chunks,
                chunk_vecs=chunk_vecs,
                k=TOP_K,
                query_expansions=query_expansions,
            )
            packed_docs, packing_stats = pack_retrieved_documents(
                client=client,
                question=q,
                retrieved=retrieved,
                preamble=CHAT_PREAMBLE,
                chat_history=chat_history,
                max_input_tokens=CHAT_MAX_INPUT_TOKENS,
            )
            if not packed_docs:
                print("\nNo context docs fit in the token budget. Try a shorter question.")
                continue

            conflicts = []
            conflict_section = "Conflict detected:\n- not evaluated."
            if ENABLE_CONTRADICTION_ANALYSIS:
                conflicts = analyze_conflicts(client=client, retrieved=retrieved)
                conflict_section = build_conflict_prompt_section(conflicts)

            chat_message = (
                f"{q}\n\n"
                "Instructions: Use only the provided documents. "
                "Cite CHUNK_ID values in square brackets for every policy claim. "
                "If evidence is missing, explicitly say so. "
                "Always include a short 'Conflict detected' section in your response.\n\n"
                f"{conflict_section}"
            )
            resp = client.chat(
                model=CHAT_MODEL,
                preamble=CHAT_PREAMBLE,
                message=chat_message,
                documents=packed_docs,
                chat_history=chat_history,
                temperature=0.2,
                max_tokens=CHAT_MAX_OUTPUT_TOKENS,
                max_input_tokens=CHAT_MAX_INPUT_TOKENS,
                citation_quality="off",
                prompt_truncation="AUTO_PRESERVE_ORDER",
            )
            answer = (resp.text or "").strip()
        except Exception as exc:  # noqa: BLE001
            print(f"\nRequest failed: {exc}")
            print("Retry in a few seconds or simplify the question.")
            continue

        chat_history.extend(
            [
                {"role": "USER", "message": q},
                {"role": "CHATBOT", "message": answer},
            ]
        )
        max_messages = max(0, MAX_HISTORY_TURNS * 2)
        if max_messages and len(chat_history) > max_messages:
            chat_history = chat_history[-max_messages:]

        print("\nANSWER:\n", answer)
        print(
            "\nCONTEXT STATS:\n"
            f"- query expansions: {len(query_expansions)}\n"
            f"- retrieved chunks: {len(retrieved)}\n"
            f"- conflict pairs analyzed: {len(conflicts)}\n"
            f"- packed docs: {packing_stats['packed_docs']} / {packing_stats['retrieved_docs']}\n"
            f"- packed tokens: {packing_stats['used_doc_tokens']} / {packing_stats['budget_for_docs_tokens']}"
        )
        print("\nCITATIONS (retrieved excerpts):")
        packed_ids = {doc["chunk_id"] for doc in packed_docs}
        for ch, score in retrieved:
            if ch.chunk_id not in packed_ids:
                continue
            quote = ch.text.replace("\n", " ")
            print(f"- [{ch.chunk_id}] {ch.title} (score={score:.3f})")
            print(f"  {quote}\n")


if __name__ == "__main__":
    main()
