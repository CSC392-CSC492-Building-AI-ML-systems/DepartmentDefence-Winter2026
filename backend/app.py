from flask import Flask, request, jsonify
from flask_cors import CORS
from pathlib import Path
import json
import time
# Import your RAG functions directly:
from rag.app_config import (
    CHAT_MAX_INPUT_TOKENS, CHAT_MAX_OUTPUT_TOKENS, CHAT_MODEL, CHAT_PREAMBLE,
    ENABLE_CONTRADICTION_ANALYSIS, MAX_HISTORY_TURNS, TOP_K
)
from rag.contradiction import analyze_conflicts, build_conflict_prompt_section
from rag.corpus import list_docs
from rag.embedding_client import create_client
from rag.pipeline import embed_chunks, load_chunks_from_docs
from rag.prompting import pack_retrieved_documents
from rag.query_rewrite import generate_query_expansions
from rag.retrieval import retrieve
from rag.self_rag import generate_answer_with_critique_loop
from dashboard.eval_api import dashboard_bp

app = Flask(__name__)
CORS(app)
app.register_blueprint(dashboard_bp)

docs = None
chunks = None
chunk_vecs = None
client = None
chat_history = []
TOP_CITATIONS = 3
backup_weights: dict = {}         # conversation_id -> {turn_id: {chunk_id: weight}}
latest_feedback: dict = {}        # conversation_id -> feedback record
feedback_weights: dict = {}       # conversation_id -> {chunk_id: weight}

def load_rag_pipeline():
    global docs, chunks, chunk_vecs, client
    docs = list_docs()
    if not docs:
        raise RuntimeError("No docs found. Put .txt/.md files into data/")
    
    chunks = load_chunks_from_docs(docs)
    client = create_client()
    chunk_vecs = embed_chunks(client, chunks)
    print(f"Loaded {len(docs)} docs -> {len(chunks)} chunks")


load_rag_pipeline()  # Load everything at startup


@app.route('/api/chat', methods=['POST'])
def chat():
    global chat_history, latest_feedback, feedback_weights
    
    data = request.get_json()
    q = data.get('message', '').strip()
    conversation_id = str(data.get("conversation_id", "default")).strip() or "default"
    
    if not q:
        return jsonify({'reply': 'Please enter a question.'}), 400
    
    try:
        # Your exact RAG logic from main()
        query_expansions = generate_query_expansions(
            client=client, question=q, chat_history=chat_history
        )
        retrieved = retrieve(
            client=client, query=q, chunks=chunks, chunk_vecs=chunk_vecs,
            k=TOP_K, query_expansions=query_expansions
        )
        # Apply feedback weights to retrieval scores (session-local).
        if conversation_id in feedback_weights:
            weight_map = feedback_weights[conversation_id]
            reweighted = []
            for chunk, score in retrieved:
                w = weight_map.get(chunk.chunk_id, 1.0)
                reweighted.append((chunk, score * w))
            retrieved = sorted(reweighted, key=lambda x: x[1], reverse=True)

        packed_docs, packing_stats = pack_retrieved_documents(
            client=client, question=q, retrieved=retrieved, preamble=CHAT_PREAMBLE,
            chat_history=chat_history, max_input_tokens=CHAT_MAX_INPUT_TOKENS
        )

        if not packed_docs:
            return jsonify({'reply': 'No context docs fit. Try a shorter question.'})

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

        feedback_note = ""
        fb = latest_feedback.get(conversation_id)
        if fb and fb.get("thumb") in {"down", "side"}:
            reason = fb.get("comment", "").strip()
            thumb_text = "thumbs down" if fb["thumb"] == "down" else "thumbs sideways"
            detail = f" Reason: {reason}" if reason else ""
            feedback_note = f"\n\nFEEDBACK:\nPrevious answer was rated {thumb_text}.{detail}\nEmphasize accuracy and cite explicit values/clauses."
            chat_message = chat_message + feedback_note

        answer, self_rag_meta = generate_answer_with_critique_loop(
            client=client,
            chat_model=CHAT_MODEL,
            preamble=CHAT_PREAMBLE,
            question=q,
            chat_message=chat_message,
            documents=packed_docs,
            chat_history=chat_history,
            max_tokens=CHAT_MAX_OUTPUT_TOKENS,
            max_input_tokens=CHAT_MAX_INPUT_TOKENS,
        )
        
        # Update history (same as main())
        chat_history.extend([{"role": "USER", "message": q}, {"role": "CHATBOT", "message": answer}])
        max_messages = max(0, MAX_HISTORY_TURNS * 2)
        if max_messages and len(chat_history) > max_messages:
            chat_history = chat_history[-max_messages:]

        citations = []
        unique_links = set()

        packed_ids = {doc["chunk_id"] for doc in packed_docs}

        for ch in packed_docs:
            link = ch.get('source_url', 'Unknown URL')
            if link not in unique_links and link not in ('Unknown URL', ''):
                citations.append({
                    'title': ch.get('source_title', ch.get('title', 'Unknown Title')),
                    'link': link,
                    'chunk_id': ch.get('chunk_id')
                })
                unique_links.add(link)
            if len(unique_links) >= TOP_CITATIONS:
                break

        # Frontend expects this format
        return jsonify({
            'reply': answer,
            'citations': citations,
            'stats': {
                'retrieved': len(retrieved),
                'packed_docs': packing_stats['packed_docs'],
                'conflict_pairs_analyzed': len(conflicts),
                'self_rag_revision_applied': bool(self_rag_meta.get('revision_applied')),
                'self_rag_unsupported_claims': int(self_rag_meta.get('unsupported_claim_count', 0)),
                'self_rag_missing_citations': int(self_rag_meta.get('missing_citation_count', 0)),
            }
        })
    
    except Exception as e:
        return jsonify({'reply': f'Error: {str(e)}'}), 500

@app.route('/health')
def health():
    return jsonify({'status': 'OK', 'chunks_loaded': len(chunks) if chunks else 0})


def _append_feedback(record: dict) -> None:
    """Persist feedback as JSONL under backend/data/feedback/feedback.jsonl."""
    feedback_dir = Path("data/feedback")
    feedback_dir.mkdir(parents=True, exist_ok=True)
    out_path = feedback_dir / "feedback.jsonl"
    line = json.dumps(record, ensure_ascii=False)
    with out_path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


@app.route('/api/feedback', methods=['POST'])
def feedback():
    """Collect thumbs feedback for a conversation turn."""
    global latest_feedback, feedback_weights, backup_weights
    data = request.get_json(force=True, silent=True) or {}
    thumb = str(data.get("thumb", "")).strip().lower()

    if thumb not in {"up", "side", "down", "none"}:
        return jsonify({"error": "thumb must be one of: up, side, down, none"}), 400

    conversation_id = str(data.get("conversation_id", "default")).strip() or "default"
    turn_id = str(data.get("turn_id", "")).strip()
    cited_chunks = data.get("cited_chunk_ids", [])

    record = {
        "timestamp": int(time.time()),
        "conversation_id": conversation_id,
        "turn_id": turn_id,
        "thumb": thumb,
        "comment": str(data.get("comment", "")).strip(),
        "question": str(data.get("question", "")).strip(),
        "answer": str(data.get("answer", "")).strip(),
        "cited_chunk_ids": cited_chunks,
    }
    _append_feedback(record)

    # Access weight maps
    weight_map = feedback_weights.setdefault(conversation_id, {})
    turn_backups = backup_weights.setdefault(conversation_id, {}).setdefault(turn_id, {})

    # Revert to the pre-turn snapshot first (if we have one)
    # This wipes clean any previous clicks on this specific message.
    for cid, old_val in turn_backups.items():
        weight_map[cid] = old_val

    # Handle 'none' (Unchecking)
    if thumb == "none":
        # We already reverted the weights, so just clear the prompt warning
        latest_feedback.pop(conversation_id, None)
    # Handle actual feedback
    else:
        latest_feedback[conversation_id] = record
        factor = 1.05 if thumb == "up" else 0.95 if thumb == "side" else 0.8
        
        for cid in cited_chunks:
            current = weight_map.get(cid, 1.0)
            # Take a snapshot if this is the first time voting on this turn
            if cid not in turn_backups:
                turn_backups[cid] = current
            # Apply the math
            weight_map[cid] = max(0.2, min(2.0, current * factor))

    return jsonify({"status": "ok"})

if __name__ == '__main__':
    app.run(debug=True, port=5001)
