from flask import Flask, request, jsonify
from flask_cors import CORS
from pathlib import Path
import json
import time
# Import your RAG functions directly:
from rag.app_config import (
    CHAT_MAX_INPUT_TOKENS, CHAT_MAX_OUTPUT_TOKENS, CHAT_MODEL, CHAT_PREAMBLE,
    MAX_HISTORY_TURNS, TOP_K
)
from rag.corpus import list_docs
from rag.embedding_client import create_client
from rag.pipeline import embed_chunks, load_chunks_from_docs
from rag.prompting import pack_retrieved_documents
from rag.query_rewrite import generate_query_expansions
from rag.retrieval import retrieve

app = Flask(__name__)
CORS(app)  


docs = None
chunks = None
chunk_vecs = None
client = None
chat_history = []
TOP_CITATIONS = 3
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
        packed_docs, packing_stats = pack_retrieved_documents(
            client=client, question=q, retrieved=retrieved, preamble=CHAT_PREAMBLE,
            chat_history=chat_history, max_input_tokens=CHAT_MAX_INPUT_TOKENS
        )
        
        if not packed_docs:
            return jsonify({'reply': 'No context docs fit. Try a shorter question.'})
        
        chat_message = (
            f"{q}\n\n"
            "Instructions: Use only the provided documents. "
            "Cite CHUNK_ID values in square brackets for every policy claim. "
            "If evidence is missing, explicitly say so."
        )
        # Apply feedback weights to retrieval scores (session-local).
        if conversation_id in feedback_weights:
            weight_map = feedback_weights[conversation_id]
            reweighted = []
            for chunk, score in retrieved:
                w = weight_map.get(chunk.chunk_id, 1.0)
                reweighted.append((chunk, score * w))
            retrieved = sorted(reweighted, key=lambda x: x[1], reverse=True)

        feedback_note = ""
        fb = latest_feedback.get(conversation_id)
        if fb and fb.get("thumb") in {"down", "side"}:
            reason = fb.get("comment", "").strip()
            thumb_text = "thumbs down" if fb["thumb"] == "down" else "thumbs sideways"
            detail = f" Reason: {reason}" if reason else ""
            feedback_note = f"\n\nFEEDBACK:\nPrevious answer was rated {thumb_text}.{detail}\nEmphasize accuracy and cite explicit values/clauses."
            chat_message = chat_message + feedback_note

        resp = client.chat(
            model=CHAT_MODEL, preamble=CHAT_PREAMBLE, message=chat_message,
            documents=packed_docs, chat_history=chat_history, temperature=0.2,
            max_tokens=CHAT_MAX_OUTPUT_TOKENS, max_input_tokens=CHAT_MAX_INPUT_TOKENS,
            citation_quality="off", prompt_truncation="AUTO_PRESERVE_ORDER"
        )
        answer = (resp.text or "").strip()
        
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
                'packed_docs': packing_stats['packed_docs']
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
    global latest_feedback, feedback_weights
    data = request.get_json(force=True, silent=True) or {}
    thumb = str(data.get("thumb", "")).strip().lower()
    if thumb not in {"up", "side", "down"}:
        return jsonify({"error": "thumb must be one of: up, side, down"}), 400

    conversation_id = str(data.get("conversation_id", "default")).strip() or "default"

    record = {
        "timestamp": int(time.time()),
        "conversation_id": conversation_id,
        "turn_id": str(data.get("turn_id", "")).strip(),
        "thumb": thumb,
        "comment": str(data.get("comment", "")).strip(),
        "question": str(data.get("question", "")).strip(),
        "answer": str(data.get("answer", "")).strip(),
        "cited_chunk_ids": data.get("cited_chunk_ids", []),
    }
    _append_feedback(record)
    latest_feedback[conversation_id] = record

    # Update per-conversation weights for cited chunks.
    if record["cited_chunk_ids"]:
        weight_map = feedback_weights.setdefault(conversation_id, {})
        factor = 1.05 if thumb == "up" else 0.95 if thumb == "side" else 0.8
        for cid in record["cited_chunk_ids"]:
            current = weight_map.get(cid, 1.0)
            weight_map[cid] = max(0.2, min(2.0, current * factor))

    return jsonify({"status": "ok"})

if __name__ == '__main__':
    app.run(debug=True, port=5001)
