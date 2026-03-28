from flask import Flask, request, jsonify
from flask_cors import CORS
from pathlib import Path
import json
import time
import sqlite3
# Import your RAG functions directly:
from rag.app_config import (
    CHAT_MAX_INPUT_TOKENS, CHAT_MAX_OUTPUT_TOKENS, CHAT_MODEL, CHAT_PREAMBLE,
    ENABLE_CONTRADICTION_ANALYSIS, MAX_HISTORY_TURNS, TOP_K
)
from rag.contradiction import analyze_conflicts, build_conflict_prompt_section
from rag.corpus import list_docs
from rag.embedding_client import create_client
from rag.intent_gate import build_intent_reply, classify_message_intent
from rag.pipeline import embed_chunks, load_chunks_from_docs
from rag.prompting import pack_retrieved_documents
from rag.query_rewrite import generate_query_expansions
from rag.retrieval import retrieve
from rag.self_rag import generate_answer_with_critique_loop
from dashboard.eval_api import dashboard_bp

app = Flask(__name__)
CORS(app)
app.register_blueprint(dashboard_bp)

BACKEND_DIR = Path(__file__).resolve().parent
FEEDBACK_DIR = BACKEND_DIR / "data" / "feedback"
DB_PATH = BACKEND_DIR / "dummy_database.db"


def get_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            language TEXT DEFAULT 'en'
        );
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            title TEXT,
            created_at REAL DEFAULT (strftime('%s', 'now')),
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER NOT NULL,
            type TEXT NOT NULL,
            text TEXT NOT NULL,
            language TEXT DEFAULT 'en',
            citations TEXT,
            created_at REAL DEFAULT (strftime('%s', 'now')),
            FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
        );
    """)
    if not conn.execute("SELECT id FROM users WHERE username = 'admin'").fetchone():
        conn.execute("INSERT INTO users (username, password, language) VALUES (?, ?, ?)",
                     ('admin', 'admin', 'en'))
    conn.commit()
    conn.close()


init_db()


docs = None
chunks = None
chunk_vecs = None
client = None
chat_history = []
TOP_CITATIONS = 3
backup_weights: dict = {}         # conversation_id -> {feedback_target: {chunk_id: weight}}
latest_feedback: dict = {}        # conversation_id -> latest answer-level negative/side record
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


@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username', '').strip()
    password = data.get('password', '').strip()
    conn = get_db()
    user = conn.execute(
        "SELECT id, language FROM users WHERE username = ? AND password = ?",
        (username, password)
    ).fetchone()
    conn.close()
    if user:
        return jsonify({'user_id': user['id'], 'language': user['language']})
    return jsonify({'error': 'Invalid credentials'}), 401


@app.route('/api/conversations', methods=['GET'])
def get_conversations():
    user_id = request.args.get('user_id')
    conn = get_db()
    rows = conn.execute(
        "SELECT id, title FROM conversations WHERE user_id = ? ORDER BY created_at DESC",
        (user_id,)
    ).fetchall()
    conn.close()
    return jsonify([{'id': r['id'], 'title': r['title']} for r in rows])


@app.route('/api/conversations/<int:conv_id>/messages', methods=['GET'])
def get_conversation_messages(conv_id):
    conn = get_db()
    rows = conn.execute(
        "SELECT type, text, language, citations FROM messages WHERE conversation_id = ? ORDER BY created_at",
        (conv_id,)
    ).fetchall()
    conn.close()
    result = []
    for m in rows:
        msg = {'type': m['type'], 'text': m['text'], 'language': m['language']}
        if m['citations']:
            msg['citations'] = json.loads(m['citations'])
        result.append(msg)
    return jsonify(result)


@app.route('/api/conversations/<int:conv_id>', methods=['DELETE'])
def delete_conversation(conv_id):
    conn = get_db()
    conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conv_id,))
    conn.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))
    conn.commit()
    conn.close()
    return jsonify({'status': 'ok'})


@app.route('/api/language', methods=['POST'])
def set_language():
    data = request.get_json()
    user_id = data.get('user_id')
    lang = data.get('language', 'en')
    conn = get_db()
    conn.execute("UPDATE users SET language = ? WHERE id = ?", (lang, user_id))
    conn.commit()
    conn.close()
    return jsonify({'status': 'ok'})


# Build a stable key for the current feedback target so citation votes and
# answer-level votes can be reverted independently.
def _feedback_target_key(feedback_type: str, turn_id: str, cited_chunk_ids: list[str]) -> str:
    if feedback_type == "citation":
        chunk_id = cited_chunk_ids[0] if cited_chunk_ids else ""
        return f"citation::{turn_id}::{chunk_id}"
    return f"answer::{turn_id}"


# Return the same payload shape as normal chat responses when we short-circuit
# non-policy or clarification requests before running the full RAG pipeline.
def _shortcut_chat_response(reply: str, intent_route: str, conversation_id=None):
    return jsonify(
        {
            "reply": reply,
            "citations": [],
            "conversation_id": conversation_id,
            "stats": {
                "intent_route": intent_route,
                "retrieved": 0,
                "packed_docs": 0,
                "conflict_pairs_analyzed": 0,
                "self_rag_revision_applied": False,
                "self_rag_unsupported_claims": 0,
                "self_rag_missing_citations": 0,
            },
        }
    )


@app.route('/api/chat', methods=['POST'])
def chat():
    global chat_history, latest_feedback, feedback_weights
    
    data = request.get_json()
    q = data.get('message', '').strip()
    user_id = data.get('user_id')
    conversation_id = data.get("conversation_id")
    language = data.get('language', 'en').strip().lower()
    
    if not q:
        return jsonify({'reply': 'Please enter a question.'}), 400

    conn = get_db()
    if not conversation_id and user_id:
        title = q[:50] + ('...' if len(q) > 50 else '')
        cursor = conn.execute(
            "INSERT INTO conversations (user_id, title) VALUES (?, ?)",
            (user_id, title)
        )
        conversation_id = cursor.lastrowid
        conn.commit()

    if conversation_id and user_id:
        conn.execute(
            "INSERT INTO messages (conversation_id, type, text, language) VALUES (?, ?, ?, ?)",
            (conversation_id, 'user', q, language)
        )
        conn.commit()

    conv_id_str = str(conversation_id) if conversation_id else "default"
    
    try:
        # Route greetings, capability prompts, vague procurement asks, and
        # out-of-scope messages before running the full retrieval pipeline.
        intent_decision = classify_message_intent(client=client, message=q, chat_history=chat_history)
        intent_route = intent_decision["route"]
        if intent_route != "policy_question":
            shortcut_reply = build_intent_reply(
                intent_route,
                language,
                clarifying_question=intent_decision.get("clarifying_question", ""),
            )
            if conversation_id and user_id:
                conn.execute(
                    "INSERT INTO messages (conversation_id, type, text, language) VALUES (?, ?, ?, ?)",
                    (conversation_id, 'bot', shortcut_reply, language)
                )
                conn.commit()
            conn.close()
            return _shortcut_chat_response(
                reply=shortcut_reply,
                intent_route=intent_route,
                conversation_id=conversation_id,
            )

        # Your exact RAG logic from main()
        query_expansions = generate_query_expansions(
            client=client, question=q, chat_history=chat_history
        )
        retrieved = retrieve(
            client=client, query=q, chunks=chunks, chunk_vecs=chunk_vecs,
            k=TOP_K, query_expansions=query_expansions
        )
        # Apply feedback weights to retrieval scores (session-local).
        if conv_id_str in feedback_weights:
            weight_map = feedback_weights[conv_id_str]
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
        if language == 'fr':
            chat_message += (
                "\n\nIMPORTANT: You must respond entirely in French (français). "
                "Translate any policy content into French in your answer."
            )

        feedback_note = ""
        fb = latest_feedback.get(conv_id_str)
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

        if conversation_id and user_id:
            conn.execute(
                "INSERT INTO messages (conversation_id, type, text, language, citations) VALUES (?, ?, ?, ?, ?)",
                (conversation_id, 'bot', answer, language, json.dumps(citations))
            )
            conn.commit()
        conn.close()

        return jsonify({
            'reply': answer,
            'citations': citations,
            'conversation_id': conversation_id,
            'stats': {
                'retrieved': len(retrieved),
                'packed_docs': packing_stats['packed_docs'],
                'conflict_pairs_analyzed': len(conflicts),
                'self_rag_revision_applied': bool(self_rag_meta.get('revision_applied')),
                'self_rag_unsupported_claims': int(self_rag_meta.get('unsupported_claim_count', 0)),
                'self_rag_missing_citations': int(self_rag_meta.get('missing_citation_count', 0)),
                'intent_route': 'policy_question',
            }
        })
    
    except Exception as e:
        return jsonify({'reply': f'Error: {str(e)}'}), 500

@app.route('/health')
def health():
    return jsonify({'status': 'OK', 'chunks_loaded': len(chunks) if chunks else 0})


def _append_feedback(record: dict) -> None:
    """Persist feedback as JSONL under backend/data/feedback/feedback.jsonl."""
    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FEEDBACK_DIR / "feedback.jsonl"
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
    feedback_type = str(data.get("feedback_type", "citation")).strip().lower() or "citation"
    if feedback_type not in {"citation", "answer"}:
        return jsonify({"error": "feedback_type must be one of: citation, answer"}), 400

    cited_chunks = [
        str(chunk_id).strip()
        for chunk_id in data.get("cited_chunk_ids", [])
        if str(chunk_id).strip()
    ]
    target_chunk_id = cited_chunks[0] if cited_chunks else ""

    record = {
        "timestamp": int(time.time()),
        "conversation_id": conversation_id,
        "turn_id": turn_id,
        "feedback_type": feedback_type,
        "thumb": thumb,
        "comment": str(data.get("comment", "")).strip(),
        "question": str(data.get("question", "")).strip(),
        "answer": str(data.get("answer", "")).strip(),
        "cited_chunk_ids": cited_chunks,
        "target_chunk_id": target_chunk_id,
    }
    _append_feedback(record)

    if feedback_type == "citation":
        # Access weight maps for this conversation and restore the pre-vote
        # snapshot for this citation target before applying a new vote.
        weight_map = feedback_weights.setdefault(conversation_id, {})
        target_key = _feedback_target_key(feedback_type, turn_id, cited_chunks)
        target_backups = backup_weights.setdefault(conversation_id, {}).setdefault(target_key, {})

        for cid, old_val in target_backups.items():
            weight_map[cid] = old_val

        # Handle actual citation feedback by reapplying a fresh multiplier to the
        # cited chunk IDs after the reset above. A "none" vote leaves the restored
        # baseline in place.
        if thumb != "none":
            factor = 1.05 if thumb == "up" else 0.95 if thumb == "side" else 0.8
            for cid in cited_chunks:
                current = weight_map.get(cid, 1.0)
                if cid not in target_backups:
                    target_backups[cid] = current
                weight_map[cid] = max(0.2, min(2.0, current * factor))
    else:
        # Answer-level feedback does not reweight chunks. We only keep the latest
        # negative/side signal so the next answer can be nudged toward higher accuracy.
        if thumb in {"down", "side"}:
            latest_feedback[conversation_id] = record
        else:
            latest_feedback.pop(conversation_id, None)

    return jsonify({"status": "ok"})

if __name__ == '__main__':
    app.run(debug=True, port=5001)
