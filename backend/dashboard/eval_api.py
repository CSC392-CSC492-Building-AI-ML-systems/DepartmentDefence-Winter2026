import json
import os
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

from flask import Blueprint, jsonify
from .auth import require_dashboard_secret
from rag.corpus import list_docs
from rag.pipeline import load_chunks_from_docs


dashboard_bp = Blueprint("dashboard_eval", __name__)

EVAL_RUNS_DIR = Path(__file__).resolve().parent.parent / "evaluation" / "runs"
FEEDBACK_JSONL_PATH = Path(__file__).resolve().parent.parent / "data" / "feedback" / "feedback.jsonl"
EVAL_CASES_DIR = Path(__file__).resolve().parent.parent / "evaluation" / "cases"

PREFERRED_HEADLINE_METRICS = [
    "retrieval_gold_doc_recall_at_k_mean",
    "retrieval_gold_doc_precision_at_k_mean",
    "retrieval_gold_doc_mrr_mean",
    "retrieval_claim_evidence_coverage_mean",
    "retrieval_noise_rate_mean",
    "answer_required_claim_recall_mean",
    "answer_citation_support_rate_mean",
    "answer_forbidden_violation_rate",
    "answer_abstention_accuracy",
]


# Read a non-empty environment variable with a fallback default.
def _read_env_str(name: str, default: str) -> str:
    value = os.getenv(name, "").strip()
    return value or default


# Collect distinct case-mode labels from evaluation case files.
def _load_case_modes() -> list[str]:
    modes: set[str] = set()
    for path in (
        EVAL_CASES_DIR / "eval_cases.jsonl",
        EVAL_CASES_DIR / "eval_cases_reference.jsonl",
    ):
        if not path.exists() or not path.is_file():
            continue
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                raw = line.strip()
                if not raw or raw.startswith("#"):
                    continue
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                mode = str(payload.get("mode", "")).strip()
                if mode:
                    modes.add(mode)
        except OSError:
            continue
    return sorted(modes)


# Build the default feedback summary shape returned to the dashboard.
def _feedback_template() -> dict:
    return {
        "total_count": 0,
        "thumb_counts": {"up": 0, "side": 0, "down": 0},
        "thumb_rates": {"up": 0.0, "side": 0.0, "down": 0.0},
        "attention_rate": 0.0,
        "count_last_24h": 0,
        "count_last_7d": 0,
        "last_updated_ts": None,
        "recent_negative_feedback": [],
    }


# Coerce known timestamp formats into an integer epoch value.
def _to_int_timestamp(value) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return int(float(text))
        except ValueError:
            return None
    return None


# Normalize raw feedback JSONL rows to a stable schema.
def _normalize_feedback_record(record: dict) -> dict:
    if not isinstance(record, dict):
        return {}
    feedback_type = str(record.get("feedback_type", "citation")).strip().lower() or "citation"
    if feedback_type not in {"citation", "answer"}:
        feedback_type = "citation"
    raw_chunk_ids = record.get("cited_chunk_ids", [])
    if not isinstance(raw_chunk_ids, list):
        raw_chunk_ids = []
    cited_chunk_ids = [str(chunk_id).strip() for chunk_id in raw_chunk_ids if str(chunk_id).strip()]
    target_chunk_id = str(record.get("target_chunk_id", "")).strip()
    if not target_chunk_id and cited_chunk_ids:
        target_chunk_id = cited_chunk_ids[0]
    return {
        "timestamp": _to_int_timestamp(record.get("timestamp")),
        "feedback_type": feedback_type,
        "thumb": str(record.get("thumb", "")).strip().lower(),
        "comment": str(record.get("comment", "")).strip(),
        "conversation_id": str(record.get("conversation_id", "")).strip(),
        "turn_id": str(record.get("turn_id", "")).strip(),
        "question": str(record.get("question", "")).strip(),
        "answer": str(record.get("answer", "")).strip(),
        "cited_chunk_ids": cited_chunk_ids,
        "target_chunk_id": target_chunk_id,
    }


@lru_cache(maxsize=1)
def _chunk_feedback_lookup() -> dict[str, dict]:
    # Load chunk metadata once so citation feedback rows can show human-readable
    # source labels and expandable debugging details in the dashboard.
    lookup: dict[str, dict] = {}
    try:
        for chunk in load_chunks_from_docs(list_docs()):
            source_title = str(chunk.source_title or chunk.title or Path(chunk.source_path).stem).strip()
            chunk_preview = " ".join(str(chunk.text or "").split())
            lookup[chunk.chunk_id] = {
                "source_title": source_title,
                "source_url": str(chunk.source_url or "").strip(),
                "source_path": str(chunk.source_path or "").strip(),
                "section_title": str(chunk.section_title or "").strip(),
                "doc_type": str(chunk.doc_type or "").strip(),
                "authority_rank": int(chunk.authority_rank or 0),
                "chunk_preview": chunk_preview[:280],
            }
    except OSError:
        return {}
    except Exception:
        return {}
    return lookup


def _enrich_feedback_record(record: dict) -> dict:
    # Only citation feedback can be enriched from chunk metadata.
    if record.get("feedback_type") != "citation":
        return record

    chunk_id = record.get("target_chunk_id", "")
    if not chunk_id:
        return record

    chunk_meta = _chunk_feedback_lookup().get(chunk_id)
    if not chunk_meta:
        return record

    enriched = dict(record)
    enriched.update(chunk_meta)
    return enriched


# Feedback rows are stored as an event log. Collapse them to one current state
# per answer or per cited chunk so toggles do not inflate dashboard counts.
def _feedback_target_key(record: dict) -> str:
    feedback_type = record.get("feedback_type", "citation")
    conversation_id = record.get("conversation_id", "")
    turn_id = record.get("turn_id", "")
    if feedback_type == "citation":
        chunk_id = record.get("target_chunk_id", "")
        return f"citation::{conversation_id}::{turn_id}::{chunk_id}"
    return f"answer::{conversation_id}::{turn_id}"


# Aggregate feedback JSONL into counts, rates, and recent issues.
def _load_feedback_summary() -> dict:
    summaries = {
        "citation": _feedback_template(),
        "answer": _feedback_template(),
    }
    if not FEEDBACK_JSONL_PATH.exists() or not FEEDBACK_JSONL_PATH.is_file():
        return summaries

    now_ts = int(datetime.now(tz=timezone.utc).timestamp())
    day_ago = now_ts - (24 * 60 * 60)
    week_ago = now_ts - (7 * 24 * 60 * 60)

    recent_negative = {"citation": [], "answer": []}
    latest_by_target: dict[str, dict] = {}

    try:
        with FEEDBACK_JSONL_PATH.open("r", encoding="utf-8") as handle:
            for line in handle:
                raw = line.strip()
                if not raw:
                    continue
                try:
                    record = _normalize_feedback_record(json.loads(raw))
                except json.JSONDecodeError:
                    continue
                # Keep only the most recent event for each feedback target.
                target_key = _feedback_target_key(record)
                latest_by_target[target_key] = record
    except OSError:
        return summaries

    for record in latest_by_target.values():
        # Enrich citation rows with source metadata before building the summary
        # consumed by the moderator dashboard.
        record = _enrich_feedback_record(record)
        thumb = record.get("thumb")
        feedback_type = record.get("feedback_type", "citation")
        summary = summaries.get(feedback_type, summaries["citation"])
        ts = record.get("timestamp")

        if isinstance(ts, int):
            current_last = summary.get("last_updated_ts")
            if current_last is None or ts > current_last:
                summary["last_updated_ts"] = ts

        if thumb not in {"up", "side", "down"}:
            continue

        summary["total_count"] += 1
        summary["thumb_counts"][thumb] += 1

        if isinstance(ts, int):
            if ts >= day_ago:
                summary["count_last_24h"] += 1
            if ts >= week_ago:
                summary["count_last_7d"] += 1

        if thumb in {"side", "down"}:
            recent_negative[feedback_type].append(record)

    for feedback_type, summary in summaries.items():
        total = summary["total_count"]
        if total > 0:
            for thumb in ("up", "side", "down"):
                summary["thumb_rates"][thumb] = round(summary["thumb_counts"][thumb] / total, 4)
            summary["attention_rate"] = round(
                (summary["thumb_counts"]["side"] + summary["thumb_counts"]["down"]) / total,
                4,
            )

        # Keep the full negative/side history in reverse chronological order so
        # the dashboard can render a scrollable review list.
        summary["recent_negative_feedback"] = sorted(
            recent_negative[feedback_type],
            key=lambda item: item.get("timestamp") or 0,
            reverse=True,
        )

    return summaries


# Resolve case count from config when possible, otherwise from payload rows.
def _count_cases(payload: dict, config: dict) -> int:
    case_count = config.get("case_count")
    if isinstance(case_count, int):
        return case_count
    cases = payload.get("cases")
    if isinstance(cases, list):
        return len(cases)
    return 0


# Count per-run chat/judge/empty-answer errors from case rows.
def _build_error_breakdown(payload: dict) -> dict:
    cases = payload.get("cases")
    if not isinstance(cases, list):
        return {
            "total_cases": 0,
            "chat_error_count": 0,
            "judge_error_count": 0,
            "empty_answer_count": 0,
        }

    chat_error_count = 0
    judge_error_count = 0
    empty_answer_count = 0

    for case in cases:
        if not isinstance(case, dict):
            continue
        if case.get("chat_error"):
            chat_error_count += 1

        judge = case.get("judge")
        if isinstance(judge, dict) and judge.get("error"):
            judge_error_count += 1

        answer = case.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            empty_answer_count += 1

    return {
        "total_cases": len(cases),
        "chat_error_count": chat_error_count,
        "judge_error_count": judge_error_count,
        "empty_answer_count": empty_answer_count,
    }


# Extract grouped headline metrics used by the dashboard overview.
def _build_key_metrics(payload: dict) -> dict:
    overall = payload.get("overall_metrics") or {}
    return {
        "retrieval": {
            "recall_at_k": overall.get("retrieval_gold_doc_recall_at_k_mean"),
            "mrr": overall.get("retrieval_gold_doc_mrr_mean"),
            "noise_rate": overall.get("retrieval_noise_rate_mean"),
        },
        "grounding_citation": {
            "claim_evidence_coverage": overall.get("retrieval_claim_evidence_coverage_mean"),
            "citation_support_rate": overall.get("answer_citation_support_rate_mean"),
        },
        "safety": {
            "forbidden_violation_rate": overall.get("answer_forbidden_violation_rate"),
            "abstention_accuracy": overall.get("answer_abstention_accuracy"),
        },
    }


@dashboard_bp.route("/api/eval/health", methods=["GET"])
@require_dashboard_secret
def eval_health():
    return jsonify({"status": "OK"})


@dashboard_bp.route("/api/eval/feedback/summary", methods=["GET"])
@require_dashboard_secret
def get_feedback_summary():
    return jsonify(_load_feedback_summary())


@dashboard_bp.route("/api/eval/meta", methods=["GET"])
@require_dashboard_secret
def get_dashboard_meta():
    return jsonify(
        {
            "models": {
                "chat_model": _read_env_str("COHERE_CHAT_MODEL", "command-r-plus-08-2024"),
                "embed_model": _read_env_str("COHERE_EMBED_MODEL", "embed-english-v3.0"),
                "rerank_model": _read_env_str("COHERE_RERANK_MODEL", "rerank-english-v3.0"),
                "judge_model_default": _read_env_str("COHERE_CHAT_MODEL", "command-r-plus-08-2024"),
            },
            "execution_modes": ["retrieval-only", "chat", "chat+judge"],
            "case_modes": _load_case_modes(),
        }
    )


# Resolve an eval run id to a JSON artifact path if present.
def _load_run_path(run_id: str):
    if not run_id:
        return None
    path = EVAL_RUNS_DIR / f"{run_id}.json"
    if not path.exists() or not path.is_file():
        return None
    return path


# Read and parse a run artifact JSON payload.
def _read_run_payload(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    return json.loads(text)


# Find the most recently modified run artifact in evaluation/runs.
def _get_latest_run_path():
    if not EVAL_RUNS_DIR.exists():
        return None
    latest_path = None
    latest_mtime = None
    for path in EVAL_RUNS_DIR.glob("*.json"):
        try:
            stat = path.stat()
        except OSError:
            continue
        mtime = stat.st_mtime
        if latest_mtime is None or mtime > latest_mtime:
            latest_mtime = mtime
            latest_path = path
    return latest_path


@dashboard_bp.route("/api/eval/runs", methods=["GET"])
@require_dashboard_secret
def list_eval_runs():
    if not EVAL_RUNS_DIR.exists():
        return jsonify({"runs": []})

    runs_with_mtime = []
    for path in EVAL_RUNS_DIR.glob("*.json"):
        try:
            stat = path.stat()
        except OSError:
            continue

        mtime = stat.st_mtime
        created_at = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()

        run_id = path.stem
        filename = path.name
        cases_file = None
        headline = {}

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            config = payload.get("config") or {}
            cases_file = config.get("cases_file")
            case_count = _count_cases(payload, config)
            with_chat = bool(config.get("with_chat"))
            with_judge = bool(config.get("with_judge"))
            overall = payload.get("overall_metrics") or {}
            for key in PREFERRED_HEADLINE_METRICS:
                if key in overall:
                    headline[key] = overall[key]
                    if len(headline) >= 3:
                        break
        except Exception:
            cases_file = None
            headline = {}
            case_count = 0
            with_chat = False
            with_judge = False

        runs_with_mtime.append(
            (
                mtime,
                {
                    "run_id": run_id,
                    "filename": filename,
                    "created_at": created_at,
                    "cases_file": cases_file,
                    "case_count": case_count,
                    "with_chat": with_chat,
                    "with_judge": with_judge,
                    "headline": headline,
                },
            )
        )

    runs_with_mtime.sort(key=lambda item: item[0], reverse=True)
    runs = [item[1] for item in runs_with_mtime]
    return jsonify({"runs": runs})


@dashboard_bp.route("/api/eval/runs/<run_id>", methods=["GET"])
@require_dashboard_secret
def get_eval_run(run_id):
    path = _load_run_path(run_id)
    if path is None:
        return ("", 404)
    try:
        payload = _read_run_payload(path)
    except Exception:
        return jsonify({"error": "Failed to load evaluation run."}), 500
    return jsonify(payload)


@dashboard_bp.route("/api/eval/runs/latest", methods=["GET"])
@require_dashboard_secret
def get_latest_eval_run():
    path = _get_latest_run_path()
    if path is None:
        return ("", 404)
    try:
        payload = _read_run_payload(path)
    except Exception:
        return jsonify({"error": "Failed to load evaluation run."}), 500
    return jsonify(payload)


@dashboard_bp.route("/api/eval/runs/<run_id>/summary", methods=["GET"])
@require_dashboard_secret
def get_eval_run_summary(run_id):
    path = _load_run_path(run_id)
    if path is None:
        return ("", 404)
    try:
        payload = _read_run_payload(path)
    except Exception:
        return jsonify({"error": "Failed to load evaluation run."}), 500

    summary = {
        "config": payload.get("config"),
        "timing_summary_ms": payload.get("timing_summary_ms"),
        "overall_metrics": payload.get("overall_metrics"),
        "overall_metric_ci95": payload.get("overall_metric_ci95"),
        "subgroup_metrics": payload.get("subgroup_metrics"),
        "error_breakdown": _build_error_breakdown(payload),
        "key_metrics": _build_key_metrics(payload),
    }
    return jsonify(summary)
