import json
import os
from datetime import datetime, timezone
from pathlib import Path

from flask import Blueprint, jsonify

from .auth import require_dashboard_secret


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
        "last_updated_ts": int(datetime.now(tz=timezone.utc).timestamp()),
        "recent_negative_feedback": [],
        "top_issue_buckets": [],
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
    return {
        "timestamp": _to_int_timestamp(record.get("timestamp")),
        "thumb": str(record.get("thumb", "")).strip().lower(),
        "comment": str(record.get("comment", "")).strip(),
        "conversation_id": str(record.get("conversation_id", "")).strip(),
        "turn_id": str(record.get("turn_id", "")).strip(),
        "question": str(record.get("question", "")).strip(),
    }


# Map free-text feedback comments to coarse issue buckets.
def _bucket_issue(comment: str) -> str | None:
    text = comment.lower()
    if not text:
        return None
    keyword_buckets = {
        "citations": ["citation", "source", "evidence", "reference", "no citation"],
        "accuracy": ["wrong", "incorrect", "inaccurate", "false", "mistake"],
        "clarity": ["unclear", "confusing", "vague", "hard to understand"],
        "coverage": ["missing", "didn't answer", "not answer", "incomplete", "partial"],
        "latency": ["slow", "delay", "long", "lag", "time"],
    }
    for bucket, keywords in keyword_buckets.items():
        if any(keyword in text for keyword in keywords):
            return bucket
    return "other"


# Aggregate feedback JSONL into counts, rates, and recent issues.
def _load_feedback_summary() -> dict:
    summary = _feedback_template()
    if not FEEDBACK_JSONL_PATH.exists() or not FEEDBACK_JSONL_PATH.is_file():
        return summary

    now_ts = int(datetime.now(tz=timezone.utc).timestamp())
    day_ago = now_ts - (24 * 60 * 60)
    week_ago = now_ts - (7 * 24 * 60 * 60)

    issue_counts: dict[str, int] = {}
    recent_negative: list[dict] = []

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

                thumb = record.get("thumb")
                if thumb not in {"up", "side", "down"}:
                    continue

                summary["total_count"] += 1
                summary["thumb_counts"][thumb] += 1

                ts = record.get("timestamp")
                if isinstance(ts, int):
                    if ts >= day_ago:
                        summary["count_last_24h"] += 1
                    if ts >= week_ago:
                        summary["count_last_7d"] += 1

                if thumb in {"side", "down"}:
                    recent_negative.append(record)
                    bucket = _bucket_issue(record.get("comment", ""))
                    if bucket:
                        issue_counts[bucket] = issue_counts.get(bucket, 0) + 1
    except OSError:
        return summary

    total = summary["total_count"]
    if total > 0:
        for thumb in ("up", "side", "down"):
            summary["thumb_rates"][thumb] = round(summary["thumb_counts"][thumb] / total, 4)
        summary["attention_rate"] = round(
            (summary["thumb_counts"]["side"] + summary["thumb_counts"]["down"]) / total,
            4,
        )

    recent_negative_sorted = sorted(
        recent_negative,
        key=lambda item: item.get("timestamp") or 0,
        reverse=True,
    )[:10]
    summary["recent_negative_feedback"] = recent_negative_sorted
    if recent_negative_sorted and recent_negative_sorted[0].get("timestamp"):
        summary["last_updated_ts"] = recent_negative_sorted[0]["timestamp"]

    summary["top_issue_buckets"] = [
        {"bucket": bucket, "count": count}
        for bucket, count in sorted(issue_counts.items(), key=lambda item: item[1], reverse=True)[:5]
    ]
    return summary


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
