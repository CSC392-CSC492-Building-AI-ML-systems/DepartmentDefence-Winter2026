"""Optional LLM-as-judge scoring for stack evaluation."""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from rag.rag_types import Chunk

from .cases import case_forbidden_claims, case_required_claims
from .common import try_parse_json_object


def judge_answer(
    client,
    case: Dict[str, Any],
    answer: str,
    retrieved: List[Tuple["Chunk", float]],
    judge_model: str,
    reference_answer: Optional[str],
) -> Tuple[Optional[Dict[str, Any]], Optional[str], float]:
    required_claims = case_required_claims(case)
    forbidden_claims = case_forbidden_claims(case)

    snippets = []
    for chunk, score in retrieved[:6]:
        text = chunk.text.replace("\n", " ").strip()
        snippets.append(f"- {chunk.chunk_id} (score={score:.3f}): {text[:600]}")
    evidence_text = "\n".join(snippets) if snippets else "(none)"

    prompt = (
        "You are evaluating a policy RAG answer.\n"
        "Use ONLY the question, answer, and evidence snippets below.\n"
        "Return ONLY JSON with this schema:\n"
        "{"
        "\"decision_correctness\": 0|1|2, "
        "\"groundedness\": 0|1|2, "
        "\"completeness\": 0|1|2, "
        "\"mandatory_optional_precision\": 0|1|2, "
        "\"uncertainty_handling\": 0|1|2, "
        "\"reference_alignment\": 0|1|2, "
        "\"required_claim_recall\": number, "
        "\"forbidden_claim_violation\": 0|1, "
        "\"required_claim_checks\": [{\"claim\":\"...\",\"covered\":0|1,\"reason\":\"...\"}], "
        "\"forbidden_claim_checks\": [{\"claim\":\"...\",\"violated\":0|1,\"reason\":\"...\"}], "
        "\"notes\": \"short rationale\""
        "}\n\n"
        "Scoring guide:\n"
        "- 2 = strong/correct\n"
        "- 1 = partial/mixed\n"
        "- 0 = wrong/unsupported\n"
        "- required_claim_recall in [0,1]\n"
        "- forbidden_claim_violation: 1 if answer asserts forbidden claim meaningfully\n\n"
        "- reference_alignment: how well answer matches reference answer policy meaning\n\n"
        "Coverage rules:\n"
        "- Mark a required claim covered only if the answer states it with policy-faithful meaning.\n"
        "- Prefer claims that are explicitly supported by the provided evidence snippets.\n"
        "- Mark forbidden claim violated when the answer clearly asserts that meaning.\n\n"
        f"Question:\n{case['question']}\n\n"
        f"Required claims:\n{json.dumps(required_claims)}\n\n"
        f"Forbidden claims:\n{json.dumps(forbidden_claims)}\n\n"
        f"Expect abstain:\n{json.dumps(case.get('expect_abstain'))}\n\n"
        f"Reference answer:\n{(reference_answer or '(none provided)')}\n\n"
        f"Answer:\n{answer}\n\n"
        f"Evidence snippets:\n{evidence_text}\n"
    )

    t0 = time.perf_counter()
    try:
        resp = client.chat(
            model=judge_model,
            message=prompt,
            temperature=0.0,
            max_tokens=420,
            response_format={"type": "json_object"},
        )
        raw = (resp.text or "").strip()
        payload = try_parse_json_object(raw)

        score_keys = [
            "decision_correctness",
            "groundedness",
            "completeness",
            "mandatory_optional_precision",
            "uncertainty_handling",
        ]
        if reference_answer:
            score_keys.append("reference_alignment")

        for key in score_keys:
            value = int(payload.get(key, 0))
            payload[key] = max(0, min(2, value))
        if not reference_answer:
            payload["reference_alignment"] = None

        required_checks_raw = payload.get("required_claim_checks", [])
        required_checks: List[Dict[str, Any]] = []
        if isinstance(required_checks_raw, list):
            for item in required_checks_raw[: len(required_claims)]:
                if not isinstance(item, dict):
                    continue
                claim = str(item.get("claim", "")).strip()
                covered = 1 if int(item.get("covered", 0)) else 0
                reason = str(item.get("reason", "")).strip()
                required_checks.append({"claim": claim, "covered": covered, "reason": reason})
        payload["required_claim_checks"] = required_checks

        forbidden_checks_raw = payload.get("forbidden_claim_checks", [])
        forbidden_checks: List[Dict[str, Any]] = []
        if isinstance(forbidden_checks_raw, list):
            for item in forbidden_checks_raw[: len(forbidden_claims)]:
                if not isinstance(item, dict):
                    continue
                claim = str(item.get("claim", "")).strip()
                violated = 1 if int(item.get("violated", 0)) else 0
                reason = str(item.get("reason", "")).strip()
                forbidden_checks.append({"claim": claim, "violated": violated, "reason": reason})
        payload["forbidden_claim_checks"] = forbidden_checks

        recall = float(payload.get("required_claim_recall", 0.0))
        if required_checks:
            recall = sum(int(item["covered"]) for item in required_checks) / len(required_checks)
        payload["required_claim_recall"] = min(max(recall, 0.0), 1.0)

        forbidden = int(payload.get("forbidden_claim_violation", 0))
        if forbidden_checks:
            forbidden = 1 if any(int(item["violated"]) for item in forbidden_checks) else 0
        payload["forbidden_claim_violation"] = 1 if forbidden else 0
        payload["notes"] = str(payload.get("notes", "")).strip()

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return payload, None, elapsed_ms
    except Exception as exc:  # noqa: BLE001
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return None, str(exc), elapsed_ms
