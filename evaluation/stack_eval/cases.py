"""Case loading and case-field helpers for stack evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def load_cases(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Cases file not found: {path}")

    out: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#"):
            continue
        payload = json.loads(raw)
        if "id" not in payload or "question" not in payload:
            raise ValueError(f"Case must include 'id' and 'question': {raw}")
        out.append(payload)
    return out


def case_list(case: Dict[str, Any], key: str) -> List[str]:
    return [str(item).strip() for item in case.get(key, []) if str(item).strip()]


def case_required_claims(case: Dict[str, Any]) -> List[str]:
    claims = [str(item).strip() for item in case.get("required_claims", []) if str(item).strip()]
    if claims:
        return claims

    fallback: List[str] = []
    for group in case.get("required_phrase_groups", []):
        if not group:
            continue
        fallback.append(str(group[0]).strip())
    return [claim for claim in fallback if claim]


def case_forbidden_claims(case: Dict[str, Any]) -> List[str]:
    claims = [str(item).strip() for item in case.get("forbidden_claims", []) if str(item).strip()]
    if claims:
        return claims
    return [str(item).strip() for item in case.get("forbidden_phrases", []) if str(item).strip()]
