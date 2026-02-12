"""Shared utility helpers for stack evaluation."""

from __future__ import annotations

import json
import math
import re
from typing import Any, Dict, List, Optional, Sequence

ABSTAIN_PATTERNS = [
    "i don't have enough information",
    "i do not have enough information",
    "insufficient evidence",
    "insufficient information",
    "cannot determine",
    "not enough information",
]

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
TOKEN_RE = re.compile(r"[a-z0-9]+")


def round_float(value: Optional[float], ndigits: int = 6) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), ndigits)


def percentile(values: Sequence[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    position = (len(sorted_vals) - 1) * p
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(sorted_vals[lower])
    weight = position - lower
    return float(sorted_vals[lower] * (1 - weight) + sorted_vals[upper] * weight)


def mean(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return float(sum(values) / len(values))


def bootstrap_ci95(values: Sequence[float], samples: int = 1000) -> Optional[Dict[str, float]]:
    if not values:
        return None
    arr = [float(v) for v in values]
    n = len(arr)
    if n == 1:
        return {"low": arr[0], "high": arr[0], "n": 1}

    seed = 1337
    boot_means: List[float] = []
    for _ in range(samples):
        sample = []
        for _ in range(n):
            # Deterministic xorshift32 for reproducible CI estimates.
            seed ^= (seed << 13) & 0xFFFFFFFF
            seed ^= (seed >> 17) & 0xFFFFFFFF
            seed ^= (seed << 5) & 0xFFFFFFFF
            idx = seed % n
            sample.append(arr[idx])
        boot_means.append(sum(sample) / n)

    boot_means.sort()
    low = percentile(boot_means, 0.025)
    high = percentile(boot_means, 0.975)
    return {"low": round_float(low), "high": round_float(high), "n": n}


def split_sentences(text: str) -> List[str]:
    parts = [segment.strip() for segment in SENTENCE_SPLIT_RE.split(text) if segment.strip()]
    return [part for part in parts if len(part) >= 10]


def parse_citations(text: str) -> List[str]:
    citations: List[str] = []
    for content in re.findall(r"\[([^\]]+)\]", text):
        for part in content.split(","):
            token = part.strip()
            token = re.sub(r"^chunk_id:\s*", "", token, flags=re.IGNORECASE)
            if "__" in token:
                citations.append(token)
    deduped: List[str] = []
    seen = set()
    for item in citations:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def abstained(answer_lower: str) -> bool:
    return any(pattern in answer_lower for pattern in ABSTAIN_PATTERNS)


def try_parse_json_object(raw: str) -> Dict[str, Any]:
    raw = raw.strip()
    candidates = [raw]

    if raw.startswith("```"):
        stripped = raw.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
        candidates.append(stripped)

    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(raw[start : end + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:  # noqa: BLE001
            continue
    raise ValueError("Unable to parse judge JSON payload")
