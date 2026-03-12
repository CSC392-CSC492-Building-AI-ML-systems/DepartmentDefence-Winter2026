#!/usr/bin/env python3
"""Create an HTML comparison view for two eval run JSON files."""

from __future__ import annotations

import argparse
import json
from html import escape
from pathlib import Path
from typing import Any, Dict, List, Tuple


DEFAULT_METRICS = [
    "retrieval_gold_doc_recall_at_k_mean",
    "retrieval_gold_doc_precision_at_k_mean",
    "retrieval_gold_doc_mrr_mean",
    "retrieval_gold_doc_ndcg_mean",
    "retrieval_claim_evidence_coverage_mean",
    "retrieval_noise_rate_mean",
    "answer_required_claim_recall_mean",
    "answer_citation_support_rate_mean",
    "answer_forbidden_violation_rate",
    "answer_abstention_accuracy",
]

LOWER_IS_BETTER = {
    "retrieval_noise_rate_mean",
    "answer_forbidden_violation_rate",
}


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _label_from_path(path: Path, fallback: str) -> str:
    return path.stem or fallback


def _get_metric(run: Dict[str, Any], key: str) -> float | None:
    value = run.get("overall_metrics", {}).get(key)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _timing_values(run: Dict[str, Any]) -> Dict[str, float | None]:
    total = run.get("timing_summary_ms", {}).get("total", {}) or {}
    out: Dict[str, float | None] = {}
    for key in ("p50", "p95", "mean"):
        value = total.get(key)
        out[key] = float(value) if isinstance(value, (int, float)) else None
    return out


def _format_value(value: float | None, ndigits: int = 6) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{ndigits}f}"


def _delta_text(base: float | None, trial: float | None, lower_is_better: bool = False) -> Tuple[str, str]:
    if base is None or trial is None:
        return "N/A", "neutral"
    delta = trial - base
    sign = "+" if delta >= 0 else ""
    txt = f"{sign}{delta:.6f}"
    if abs(delta) < 1e-12:
        return txt, "neutral"
    improved = delta < 0 if lower_is_better else delta > 0
    return txt, "good" if improved else "bad"


def _metric_rows(
    left_run: Dict[str, Any],
    right_run: Dict[str, Any],
    metrics: List[str],
) -> List[Tuple[str, float | None, float | None]]:
    rows = []
    for key in metrics:
        rows.append((key, _get_metric(left_run, key), _get_metric(right_run, key)))
    return rows


def _render_bar(value: float | None, max_value: float) -> str:
    if value is None or max_value <= 0:
        return '<div class="bar-wrap"><div class="bar na"></div></div>'
    width = max(0.0, min(100.0, (value / max_value) * 100.0))
    return f'<div class="bar-wrap"><div class="bar" style="width:{width:.2f}%"></div></div>'


def build_html(
    left_label: str,
    right_label: str,
    left_run: Dict[str, Any],
    right_run: Dict[str, Any],
    metrics: List[str],
) -> str:
    rows = _metric_rows(left_run, right_run, metrics)
    numeric_values = [v for _, a, b in rows for v in (a, b) if isinstance(v, float)]
    max_metric = max(numeric_values) if numeric_values else 1.0

    left_time = _timing_values(left_run)
    right_time = _timing_values(right_run)
    max_time = max(
        [v for v in list(left_time.values()) + list(right_time.values()) if isinstance(v, float)],
        default=1.0,
    )

    metric_table_rows = []
    for key, left_val, right_val in rows:
        delta_txt, delta_cls = _delta_text(
            left_val, right_val, lower_is_better=key in LOWER_IS_BETTER
        )
        metric_table_rows.append(
            "<tr class='metric-row'>"
            f"<td><code>{escape(key)}</code></td>"
            f"<td>{_format_value(left_val)}</td>"
            f"<td>{_render_bar(left_val, max_metric)}</td>"
            f"<td>{_format_value(right_val)}</td>"
            f"<td>{_render_bar(right_val, max_metric)}</td>"
            f"<td class='delta {delta_cls}'>{delta_txt}</td>"
            "</tr>"
        )

    timing_rows = []
    for key in ("p50", "p95", "mean"):
        lv = left_time[key]
        rv = right_time[key]
        delta_txt, delta_cls = _delta_text(lv, rv, lower_is_better=True)
        timing_rows.append(
            "<tr>"
            f"<td><code>total_{key}_ms</code></td>"
            f"<td>{_format_value(lv, 3)}</td>"
            f"<td>{_render_bar(lv, max_time)}</td>"
            f"<td>{_format_value(rv, 3)}</td>"
            f"<td>{_render_bar(rv, max_time)}</td>"
            f"<td class='delta {delta_cls}'>{delta_txt}</td>"
            "</tr>"
        )

    key_cards = [
        ("Req Claim Recall", "answer_required_claim_recall_mean", False),
        ("Citation Support", "answer_citation_support_rate_mean", False),
        ("Forbidden Rate", "answer_forbidden_violation_rate", True),
        ("Total p50 (ms)", "total_p50_ms", True),
    ]
    cards_html = []
    for title, metric_key, lower_is_better in key_cards:
        if metric_key == "total_p50_ms":
            a_val = left_time["p50"]
            b_val = right_time["p50"]
            ndigits = 3
        else:
            a_val = _get_metric(left_run, metric_key)
            b_val = _get_metric(right_run, metric_key)
            ndigits = 6
        delta_txt, delta_cls = _delta_text(a_val, b_val, lower_is_better=lower_is_better)
        cards_html.append(
            "<div class='card'>"
            f"<div class='card-title'>{escape(title)}</div>"
            f"<div class='card-values'>{escape(left_label)}: {_format_value(a_val, ndigits)} | "
            f"{escape(right_label)}: {_format_value(b_val, ndigits)}</div>"
            f"<div class='card-delta {delta_cls}'>Δ {delta_txt}</div>"
            "</div>"
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Eval Run Comparison</title>
  <style>
    :root {{
      --bg: #f4f7fb;
      --surface: #ffffff;
      --text: #101827;
      --muted: #5b6472;
      --line: #d7dde8;
      --good: #1f9d55;
      --bad: #d64545;
      --neutral: #5b6472;
      --bar: #2f66d0;
      --bar-bg: #e7eefc;
    }}
    body {{
      margin: 0;
      background: radial-gradient(circle at 10% 10%, #e9efff, transparent 45%), var(--bg);
      color: var(--text);
      font-family: "SF Pro Text", "Segoe UI", -apple-system, sans-serif;
    }}
    .container {{ max-width: 1200px; margin: 28px auto; padding: 0 20px; }}
    .header {{
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 18px 20px;
      box-shadow: 0 10px 24px rgba(16, 24, 39, 0.05);
    }}
    h1, h2 {{ margin: 0 0 12px; }}
    p {{ margin: 6px 0 0; color: var(--muted); }}
    .legend {{ display: flex; flex-wrap: wrap; gap: 16px; margin-top: 10px; }}
    .legend-item {{ background: #f6f9ff; border: 1px solid #dce6ff; border-radius: 999px; padding: 6px 12px; font-size: 13px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; margin: 16px 0 20px; }}
    .card {{
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 12px;
      box-shadow: 0 6px 16px rgba(16, 24, 39, 0.05);
    }}
    .card-title {{ font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: .04em; }}
    .card-values {{ margin-top: 8px; font-size: 13px; color: var(--text); }}
    .card-delta {{ margin-top: 6px; font-weight: 600; font-size: 13px; }}
    table {{
      border-collapse: separate;
      border-spacing: 0;
      width: 100%;
      margin: 10px 0 24px;
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 12px;
      overflow: hidden;
      box-shadow: 0 10px 24px rgba(16, 24, 39, 0.05);
    }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 10px 10px; font-size: 13px; vertical-align: middle; }}
    tr:last-child td {{ border-bottom: none; }}
    th {{ background: #f8fbff; text-align: left; }}
    code {{ background: #f0f4fb; padding: 2px 6px; border-radius: 6px; }}
    .bar-wrap {{ background: var(--bar-bg); width: 100%; height: 11px; border-radius: 999px; overflow: hidden; }}
    .bar {{ height: 100%; background: linear-gradient(90deg, #76a0ff, var(--bar)); }}
    .bar.na {{ background: #d0d4da; width: 100%; opacity: .4; }}
    .delta {{ font-weight: 600; }}
    .good {{ color: var(--good); }}
    .bad {{ color: var(--bad); }}
    .neutral {{ color: var(--neutral); }}
  </style>
</head>
<body>
  <div class="container">
  <div class="header">
  <h1>Eval Run Comparison</h1>
  <div class="legend">
    <div class="legend-item"><b>Run A:</b> {escape(left_label)}</div>
    <div class="legend-item"><b>Run B:</b> {escape(right_label)}</div>
  </div>
  <p>Bars are scaled within each section (metrics vs latency). N/A means metric was not computed in that run.</p>
  </div>

  <div class="grid">
    {''.join(cards_html)}
  </div>

  <h2>Overall Metrics</h2>
  <table>
    <thead>
      <tr>
        <th>Metric</th>
        <th>{escape(left_label)} value</th>
        <th>{escape(left_label)} bar</th>
        <th>{escape(right_label)} value</th>
        <th>{escape(right_label)} bar</th>
        <th>Δ (Run B - Run A)</th>
      </tr>
    </thead>
    <tbody>
      {''.join(metric_table_rows)}
    </tbody>
  </table>

  <h2>Total Latency</h2>
  <table>
    <thead>
      <tr>
        <th>Metric</th>
        <th>{escape(left_label)} value</th>
        <th>{escape(left_label)} bar</th>
        <th>{escape(right_label)} value</th>
        <th>{escape(right_label)} bar</th>
        <th>Δ (Run B - Run A)</th>
      </tr>
    </thead>
    <tbody>
      {''.join(timing_rows)}
    </tbody>
  </table>
  </div>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize comparison between two eval run JSON files.")
    parser.add_argument("run_a", type=Path, help="First eval run JSON path.")
    parser.add_argument("run_b", type=Path, help="Second eval run JSON path.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation/runs/compare_runs.html"),
        help="HTML output path.",
    )
    parser.add_argument(
        "--metrics",
        default=",".join(DEFAULT_METRICS),
        help="Comma-separated overall metric keys to include.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_a = _load_json(args.run_a)
    run_b = _load_json(args.run_b)
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]

    label_a = _label_from_path(args.run_a, "run_a")
    label_b = _label_from_path(args.run_b, "run_b")
    html = build_html(label_a, label_b, run_a, run_b, metrics)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")
    print(f"Wrote comparison HTML: {args.output}")


if __name__ == "__main__":
    main()
