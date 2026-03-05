import { useEffect, useMemo, useState } from "react";
import {
  checkDashboardAccess,
  getDashboardMeta,
  getFeedbackSummary,
  getRunModeLabel,
  getRuns,
  getRunSummary,
} from "./api";

const METRIC_LABELS = {
  retrieval_gold_doc_recall_at_k_mean: "Recall@k",
  retrieval_gold_doc_mrr_mean: "MRR",
  retrieval_gold_doc_precision_at_k_mean: "Precision@k",
  retrieval_claim_evidence_coverage_mean: "Evidence coverage",
  retrieval_noise_rate_mean: "Noise rate",
  answer_citation_support_rate_mean: "Citation support",
  answer_forbidden_violation_rate: "Forbidden violation",
  answer_abstention_accuracy: "Abstention accuracy",
};

const PERCENT_METRIC_KEYS = new Set([
  "retrieval_gold_doc_recall_at_k_mean",
  "retrieval_gold_doc_precision_at_k_mean",
  "retrieval_gold_doc_top1_hit_rate",
  "retrieval_claim_evidence_coverage_mean",
  "retrieval_contradiction_rate_mean",
  "retrieval_noise_rate_mean",
  "answer_required_claim_recall_mean",
  "answer_citation_support_rate_mean",
  "answer_abstention_accuracy",
  "answer_forbidden_violation_rate",
  "answer_citation_sentence_rate_mean",
  "key_recall_at_k",
  "key_noise_rate",
  "key_claim_evidence_coverage",
  "key_citation_support_rate",
  "key_forbidden_violation_rate",
  "key_abstention_accuracy",
]);

function formatMetricValue(value) {
  if (typeof value === "number") return value.toFixed(3);
  if (value === null || value === undefined) return "—";
  return String(value);
}

function formatTimestamp(value) {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return date.toLocaleString();
}

function formatPercent(value) {
  if (typeof value !== "number") return "—";
  return `${(value * 100).toFixed(1)}%`;
}

function formatMetricName(key) {
  return METRIC_LABELS[key] || key.replace(/^retrieval_|^answer_/, "").replace(/_/g, " ");
}

function formatMetricDisplay(metricKey, value) {
  if (typeof value !== "number") return formatMetricValue(value);
  if (PERCENT_METRIC_KEYS.has(metricKey)) return `${(value * 100).toFixed(1)}%`;
  return value.toFixed(3);
}

function headlineItems(headline) {
  if (!headline || typeof headline !== "object") return [];
  const preferred = [
    "retrieval_gold_doc_recall_at_k_mean",
    "retrieval_gold_doc_mrr_mean",
    "retrieval_gold_doc_precision_at_k_mean",
    "answer_citation_support_rate_mean",
    "answer_abstention_accuracy",
  ];
  const seen = new Set();
  const ordered = [];
  for (const key of preferred) {
    if (key in headline) {
      ordered.push([key, headline[key]]);
      seen.add(key);
    }
  }
  for (const [key, value] of Object.entries(headline)) {
    if (!seen.has(key)) ordered.push([key, value]);
  }
  return ordered.slice(0, 3);
}

function SummaryCards({ runs }) {
  const latest = runs[0];
  const modeLabel = getRunModeLabel(latest);

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
      <div className="bg-white border border-gray-200 rounded p-4">
        <p className="text-xs uppercase tracking-wide text-gray-500">Total runs</p>
        <p className="text-2xl font-semibold text-gray-900">{runs.length}</p>
      </div>

      <div className="bg-white border border-gray-200 rounded p-4">
        <p className="text-xs uppercase tracking-wide text-gray-500">Latest run time</p>
        <p className="text-sm font-medium text-gray-900">{formatTimestamp(latest?.created_at)}</p>
      </div>

      <div className="bg-white border border-gray-200 rounded p-4">
        <p className="text-xs uppercase tracking-wide text-gray-500">Latest case count</p>
        <p className="text-2xl font-semibold text-gray-900">{latest?.case_count ?? "—"}</p>
      </div>

      <div className="bg-white border border-gray-200 rounded p-4">
        <p className="text-xs uppercase tracking-wide text-gray-500">Latest mode</p>
        <span className="inline-flex mt-2 px-2 py-1 rounded text-xs font-medium bg-blue-50 text-blue-800 border border-blue-200">
          {modeLabel}
        </span>
      </div>
    </div>
  );
}

function KeyMetricsOverview({ summary, error }) {
  const groups = summary?.key_metrics || {};

  const groupRows = [
    {
      name: "Retrieval",
      metrics: [
        ["Recall@k", "key_recall_at_k", groups?.retrieval?.recall_at_k],
        ["MRR", "retrieval_gold_doc_mrr_mean", groups?.retrieval?.mrr],
        ["Noise rate", "key_noise_rate", groups?.retrieval?.noise_rate],
      ],
    },
    {
      name: "Grounding/Citation",
      metrics: [
        [
          "Claim evidence coverage",
          "key_claim_evidence_coverage",
          groups?.grounding_citation?.claim_evidence_coverage,
        ],
        ["Citation support rate", "key_citation_support_rate", groups?.grounding_citation?.citation_support_rate],
      ],
    },
    {
      name: "Safety",
      metrics: [
        ["Forbidden violation rate", "key_forbidden_violation_rate", groups?.safety?.forbidden_violation_rate],
        ["Abstention accuracy", "key_abstention_accuracy", groups?.safety?.abstention_accuracy],
      ],
    },
  ];

  return (
    <section className="bg-white border border-gray-200 rounded p-4 space-y-3">
      <h2 className="text-sm font-semibold text-gray-900">Latest Run: Key Metrics Overview</h2>

      {error && (
        <div className="text-sm text-red-700 bg-red-50 border border-red-200 px-3 py-2 rounded">
          {error}
        </div>
      )}

      {!error && !summary && <p className="text-sm text-gray-600">Loading latest run metrics...</p>}

      {!error && summary && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {groupRows.map((group) => (
            <div key={group.name} className="border border-gray-200 rounded p-3">
              <h3 className="text-xs uppercase tracking-wide text-gray-500 mb-2">{group.name}</h3>
              <div className="space-y-1 text-sm">
                {group.metrics.map(([label, metricKey, value]) => (
                  <div key={label} className="flex justify-between gap-3">
                    <span className="text-gray-600">{label}</span>
                    <span className="font-mono text-gray-900">{formatMetricDisplay(metricKey, value)}</span>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
    </section>
  );
}

function FeedbackHealth({ summary, error }) {
  const counts = summary?.thumb_counts || {};
  const rates = summary?.thumb_rates || {};
  const recent = summary?.recent_negative_feedback || [];
  const buckets = summary?.top_issue_buckets || [];
  const lastUpdated = summary?.last_updated_ts || null;

  return (
    <section className="bg-white border border-gray-200 rounded p-4 space-y-3">
      <h2 className="text-sm font-semibold text-gray-900">Feedback Health</h2>

      {error && (
        <div className="text-sm text-red-700 bg-red-50 border border-red-200 px-3 py-2 rounded">
          {error}
        </div>
      )}

      {!error && !summary && <p className="text-sm text-gray-600">Loading feedback summary...</p>}

      {!error && summary && (
        <>
          <div className="text-xs text-gray-500">
            Last updated: {formatTimestamp(lastUpdated ? lastUpdated * 1000 : null)}
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
            <div className="bg-gray-50 border border-gray-200 rounded px-3 py-2">
              <p className="text-xs text-gray-500">Total feedback</p>
              <p className="text-xl font-semibold text-gray-900">{summary.total_count ?? 0}</p>
            </div>
            <div className="bg-gray-50 border border-gray-200 rounded px-3 py-2">
              <p className="text-xs text-gray-500">Positive rate</p>
              <p className="text-xl font-semibold text-gray-900">{formatPercent(rates.up)}</p>
            </div>
            <div className="bg-gray-50 border border-gray-200 rounded px-3 py-2">
              <p className="text-xs text-gray-500">Attention rate (side+down)</p>
              <p className="text-xl font-semibold text-gray-900">{formatPercent(summary.attention_rate)}</p>
            </div>
            <div className="bg-gray-50 border border-gray-200 rounded px-3 py-2">
              <p className="text-xs text-gray-500">Activity (24h / 7d)</p>
              <p className="text-xl font-semibold text-gray-900">
                {summary.count_last_24h ?? 0} / {summary.count_last_7d ?? 0}
              </p>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <div className="border border-gray-200 rounded p-3">
              <h3 className="text-xs uppercase tracking-wide text-gray-500 mb-2">Feedback breakdown</h3>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between gap-3">
                  <span className="text-gray-700">Up ↑</span>
                  <span className="font-mono text-gray-900">
                    {counts.up ?? 0} ({formatPercent(rates.up)})
                  </span>
                </div>
                <div className="flex justify-between gap-3">
                  <span className="text-gray-700">Sideways →</span>
                  <span className="font-mono text-gray-900">
                    {counts.side ?? 0} ({formatPercent(rates.side)})
                  </span>
                </div>
                <div className="flex justify-between gap-3">
                  <span className="text-gray-700">Down ↓</span>
                  <span className="font-mono text-gray-900">
                    {counts.down ?? 0} ({formatPercent(rates.down)})
                  </span>
                </div>
              </div>
            </div>

            <div className="border border-gray-200 rounded p-3">
              <h3 className="text-xs uppercase tracking-wide text-gray-500 mb-2">Top issue buckets</h3>
              {buckets.length === 0 ? (
                <p className="text-sm text-gray-500">No issue buckets yet.</p>
              ) : (
                <div className="space-y-1 text-sm">
                  {buckets.map((item) => (
                    <div key={item.bucket} className="flex justify-between gap-3">
                      <span className="text-gray-600">{item.bucket}</span>
                      <span className="font-mono text-gray-900">{item.count}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>

          <div className="border border-gray-200 rounded p-3">
            <h3 className="text-xs uppercase tracking-wide text-gray-500 mb-2">Recent negative/side events</h3>
            {recent.length === 0 ? (
              <p className="text-sm text-gray-500">No recent negative or sideways events.</p>
            ) : (
              <div className="max-h-36 overflow-y-auto space-y-2">
                {recent.slice(0, 3).map((item, idx) => (
                  <div key={`event-${item.timestamp || 0}-${idx}`} className="flex items-start justify-between gap-3 text-sm">
                    <div className="min-w-0">
                      <p className="text-gray-800 truncate">{item.comment || "(no comment)"}</p>
                      <p className="text-xs text-gray-500">{formatTimestamp(item.timestamp ? item.timestamp * 1000 : null)}</p>
                    </div>
                    <span
                      className={`px-2 py-0.5 rounded text-xs font-medium border ${
                        item.thumb === "down"
                          ? "bg-red-50 text-red-700 border-red-200"
                          : "bg-amber-50 text-amber-700 border-amber-200"
                      }`}
                    >
                      {item.thumb}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>

        </>
      )}
    </section>
  );
}

function RunsTable({ runs, onSelect, selectedRunId }) {
  if (!runs.length) {
    return (
      <p className="text-sm text-gray-600">
        No evaluation runs found. Generate a run with the evaluation stack to see results here.
      </p>
    );
  }

  return (
    <div className="overflow-x-auto border border-gray-200 rounded bg-white">
      <table className="min-w-full text-sm">
        <thead className="bg-gray-50 border-b border-gray-200">
          <tr>
            <th className="text-left px-3 py-2 font-semibold text-gray-700">Run ID</th>
            <th className="text-left px-3 py-2 font-semibold text-gray-700">Timestamp</th>
            <th className="text-left px-3 py-2 font-semibold text-gray-700">Case count</th>
            <th className="text-left px-3 py-2 font-semibold text-gray-700">Mode</th>
            <th className="text-left px-3 py-2 font-semibold text-gray-700">Headline</th>
          </tr>
        </thead>
        <tbody>
          {runs.map((run) => {
            const isSelected = run.run_id === selectedRunId;
            return (
              <tr
                key={run.run_id}
                className={`border-b border-gray-100 cursor-pointer ${
                  isSelected ? "bg-blue-50" : "hover:bg-gray-50"
                }`}
                onClick={() => onSelect(run.run_id)}
              >
                <td className="px-3 py-2 align-top font-medium text-gc-link underline break-all">{run.run_id}</td>
                <td className="px-3 py-2 align-top text-gray-800 whitespace-nowrap">{formatTimestamp(run.created_at)}</td>
                <td className="px-3 py-2 align-top text-gray-800">{run.case_count ?? "—"}</td>
                <td className="px-3 py-2 align-top text-gray-800">{getRunModeLabel(run)}</td>
                <td className="px-3 py-2 align-top text-gray-800">
                  {headlineItems(run.headline).length > 0 ? (
                    <div className="space-y-1 min-w-[14rem]">
                      {headlineItems(run.headline).map(([key, value]) => (
                        <div key={key} className="flex justify-between gap-2 text-xs">
                          <span className="text-gray-600 truncate">{formatMetricName(key)}</span>
                          <span className="font-mono text-gray-900">{formatMetricDisplay(key, value)}</span>
                        </div>
                      ))}
                    </div>
                  ) : (
                    "—"
                  )}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function RunDetail({ runId }) {
  const [summary, setSummary] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!runId) return;
    setLoading(true);
    setError(null);
    setSummary(null);

    getRunSummary(runId)
      .then((data) => setSummary(data))
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, [runId]);

  const config = summary?.config || {};
  const overall = summary?.overall_metrics || {};
  const timing = summary?.timing_summary_ms || {};
  const subgroup = summary?.subgroup_metrics || {};
  const errors = summary?.error_breakdown || {};

  return (
    <section className="bg-white border border-gray-200 rounded p-4 space-y-3">
      <h2 className="text-sm font-semibold text-gray-900 break-all">Run Detail: {runId || "—"}</h2>

      {loading && <p className="text-sm text-gray-600">Loading run detail...</p>}
      {error && (
        <div className="text-sm text-red-700 bg-red-50 border border-red-200 px-3 py-2 rounded">
          {error}
        </div>
      )}

      {!loading && !error && summary && (
        <>
          <div className="border border-gray-200 rounded p-3">
            <h3 className="text-xs uppercase tracking-wide text-gray-500 mb-2">Config Summary</h3>
            <div className="space-y-2 text-sm">
              <div className="grid grid-cols-[8rem_1fr] gap-2 items-start">
                <span className="text-gray-600">Cases file</span>
                <span className="text-gray-900 break-all leading-snug">{config.cases_file || "—"}</span>
              </div>
              <div className="grid grid-cols-[8rem_1fr] gap-2 items-start">
                <span className="text-gray-600">Top K</span>
                <span className="text-gray-900">{config.top_k ?? "—"}</span>
              </div>
              <div className="grid grid-cols-[8rem_1fr] gap-2 items-start">
                <span className="text-gray-600">With chat</span>
                <span className="text-gray-900">{String(config.with_chat ?? "—")}</span>
              </div>
              <div className="grid grid-cols-[8rem_1fr] gap-2 items-start">
                <span className="text-gray-600">With judge</span>
                <span className="text-gray-900">{String(config.with_judge ?? "—")}</span>
              </div>
            </div>
          </div>

          <div className="border border-gray-200 rounded p-3">
            <h3 className="text-xs uppercase tracking-wide text-gray-500 mb-2">Error Breakdown</h3>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-2 text-sm">
              <div className="bg-gray-50 border border-gray-200 rounded px-3 py-2">
                <p className="text-gray-500 text-xs">Total cases</p>
                <p className="font-semibold text-gray-900">{errors.total_cases ?? "—"}</p>
              </div>
              <div className="bg-gray-50 border border-gray-200 rounded px-3 py-2">
                <p className="text-gray-500 text-xs">Chat errors</p>
                <p className="font-semibold text-gray-900">{errors.chat_error_count ?? "—"}</p>
              </div>
              <div className="bg-gray-50 border border-gray-200 rounded px-3 py-2">
                <p className="text-gray-500 text-xs">Judge errors</p>
                <p className="font-semibold text-gray-900">{errors.judge_error_count ?? "—"}</p>
              </div>
              <div className="bg-gray-50 border border-gray-200 rounded px-3 py-2">
                <p className="text-gray-500 text-xs">Empty answers</p>
                <p className="font-semibold text-gray-900">{errors.empty_answer_count ?? "—"}</p>
              </div>
            </div>
          </div>

          <div className="border border-gray-200 rounded p-3">
            <h3 className="text-xs uppercase tracking-wide text-gray-500 mb-2">Metric Summary</h3>
            <div className="max-h-64 overflow-y-auto border border-gray-200 rounded">
              <table className="min-w-full text-xs">
                <thead className="bg-gray-50 border-b border-gray-200">
                  <tr>
                    <th className="text-left px-2 py-1 font-semibold text-gray-700">Metric</th>
                    <th className="text-left px-2 py-1 font-semibold text-gray-700">Value</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(overall).map(([key, value]) => (
                    <tr key={key} className="border-b border-gray-100">
                      <td className="px-2 py-1 text-gray-900">{key}</td>
                      <td className="px-2 py-1 font-mono text-gray-900">{formatMetricDisplay(key, value)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <details className="border border-gray-200 rounded p-3">
            <summary className="text-xs uppercase tracking-wide text-gray-500 cursor-pointer">
              Secondary details (timing and subgroup metrics)
            </summary>
            <div className="mt-3 space-y-3 text-xs">
              <div>
                <p className="font-semibold text-gray-700 mb-1">Timing Summary</p>
                <pre className="bg-gray-50 border border-gray-200 rounded p-2 overflow-x-auto text-gray-800">
                  {JSON.stringify(timing, null, 2)}
                </pre>
              </div>
              <div>
                <p className="font-semibold text-gray-700 mb-1">Subgroup Metrics</p>
                <pre className="bg-gray-50 border border-gray-200 rounded p-2 overflow-x-auto text-gray-800">
                  {JSON.stringify(subgroup, null, 2)}
                </pre>
              </div>
            </div>
          </details>
        </>
      )}
    </section>
  );
}

function shellEscape(value) {
  const text = String(value ?? "");
  if (/^[a-zA-Z0-9_./:-]+$/.test(text)) return text;
  return `'${text.replace(/'/g, `'\"'\"'`)}'`;
}

function buildDefaultOutputName() {
  const now = new Date();
  const yyyy = now.getFullYear();
  const mm = String(now.getMonth() + 1).padStart(2, "0");
  const dd = String(now.getDate()).padStart(2, "0");
  const hh = String(now.getHours()).padStart(2, "0");
  const min = String(now.getMinutes()).padStart(2, "0");
  const ss = String(now.getSeconds()).padStart(2, "0");
  return `run_demo_${yyyy}${mm}${dd}_${hh}${min}${ss}.json`;
}

function clampNumber(value, min, max, fallback) {
  const n = Number(value);
  if (!Number.isFinite(n)) return fallback;
  return Math.max(min, Math.min(max, n));
}

function RunBuilder({ meta }) {
  const [casesFile, setCasesFile] = useState("evaluation/cases/eval_cases.jsonl");
  const [topK, setTopK] = useState(10);
  const [split, setSplit] = useState("all");
  const [withChat, setWithChat] = useState(true);
  const [withJudge, setWithJudge] = useState(true);
  const [requiredThreshold, setRequiredThreshold] = useState(0.7);
  const [forbiddenThreshold, setForbiddenThreshold] = useState(0.8);
  const [citationThreshold, setCitationThreshold] = useState(0.65);
  const [limit, setLimit] = useState(0);
  const [outputName, setOutputName] = useState(buildDefaultOutputName());
  const [copied, setCopied] = useState(false);

  const command = useMemo(() => {
    const safeTopK = Math.floor(clampNumber(topK, 1, 1000, 10));
    const safeRequired = clampNumber(requiredThreshold, 0, 1, 0.7);
    const safeForbidden = clampNumber(forbiddenThreshold, 0, 1, 0.8);
    const safeCitation = clampNumber(citationThreshold, 0, 1, 0.65);
    const safeLimit = Math.floor(clampNumber(limit, 0, 100000, 0));
    const safeSplit = (split || "").trim() || "all";
    const safeOutputBase = ((outputName || "").trim() || buildDefaultOutputName()).replace(/\s+/g, "_");
    const safeOutputName = safeOutputBase.endsWith(".json") ? safeOutputBase : `${safeOutputBase}.json`;

    const parts = [
      "python evaluation/eval_stack_runner.py",
      `--cases-file ${shellEscape(casesFile)}`,
      `--top-k ${safeTopK}`,
      `--split ${shellEscape(safeSplit)}`,
      `--required-claim-threshold ${safeRequired}`,
      `--forbidden-claim-threshold ${safeForbidden}`,
      `--citation-support-threshold ${safeCitation}`,
    ];

    if (withChat) parts.push("--with-chat");
    if (withChat && withJudge) {
      parts.push("--with-judge");
    }
    if (safeLimit > 0) parts.push(`--limit ${safeLimit}`);

    const outputPath = `evaluation/runs/${safeOutputName}`;
    parts.push(`--output ${shellEscape(outputPath)}`);
    return parts.join(" ");
  }, [
    casesFile,
    topK,
    split,
    requiredThreshold,
    forbiddenThreshold,
    citationThreshold,
    withChat,
    withJudge,
    limit,
    outputName,
  ]);

  const copyCommand = async () => {
    try {
      await navigator.clipboard.writeText(command);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {}
  };

  return (
    <section className="bg-white border border-gray-200 rounded p-4 space-y-3">
      <h2 className="text-sm font-semibold text-gray-900">Run Builder</h2>
      <p className="text-sm text-gray-700">
        Configure an evaluation run, copy the command, and execute it from the backend folder.
      </p>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-sm">
        <label className="space-y-1">
          <span className="text-gray-600">Cases file</span>
          <select
            className="w-full border border-gray-300 rounded px-2 py-1"
            value={casesFile}
            onChange={(e) => setCasesFile(e.target.value)}
          >
            <option value="evaluation/cases/eval_cases.jsonl">evaluation/cases/eval_cases.jsonl</option>
            <option value="evaluation/cases/eval_cases_reference.jsonl">evaluation/cases/eval_cases_reference.jsonl</option>
          </select>
        </label>

        <label className="space-y-1">
          <span className="text-gray-600">Split (all | dev | test | comma-separated)</span>
          <input
            className="w-full border border-gray-300 rounded px-2 py-1"
            value={split}
            onChange={(e) => setSplit(e.target.value)}
            placeholder="all"
            onBlur={() => setSplit((current) => (current || "").trim() || "all")}
          />
        </label>

        <label className="space-y-1">
          <span className="text-gray-600">Top K (&gt;= 1)</span>
          <input
            type="number"
            min={1}
            className="w-full border border-gray-300 rounded px-2 py-1"
            value={topK}
            onChange={(e) => setTopK(e.target.value)}
            onBlur={() => setTopK((current) => String(Math.floor(clampNumber(current, 1, 1000, 10))))}
          />
        </label>

        <label className="space-y-1">
          <span className="text-gray-600">Limit (0 = all, &gt;= 0)</span>
          <input
            type="number"
            min={0}
            className="w-full border border-gray-300 rounded px-2 py-1"
            value={limit}
            onChange={(e) => setLimit(e.target.value)}
            onBlur={() => setLimit((current) => String(Math.floor(clampNumber(current, 0, 100000, 0))))}
          />
        </label>

        <label className="space-y-1">
          <span className="text-gray-600">Required claim threshold (0.00 - 1.00)</span>
          <input
            type="number"
            step="0.01"
            min={0}
            max={1}
            className="w-full border border-gray-300 rounded px-2 py-1"
            value={requiredThreshold}
            onChange={(e) => setRequiredThreshold(e.target.value)}
            onBlur={() => setRequiredThreshold((current) => String(clampNumber(current, 0, 1, 0.7)))}
          />
        </label>

        <label className="space-y-1">
          <span className="text-gray-600">Forbidden claim threshold (0.00 - 1.00)</span>
          <input
            type="number"
            step="0.01"
            min={0}
            max={1}
            className="w-full border border-gray-300 rounded px-2 py-1"
            value={forbiddenThreshold}
            onChange={(e) => setForbiddenThreshold(e.target.value)}
            onBlur={() => setForbiddenThreshold((current) => String(clampNumber(current, 0, 1, 0.8)))}
          />
        </label>

        <label className="space-y-1">
          <span className="text-gray-600">Citation support threshold (0.00 - 1.00)</span>
          <input
            type="number"
            step="0.01"
            min={0}
            max={1}
            className="w-full border border-gray-300 rounded px-2 py-1"
            value={citationThreshold}
            onChange={(e) => setCitationThreshold(e.target.value)}
            onBlur={() => setCitationThreshold((current) => String(clampNumber(current, 0, 1, 0.65)))}
          />
        </label>

        <label className="space-y-1">
          <span className="text-gray-600">Output filename (*.json)</span>
          <div className="flex gap-2">
            <input
              className="w-full border border-gray-300 rounded px-2 py-1"
              value={outputName}
              onChange={(e) => setOutputName(e.target.value)}
              placeholder="run_demo_YYYYMMDD_HHMMSS.json"
              onBlur={() =>
                setOutputName((current) => {
                  const cleaned = ((current || "").trim() || buildDefaultOutputName()).replace(/\s+/g, "_");
                  return cleaned.endsWith(".json") ? cleaned : `${cleaned}.json`;
                })
              }
            />
            <button
              type="button"
              className="px-2 py-1 border border-gray-300 rounded bg-gray-50 text-xs"
              onClick={() => setOutputName(buildDefaultOutputName())}
            >
              Reset
            </button>
          </div>
        </label>
      </div>

      <div className="flex flex-wrap items-center gap-4 text-sm">
        <label className="inline-flex items-center gap-2">
          <input
            type="checkbox"
            checked={withChat}
            onChange={(e) => {
              const next = e.target.checked;
              setWithChat(next);
              if (!next) setWithJudge(false);
            }}
          />
          <span className="text-gray-700">With chat</span>
        </label>
        <label className="inline-flex items-center gap-2">
          <input
            type="checkbox"
            checked={withJudge}
            disabled={!withChat}
            onChange={(e) => setWithJudge(e.target.checked)}
          />
          <span className={`${withChat ? "text-gray-700" : "text-gray-400"}`}>With judge</span>
        </label>
      </div>

      <div className="space-y-2">
        <p className="text-xs uppercase tracking-wide text-gray-500">Command</p>
        <pre className="text-xs bg-gray-50 border border-gray-200 rounded p-2 overflow-x-auto text-gray-800">
          {command}
        </pre>
        <div className="flex items-center gap-2">
          <button
            type="button"
            className="px-3 py-1.5 border border-gray-300 rounded bg-gray-900 text-white text-sm"
            onClick={copyCommand}
          >
            {copied ? "Copied" : "Copy command"}
          </button>
          <span className="text-xs text-gray-500">Run from: <code>/Users/jason/DepartmentDefence-Winter2026/backend</code></span>
        </div>
      </div>
    </section>
  );
}

function DashboardReference({ meta, error }) {
  const models = meta?.models || {};
  const executionModes = Array.isArray(meta?.execution_modes) ? meta.execution_modes : [];
  const caseModes = Array.isArray(meta?.case_modes) ? meta.case_modes : [];

  return (
    <section className="bg-white border border-gray-200 rounded p-4 space-y-4">
      <h2 className="text-sm font-semibold text-gray-900">Dashboard Reference</h2>

      {error && (
        <div className="text-sm text-red-700 bg-red-50 border border-red-200 px-3 py-2 rounded">
          {error}
        </div>
      )}

      {!error && !meta && <p className="text-sm text-gray-600">Loading reference information...</p>}

      {!error && meta && (
        <>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="border border-gray-200 rounded p-3 space-y-2">
              <h3 className="text-xs uppercase tracking-wide text-gray-500">Current model stack</h3>
              <div className="space-y-1 text-sm">
                <div className="grid grid-cols-[9rem_1fr] gap-2">
                  <span className="text-gray-600">Chat model</span>
                  <span className="font-mono text-gray-900 break-all">{models.chat_model || "—"}</span>
                </div>
                <div className="grid grid-cols-[9rem_1fr] gap-2">
                  <span className="text-gray-600">Embedding model</span>
                  <span className="font-mono text-gray-900 break-all">{models.embed_model || "—"}</span>
                </div>
                <div className="grid grid-cols-[9rem_1fr] gap-2">
                  <span className="text-gray-600">Rerank model</span>
                  <span className="font-mono text-gray-900 break-all">{models.rerank_model || "—"}</span>
                </div>
                <div className="grid grid-cols-[9rem_1fr] gap-2">
                  <span className="text-gray-600">Judge default</span>
                  <span className="font-mono text-gray-900 break-all">{models.judge_model_default || "—"}</span>
                </div>
              </div>
            </div>

            <div className="border border-gray-200 rounded p-3 space-y-2">
              <h3 className="text-xs uppercase tracking-wide text-gray-500">What a run means</h3>
              <p className="text-sm text-gray-700">
                A run is an offline evaluation report generated from structured test cases.
              </p>
              <p className="text-sm text-gray-700">
                Normal chat usage updates live feedback signals (thumbs up, sideways, down) shown in the Feedback Health section.
              </p>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="border border-gray-200 rounded p-3">
              <h3 className="text-xs uppercase tracking-wide text-gray-500 mb-2">Execution modes</h3>
              <div className="flex flex-wrap gap-2">
                {executionModes.length > 0 ? (
                  executionModes.map((mode) => (
                    <span key={mode} className="px-2 py-1 rounded text-xs bg-blue-50 text-blue-800 border border-blue-200">
                      {mode}
                    </span>
                  ))
                ) : (
                  <span className="text-sm text-gray-500">No execution modes available.</span>
                )}
              </div>
            </div>

            <div className="border border-gray-200 rounded p-3">
              <h3 className="text-xs uppercase tracking-wide text-gray-500 mb-2">Case scenario modes</h3>
              <div className="flex flex-wrap gap-2">
                {caseModes.length > 0 ? (
                  caseModes.map((mode) => (
                    <span key={mode} className="px-2 py-1 rounded text-xs bg-gray-50 text-gray-700 border border-gray-200">
                      {mode}
                    </span>
                  ))
                ) : (
                  <span className="text-sm text-gray-500">No case modes available.</span>
                )}
              </div>
            </div>
          </div>

          <div className="border border-gray-200 rounded p-3 space-y-3">
            <h3 className="text-xs uppercase tracking-wide text-gray-500">Metric guide</h3>
            <div className="space-y-2 text-sm">
              <div className="grid grid-cols-[9rem_1fr] gap-2 items-start">
                <p className="font-medium text-gray-800">Retrieval</p>
                <p className="text-gray-700">Checks if the right policy evidence was found. Higher recall/MRR is better; lower noise is better.</p>
              </div>
              <div className="grid grid-cols-[9rem_1fr] gap-2 items-start">
                <p className="font-medium text-gray-800">Answer quality</p>
                <p className="text-gray-700">Checks if answers include required claims, avoid forbidden claims, and use supported citations.</p>
              </div>
              <div className="grid grid-cols-[9rem_1fr] gap-2 items-start">
                <p className="font-medium text-gray-800">Judge (optional)</p>
                <p className="text-gray-700">Optional LLM review score for correctness/alignment. Useful as a signal, not final truth.</p>
              </div>
              <div className="grid grid-cols-[9rem_1fr] gap-2 items-start">
                <p className="font-medium text-gray-800">Timing</p>
                <p className="text-gray-700">p50/p95 latency for retrieval, chat, and judge. Used to track responsiveness and regressions.</p>
              </div>
            </div>
          </div>
        </>
      )}
    </section>
  );
}

function ModDashboard() {
  const [accessLoading, setAccessLoading] = useState(true);
  const [accessError, setAccessError] = useState(null);
  const [runs, setRuns] = useState([]);
  const [runsLoading, setRunsLoading] = useState(false);
  const [runsError, setRunsError] = useState(null);
  const [selectedRunId, setSelectedRunId] = useState(null);
  const [latestSummary, setLatestSummary] = useState(null);
  const [latestSummaryError, setLatestSummaryError] = useState(null);
  const [feedbackSummary, setFeedbackSummary] = useState(null);
  const [feedbackError, setFeedbackError] = useState(null);
  const [dashboardMeta, setDashboardMeta] = useState(null);
  const [dashboardMetaError, setDashboardMetaError] = useState(null);

  useEffect(() => {
    setAccessLoading(true);
    setAccessError(null);
    checkDashboardAccess()
      .then(() => setAccessLoading(false))
      .catch((err) => {
        setAccessError(err.message);
        setAccessLoading(false);
      });
  }, []);

  useEffect(() => {
    if (accessLoading || accessError) return;
    setRunsLoading(true);
    setRunsError(null);

    getRuns()
      .then((data) => {
        const list = data.runs || [];
        setRuns(list);
        if (list.length > 0) setSelectedRunId(list[0].run_id);
      })
      .catch((err) => setRunsError(err.message))
      .finally(() => setRunsLoading(false));
  }, [accessLoading, accessError]);

  useEffect(() => {
    if (accessLoading || accessError) return;
    setDashboardMeta(null);
    setDashboardMetaError(null);
    getDashboardMeta()
      .then((data) => setDashboardMeta(data))
      .catch((err) => setDashboardMetaError(err.message));
  }, [accessLoading, accessError]);

  const latestRunId = useMemo(() => runs[0]?.run_id || null, [runs]);

  useEffect(() => {
    if (accessLoading || accessError) return;
    if (!latestRunId) {
      setLatestSummary(null);
      setLatestSummaryError(null);
      return;
    }

    setLatestSummary(null);
    setLatestSummaryError(null);

    getRunSummary(latestRunId)
      .then((data) => setLatestSummary(data))
      .catch((err) => setLatestSummaryError(err.message));
  }, [latestRunId, accessLoading, accessError]);

  useEffect(() => {
    if (accessLoading || accessError) return;
    let isMounted = true;
    let timer = null;

    const loadFeedback = () => {
      getFeedbackSummary()
        .then((data) => {
          if (!isMounted) return;
          setFeedbackSummary(data);
          setFeedbackError(null);
        })
        .catch((err) => {
          if (!isMounted) return;
          setFeedbackError(err.message);
        });
    };

    setFeedbackSummary(null);
    setFeedbackError(null);
    loadFeedback();
    timer = setInterval(loadFeedback, 10000);

    return () => {
      isMounted = false;
      if (timer) clearInterval(timer);
    };
  }, [accessLoading, accessError]);

  if (accessLoading) {
    return (
      <div className="min-h-screen bg-[#F5F5F5] p-4">
        <div className="max-w-3xl mx-auto bg-white border border-gray-200 rounded p-6">
          <p className="text-sm text-gray-700">Checking dashboard access...</p>
        </div>
      </div>
    );
  }

  if (accessError) {
    return (
      <div className="min-h-screen bg-[#F5F5F5] p-4">
        <div className="max-w-3xl mx-auto bg-white border border-gray-200 rounded p-6 space-y-2">
          <h1 className="text-lg font-semibold text-gray-900">404 Not Found</h1>
          <p className="text-sm text-gray-700">The requested page could not be found.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-[#F5F5F5] p-4">
      <div className="max-w-6xl mx-auto space-y-4">
        <header>
          <h1 className="text-2xl font-semibold text-gray-900">Moderator Evaluation Dashboard</h1>
        </header>

        <SummaryCards runs={runs} />

        <KeyMetricsOverview summary={latestSummary} error={latestSummaryError} />
        <FeedbackHealth summary={feedbackSummary} error={feedbackError} />
        <RunBuilder meta={dashboardMeta} />

        <section className="grid grid-cols-1 lg:grid-cols-2 gap-4 items-start">
          <div className="space-y-2">
            <h2 className="text-sm font-semibold text-gray-900">Evaluation Runs</h2>
            {runsLoading && <p className="text-sm text-gray-600">Loading runs...</p>}
            {runsError && (
              <div className="text-sm text-red-700 bg-red-50 border border-red-200 px-3 py-2 rounded">
                {runsError}
              </div>
            )}
            {!runsLoading && !runsError && (
              <RunsTable runs={runs} onSelect={setSelectedRunId} selectedRunId={selectedRunId} />
            )}
          </div>

          <RunDetail runId={selectedRunId} />
        </section>

        <DashboardReference meta={dashboardMeta} error={dashboardMetaError} />
      </div>
    </div>
  );
}

export default ModDashboard;
