import { useEffect, useMemo, useReducer, useState } from "react";
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

const COMPARISON_METRICS = [
  "retrieval_gold_doc_recall_at_k_mean",
  "retrieval_gold_doc_mrr_mean",
  "retrieval_gold_doc_precision_at_k_mean",
  "retrieval_claim_evidence_coverage_mean",
  "retrieval_noise_rate_mean",
  "answer_required_claim_recall_mean",
  "answer_citation_support_rate_mean",
  "answer_forbidden_violation_rate",
  "answer_abstention_accuracy",
];

const LOWER_IS_BETTER_METRICS = new Set([
  "retrieval_noise_rate_mean",
  "answer_forbidden_violation_rate",
  "timing_total_p50_ms",
]);

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

function EmptyState({ title, body }) {
  return (
    <div className="text-sm text-gray-700 bg-gray-50 border border-gray-200 rounded px-3 py-3 space-y-1">
      <p className="font-medium text-gray-900">{title}</p>
      <p className="text-gray-600">{body}</p>
    </div>
  );
}

function formatMetricName(key) {
  return METRIC_LABELS[key] || key.replace(/^retrieval_|^answer_/, "").replace(/_/g, " ");
}

function formatMetricDisplay(metricKey, value) {
  if (typeof value !== "number") return formatMetricValue(value);
  if (PERCENT_METRIC_KEYS.has(metricKey)) return `${(value * 100).toFixed(1)}%`;
  return value.toFixed(3);
}

function formatDelta(value) {
  if (typeof value !== "number") return "—";
  const sign = value > 0 ? "+" : "";
  return `${sign}${value.toFixed(3)}`;
}

function getDeltaTone(metricKey, delta) {
  if (typeof delta !== "number" || Math.abs(delta) < 1e-12) return "text-gray-700";
  const improved = LOWER_IS_BETTER_METRICS.has(metricKey) ? delta < 0 : delta > 0;
  return improved ? "text-emerald-700" : "text-red-700";
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

function createLoadState(initialData = null, initialLoading = false) {
  return {
    data: initialData,
    loading: initialLoading,
    error: null,
  };
}

function loadStateReducer(state, action) {
  switch (action.type) {
    case "start":
      return {
        data: action.keepData ? state.data : action.data ?? null,
        loading: true,
        error: null,
      };
    case "success":
      return {
        data: action.data,
        loading: false,
        error: null,
      };
    case "error":
      return {
        data: action.keepData ? state.data : action.data ?? null,
        loading: false,
        error: action.error,
      };
    case "reset":
      return {
        data: action.data ?? null,
        loading: false,
        error: null,
      };
    default:
      return state;
  }
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
  const hasSummary = Boolean(summary);

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

      {!error && !hasSummary && (
        <EmptyState
          title="No latest run metrics yet"
          body="Generate at least one evaluation run to populate the latest metrics overview."
        />
      )}

      {!error && hasSummary && (
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

function feedbackSectionTitle(type) {
  return type === "answer" ? "Answer Feedback" : "Citation Feedback";
}

function feedbackSectionDescription(type) {
  return type === "answer"
    ? "Overall answer ratings and optional comments from the chat interface."
    : "Citation-card ratings that help tune retrieval weighting and source quality signals.";
}

function feedbackEmptyTitle(type) {
  return type === "answer" ? "No answer feedback yet" : "No citation feedback yet";
}

function feedbackEmptyBody(type) {
  return type === "answer"
    ? "Use the answer-level thumbs under a bot response to start tracking overall answer quality."
    : "Use the citation thumbs on source cards to start tracking citation quality and chunk weighting feedback.";
}

function prettifySourceLabel(item) {
  const sourceTitle = item?.source_title?.trim();
  const sourcePath = item?.source_path?.trim();
  if (sourceTitle) return sourceTitle;
  if (!sourcePath) return item?.target_chunk_id || "Unknown source";
  const normalized = sourcePath.replace(/\\/g, "/");
  const filename = normalized.split("/").pop() || normalized;
  return filename.replace(/\.(md|txt)$/i, "");
}

function renderCitationFeedbackLabel(item) {
  const sourceLabel = prettifySourceLabel(item);
  const sectionTitle = item?.section_title?.trim();
  const sourceUrl = item?.source_url?.trim();
  const label = sectionTitle ? `${sourceLabel} - ${sectionTitle}` : sourceLabel;

  if (sourceUrl) {
    return (
      <a
        href={sourceUrl}
        target="_blank"
        rel="noreferrer"
        className="text-gray-800 underline decoration-gray-300 underline-offset-2 hover:text-gc-blue"
      >
        {label}
      </a>
    );
  }

  return <span className="text-gray-800">{label}</span>;
}

function CitationDebugDetails({ item }) {
  const sourceUrl = item?.source_url?.trim();
  const sourcePath = item?.source_path?.trim();
  const sectionTitle = item?.section_title?.trim();
  const docType = item?.doc_type?.trim();
  const chunkPreview = item?.chunk_preview?.trim();
  const chunkId = item?.target_chunk_id?.trim();
  const authorityRank = Number.isFinite(item?.authority_rank) ? item.authority_rank : null;

  return (
    <details className="mt-2">
      <summary className="cursor-pointer text-xs text-gc-blue hover:underline">
        View chunk details
      </summary>
      <div className="mt-2 rounded border border-gray-200 bg-gray-50 p-3 text-xs text-gray-700 space-y-2">
        {chunkId && (
          <div>
            <span className="font-semibold text-gray-900">Chunk ID:</span>{" "}
            <span className="font-mono break-all">{chunkId}</span>
          </div>
        )}
        {sectionTitle && (
          <div>
            <span className="font-semibold text-gray-900">Section:</span> {sectionTitle}
          </div>
        )}
        {docType && (
          <div>
            <span className="font-semibold text-gray-900">Document type:</span> {docType}
          </div>
        )}
        {authorityRank ? (
          <div>
            <span className="font-semibold text-gray-900">Authority rank:</span> {authorityRank}
          </div>
        ) : null}
        {sourcePath && (
          <div>
            <span className="font-semibold text-gray-900">Source path:</span>{" "}
            <span className="font-mono break-all">{sourcePath}</span>
          </div>
        )}
        {sourceUrl && (
          <div>
            <span className="font-semibold text-gray-900">Source link:</span>{" "}
            <a
              href={sourceUrl}
              target="_blank"
              rel="noreferrer"
              className="text-gc-blue underline break-all"
            >
              {sourceUrl}
            </a>
          </div>
        )}
        {chunkPreview && (
          <div>
            <p className="font-semibold text-gray-900">Chunk preview</p>
            <p className="mt-1 leading-5">{chunkPreview}</p>
          </div>
        )}
      </div>
    </details>
  );
}

function AnswerDebugDetails({ item }) {
  const question = item?.question?.trim();
  const answer = item?.answer?.trim();
  const comment = item?.comment?.trim();
  const citedChunkIds = Array.isArray(item?.cited_chunk_ids) ? item.cited_chunk_ids.filter(Boolean) : [];

  return (
    <details className="mt-2">
      <summary className="cursor-pointer text-xs text-gc-blue hover:underline">
        View answer details
      </summary>
      <div className="mt-2 rounded border border-gray-200 bg-gray-50 p-3 text-xs text-gray-700 space-y-3">
        {question && (
          <div>
            <p className="font-semibold text-gray-900">Question</p>
            <p className="mt-1 leading-5">{question}</p>
          </div>
        )}
        {answer && (
          <div>
            <p className="font-semibold text-gray-900">Answer</p>
            <p className="mt-1 leading-5 whitespace-pre-wrap">{answer}</p>
          </div>
        )}
        <div>
          <p className="font-semibold text-gray-900">User comment</p>
          <p className="mt-1 leading-5">{comment || "(no comment provided)"}</p>
        </div>
        <div>
          <p className="font-semibold text-gray-900">Attached citations</p>
          <p className="mt-1 leading-5">{citedChunkIds.length} linked chunk(s)</p>
          {citedChunkIds.length > 0 && (
            <div className="mt-2 space-y-1">
              {citedChunkIds.slice(0, 5).map((chunkId) => (
                <p key={chunkId} className="font-mono break-all text-gray-600">
                  {chunkId}
                </p>
              ))}
              {citedChunkIds.length > 5 && (
                <p className="text-gray-500">+ {citedChunkIds.length - 5} more chunk IDs</p>
              )}
            </div>
          )}
        </div>
      </div>
    </details>
  );
}

function FeedbackSignalVisual({ counts, totalFeedback }) {
  const upCount = counts.up ?? 0;
  const sideCount = counts.side ?? 0;
  const downCount = counts.down ?? 0;
  const total = Math.max(totalFeedback || 0, 0);

  const percent = (value) => {
    if (!total) return 0;
    return (value / total) * 100;
  };

  return (
    <div className="border border-gray-200 rounded p-3 space-y-3">
      <h4 className="text-xs uppercase tracking-wide text-gray-500">Signal Mix</h4>

      <div className="h-3 w-full overflow-hidden rounded-full bg-gray-100 border border-gray-200 flex">
        <div
          className="h-full bg-emerald-500"
          style={{ width: `${percent(upCount)}%` }}
          aria-hidden="true"
        />
        <div
          className="h-full bg-amber-400"
          style={{ width: `${percent(sideCount)}%` }}
          aria-hidden="true"
        />
        <div
          className="h-full bg-rose-500"
          style={{ width: `${percent(downCount)}%` }}
          aria-hidden="true"
        />
      </div>

      <div className="grid grid-cols-3 gap-3 text-center text-xs">
        <div className="rounded border border-gray-200 bg-emerald-50 px-2 py-2">
          <p className="font-medium text-emerald-700">Up</p>
          <p className="mt-1 font-mono text-gray-900">{formatPercent(total ? upCount / total : 0)}</p>
        </div>
        <div className="rounded border border-gray-200 bg-amber-50 px-2 py-2">
          <p className="font-medium text-amber-700">Side</p>
          <p className="mt-1 font-mono text-gray-900">{formatPercent(total ? sideCount / total : 0)}</p>
        </div>
        <div className="rounded border border-gray-200 bg-rose-50 px-2 py-2">
          <p className="font-medium text-rose-700">Down</p>
          <p className="mt-1 font-mono text-gray-900">{formatPercent(total ? downCount / total : 0)}</p>
        </div>
      </div>
    </div>
  );
}

function FeedbackQuickTake({ counts, totalFeedback }) {
  const upCount = counts.up ?? 0;
  const sideCount = counts.side ?? 0;
  const downCount = counts.down ?? 0;
  const attentionCount = sideCount + downCount;
  const overallRead =
    upCount > attentionCount ? "Mostly positive" :
    attentionCount > upCount ? "Needs attention" :
    totalFeedback > 0 ? "Mixed signal" : "No signal yet";
  const dominantSignal =
    upCount > downCount && upCount > sideCount ? "Upvotes are leading" :
    downCount > upCount && downCount > sideCount ? "Downvotes are leading" :
    sideCount > upCount && sideCount > downCount ? "Sideways feedback is leading" :
    totalFeedback > 0 ? "No single signal is dominating" : "No signal yet";
  const snapshot =
    totalFeedback === 0
      ? "Waiting for the first feedback events."
      : attentionCount === 0
        ? "Feedback is currently positive without open concerns."
        : downCount > upCount
          ? "Users are flagging more issues than approvals right now."
          : attentionCount === upCount
            ? "Feedback is split between approvals and issues."
            : "There are some concerns, but positive feedback still leads.";

  return (
    <div className="border border-gray-200 rounded p-3 space-y-3">
      <h4 className="text-xs uppercase tracking-wide text-gray-500">Quick Take</h4>
      <div className="space-y-2 text-sm">
        <div className="flex justify-between gap-3">
          <span className="text-gray-600">Overall read</span>
          <span className="font-medium text-gray-900">{overallRead}</span>
        </div>
        <div className="flex justify-between gap-3">
          <span className="text-gray-600">Dominant signal</span>
          <span className="font-medium text-gray-900 text-right">{dominantSignal}</span>
        </div>
        <p className="rounded bg-gray-50 border border-gray-200 px-3 py-2 text-gray-700 leading-5">
          {snapshot}
        </p>
      </div>
    </div>
  );
}

function FeedbackSummarySection({ type, summary }) {
  const counts = summary?.thumb_counts || {};
  const rates = summary?.thumb_rates || {};
  const recent = summary?.recent_negative_feedback || [];
  const lastUpdated = summary?.last_updated_ts || null;
  const totalFeedback = summary?.total_count ?? 0;

  return (
    <div className="border border-gray-200 rounded p-4 space-y-3">
      <div className="space-y-1">
        <h3 className="text-sm font-semibold text-gray-900">{feedbackSectionTitle(type)}</h3>
        <p className="text-sm text-gray-600">{feedbackSectionDescription(type)}</p>
      </div>

      <div className="text-xs text-gray-500">
        Last updated: {formatTimestamp(lastUpdated ? lastUpdated * 1000 : null)}
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-3">
        <div className="bg-gray-50 border border-gray-200 rounded px-3 py-2">
          <p className="text-xs text-gray-500">Total feedback</p>
          <p className="text-xl font-semibold text-gray-900">{totalFeedback}</p>
        </div>
        <div className="bg-gray-50 border border-gray-200 rounded px-3 py-2">
          <p className="text-xs text-gray-500">Positive rate</p>
          <p className="text-xl font-semibold text-gray-900">{formatPercent(rates.up)}</p>
        </div>
        <div className="bg-gray-50 border border-gray-200 rounded px-3 py-2">
          <p className="text-xs text-gray-500">Attention rate (side+down)</p>
          <p className="text-xl font-semibold text-gray-900">{formatPercent(summary?.attention_rate)}</p>
        </div>
        <div className="bg-gray-50 border border-gray-200 rounded px-3 py-2">
          <p className="text-xs text-gray-500">Activity (24h / 7d)</p>
          <p className="text-xl font-semibold text-gray-900">
            {summary?.count_last_24h ?? 0} / {summary?.count_last_7d ?? 0}
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <FeedbackQuickTake counts={counts} totalFeedback={totalFeedback} />

        <FeedbackSignalVisual counts={counts} totalFeedback={totalFeedback} />

      </div>

      <div className="border border-gray-200 rounded p-3">
        <h4 className="text-xs uppercase tracking-wide text-gray-500 mb-2">All negative/side events</h4>
        {totalFeedback === 0 ? (
          <EmptyState title={feedbackEmptyTitle(type)} body={feedbackEmptyBody(type)} />
        ) : recent.length === 0 ? (
          <p className="text-sm text-gray-500">No recent negative or sideways events.</p>
        ) : (
          <div className="max-h-56 overflow-y-auto pr-1 space-y-2">
            {recent.map((item, idx) => (
              <div key={`${type}-${item.timestamp || 0}-${idx}`} className="flex items-start justify-between gap-3 text-sm">
                <div className="min-w-0">
                  {item.comment ? (
                    <div>
                      <p className="text-gray-800 truncate">{item.comment}</p>
                      {type === "answer" && <AnswerDebugDetails item={item} />}
                    </div>
                  ) : type === "citation" ? (
                    <div>
                      <div className="truncate">{renderCitationFeedbackLabel(item)}</div>
                      <CitationDebugDetails item={item} />
                    </div>
                  ) : (
                    <div>
                      <p className="text-gray-500">(no comment)</p>
                      {type === "answer" && <AnswerDebugDetails item={item} />}
                    </div>
                  )}
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
    </div>
  );
}

function FeedbackHealth({ summary, error, loading, onRefresh }) {
  const citationSummary = summary?.citation || null;
  const answerSummary = summary?.answer || null;

  if (!error && summary) {
    return (
      <section className="bg-white border border-gray-200 rounded p-4 space-y-3">
        <div className="flex items-center justify-between gap-3 flex-wrap">
          <h2 className="text-sm font-semibold text-gray-900">Feedback Health</h2>
          <button
            type="button"
            onClick={onRefresh}
            disabled={loading}
            className="px-3 py-1.5 rounded border border-gray-300 text-sm text-gray-700 hover:bg-gray-50 disabled:opacity-60 disabled:cursor-not-allowed"
          >
            {loading ? "Refreshing..." : "Refresh"}
          </button>
        </div>
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
          <FeedbackSummarySection type="citation" summary={citationSummary} />
          <FeedbackSummarySection type="answer" summary={answerSummary} />
        </div>
      </section>
    );
  }

  return (
    <section className="bg-white border border-gray-200 rounded p-4 space-y-3">
      <div className="flex items-center justify-between gap-3 flex-wrap">
        <h2 className="text-sm font-semibold text-gray-900">Feedback Health</h2>
        <button
          type="button"
          onClick={onRefresh}
          disabled={loading}
          className="px-3 py-1.5 rounded border border-gray-300 text-sm text-gray-700 hover:bg-gray-50 disabled:opacity-60 disabled:cursor-not-allowed"
        >
          {loading ? "Refreshing..." : "Refresh"}
        </button>
      </div>

      {error && (
        <div className="text-sm text-red-700 bg-red-50 border border-red-200 px-3 py-2 rounded">
          {error}
        </div>
      )}

      {!error && !summary && <p className="text-sm text-gray-600">Loading feedback summary...</p>}
    </section>
  );
}

function RunsTable({ runs, onSelect, selectedRunId }) {
  if (!runs.length) {
    return (
      <EmptyState
        title="No evaluation runs found"
        body="Use the run builder above to generate your first evaluation artifact. Once a run is created, it will appear here automatically."
      />
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

function RunComparison({ runs }) {
  const [leftRunId, setLeftRunId] = useState(null);
  const [rightRunId, setRightRunId] = useState(null);
  const [leftSummaryState, dispatchLeftSummary] = useReducer(loadStateReducer, createLoadState());
  const [rightSummaryState, dispatchRightSummary] = useReducer(loadStateReducer, createLoadState());
  const selectedLeftRunId = leftRunId && runs.some((run) => run.run_id === leftRunId) ? leftRunId : runs[0]?.run_id || null;
  const selectedRightRunId =
    rightRunId && runs.some((run) => run.run_id === rightRunId)
      ? rightRunId
      : runs.find((run) => run.run_id !== selectedLeftRunId)?.run_id || null;

  useEffect(() => {
    if (!selectedLeftRunId) {
      dispatchLeftSummary({ type: "reset" });
      return;
    }

    let isMounted = true;
    dispatchLeftSummary({ type: "start" });

    getRunSummary(selectedLeftRunId)
      .then((data) => {
        if (!isMounted) return;
        dispatchLeftSummary({ type: "success", data });
      })
      .catch((err) => {
        if (!isMounted) return;
        dispatchLeftSummary({ type: "error", error: err.message });
      });

    return () => {
      isMounted = false;
    };
  }, [selectedLeftRunId]);

  useEffect(() => {
    if (!selectedRightRunId) {
      dispatchRightSummary({ type: "reset" });
      return;
    }

    let isMounted = true;
    dispatchRightSummary({ type: "start" });

    getRunSummary(selectedRightRunId)
      .then((data) => {
        if (!isMounted) return;
        dispatchRightSummary({ type: "success", data });
      })
      .catch((err) => {
        if (!isMounted) return;
        dispatchRightSummary({ type: "error", error: err.message });
      });

    return () => {
      isMounted = false;
    };
  }, [selectedRightRunId]);

  const leftRun = runs.find((run) => run.run_id === selectedLeftRunId) || null;
  const rightRun = runs.find((run) => run.run_id === selectedRightRunId) || null;
  const leftSummary = leftSummaryState.data;
  const rightSummary = rightSummaryState.data;
  const leftError = leftSummaryState.error;
  const rightError = rightSummaryState.error;
  const isLoading = leftSummaryState.loading || rightSummaryState.loading;

  const comparisonRows = useMemo(() => {
    if (!leftSummary || !rightSummary) return [];

    const leftOverall = leftSummary.overall_metrics || {};
    const rightOverall = rightSummary.overall_metrics || {};
    const leftTiming = leftSummary.timing_summary_ms?.total?.p50;
    const rightTiming = rightSummary.timing_summary_ms?.total?.p50;

    const metricRows = COMPARISON_METRICS.map((metricKey) => {
      const leftValue = leftOverall[metricKey];
      const rightValue = rightOverall[metricKey];
      return {
        key: metricKey,
        label: formatMetricName(metricKey),
        leftValue,
        rightValue,
        delta: typeof leftValue === "number" && typeof rightValue === "number" ? rightValue - leftValue : null,
      };
    });

    metricRows.push({
      key: "timing_total_p50_ms",
      label: "Total p50 latency",
      leftValue: leftTiming,
      rightValue: rightTiming,
      delta: typeof leftTiming === "number" && typeof rightTiming === "number" ? rightTiming - leftTiming : null,
    });

    return metricRows;
  }, [leftSummary, rightSummary]);

  if (runs.length < 2) {
    return (
      <section className="bg-white border border-gray-200 rounded p-4 space-y-3">
        <h2 className="text-sm font-semibold text-gray-900">Run Comparison</h2>
        <EmptyState
          title="Need at least two runs to compare"
          body="Generate a second evaluation run to unlock side-by-side metric comparisons."
        />
      </section>
    );
  }

  return (
    <section className="bg-white border border-gray-200 rounded p-4 space-y-4">
      <div className="flex items-center justify-between gap-3 flex-wrap">
        <h2 className="text-sm font-semibold text-gray-900">Run Comparison</h2>
        <p className="text-xs text-gray-500">Compare two evaluation runs using the dashboard summary metrics.</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-sm">
        <label className="space-y-1">
          <span className="text-gray-600">Run A</span>
          <select
            className="w-full border border-gray-300 rounded px-2 py-1 bg-white"
            value={selectedLeftRunId || ""}
            onChange={(e) => setLeftRunId(e.target.value || null)}
          >
            {runs.map((run) => (
              <option key={`left-${run.run_id}`} value={run.run_id}>
                {run.run_id}
              </option>
            ))}
          </select>
        </label>

        <label className="space-y-1">
          <span className="text-gray-600">Run B</span>
          <select
            className="w-full border border-gray-300 rounded px-2 py-1 bg-white"
            value={selectedRightRunId || ""}
            onChange={(e) => setRightRunId(e.target.value || null)}
          >
            {runs.map((run) => (
              <option key={`right-${run.run_id}`} value={run.run_id}>
                {run.run_id}
              </option>
            ))}
          </select>
        </label>
      </div>

      {(leftError || rightError) && (
        <div className="text-sm text-red-700 bg-red-50 border border-red-200 px-3 py-2 rounded">
          {leftError || rightError}
        </div>
      )}

      {selectedLeftRunId && selectedRightRunId && selectedLeftRunId === selectedRightRunId && (
        <EmptyState
          title="Choose two different runs"
          body="Select distinct run IDs to see side-by-side metric deltas."
        />
      )}

      {isLoading && selectedLeftRunId && selectedRightRunId && selectedLeftRunId !== selectedRightRunId && (
        <p className="text-sm text-gray-600">Loading run comparison...</p>
      )}

      {!isLoading && leftRun && rightRun && selectedLeftRunId !== selectedRightRunId && comparisonRows.length > 0 && (
        <>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 text-sm">
            <div className="border border-gray-200 rounded p-3 space-y-2">
              <h3 className="text-xs uppercase tracking-wide text-gray-500">Run A</h3>
              <div className="grid grid-cols-[7rem_1fr] gap-2 items-start">
                <span className="text-gray-600">Run ID</span>
                <span className="text-gray-900 break-all">{leftRun.run_id}</span>
              </div>
              <div className="grid grid-cols-[7rem_1fr] gap-2 items-start">
                <span className="text-gray-600">Mode</span>
                <span className="text-gray-900">{getRunModeLabel(leftRun)}</span>
              </div>
              <div className="grid grid-cols-[7rem_1fr] gap-2 items-start">
                <span className="text-gray-600">Timestamp</span>
                <span className="text-gray-900">{formatTimestamp(leftRun.created_at)}</span>
              </div>
            </div>

            <div className="border border-gray-200 rounded p-3 space-y-2">
              <h3 className="text-xs uppercase tracking-wide text-gray-500">Run B</h3>
              <div className="grid grid-cols-[7rem_1fr] gap-2 items-start">
                <span className="text-gray-600">Run ID</span>
                <span className="text-gray-900 break-all">{rightRun.run_id}</span>
              </div>
              <div className="grid grid-cols-[7rem_1fr] gap-2 items-start">
                <span className="text-gray-600">Mode</span>
                <span className="text-gray-900">{getRunModeLabel(rightRun)}</span>
              </div>
              <div className="grid grid-cols-[7rem_1fr] gap-2 items-start">
                <span className="text-gray-600">Timestamp</span>
                <span className="text-gray-900">{formatTimestamp(rightRun.created_at)}</span>
              </div>
            </div>
          </div>

          <div className="overflow-x-auto border border-gray-200 rounded bg-white">
            <table className="min-w-full text-sm">
              <thead className="bg-gray-50 border-b border-gray-200">
                <tr>
                  <th className="text-left px-3 py-2 font-semibold text-gray-700">Metric</th>
                  <th className="text-left px-3 py-2 font-semibold text-gray-700">Run A</th>
                  <th className="text-left px-3 py-2 font-semibold text-gray-700">Run B</th>
                  <th className="text-left px-3 py-2 font-semibold text-gray-700">Delta (B - A)</th>
                </tr>
              </thead>
              <tbody>
                {comparisonRows.map((row) => (
                  <tr key={row.key} className="border-b border-gray-100">
                    <td className="px-3 py-2 text-gray-900">{row.label}</td>
                    <td className="px-3 py-2 font-mono text-gray-900">{formatMetricDisplay(row.key, row.leftValue)}</td>
                    <td className="px-3 py-2 font-mono text-gray-900">{formatMetricDisplay(row.key, row.rightValue)}</td>
                    <td className={`px-3 py-2 font-mono ${getDeltaTone(row.key, row.delta)}`}>{formatDelta(row.delta)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </section>
  );
}

function RunDetail({ runId }) {
  const [runDetailState, dispatchRunDetail] = useReducer(loadStateReducer, createLoadState());
  const summary = runDetailState.data;
  const loading = runDetailState.loading;
  const error = runDetailState.error;

  useEffect(() => {
    if (!runId) {
      dispatchRunDetail({ type: "reset" });
      return;
    }

    let isMounted = true;
    dispatchRunDetail({ type: "start" });

    getRunSummary(runId)
      .then((data) => {
        if (!isMounted) return;
        dispatchRunDetail({ type: "success", data });
      })
      .catch((err) => {
        if (!isMounted) return;
        dispatchRunDetail({ type: "error", error: err.message });
      });

    return () => {
      isMounted = false;
    };
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

      {!loading && !error && !summary && (
        <EmptyState
          title="No run selected"
          body="Select an evaluation run from the table to inspect its configuration, metrics, and error breakdown."
        />
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
  return `"${text.replace(/"/g, '""')}"`;
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

function RunBuilder() {
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
    } catch {
      setCopied(false);
    }
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
          <span className="text-xs text-gray-500">
            Run from the repository's <code>backend</code> folder.
          </span>
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
                Normal chat usage updates separate citation and answer feedback signals shown in the Feedback Health section.
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
                  <span className="text-sm text-gray-500">No execution modes were reported by the backend.</span>
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
                  <span className="text-sm text-gray-500">No case modes were found in the evaluation case files.</span>
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
  const [accessState, dispatchAccess] = useReducer(loadStateReducer, createLoadState(true, true));
  const [runsState, dispatchRuns] = useReducer(loadStateReducer, createLoadState([], false));
  const [selectedRunId, setSelectedRunId] = useState(null);
  const [latestSummaryState, dispatchLatestSummary] = useReducer(loadStateReducer, createLoadState());
  const [feedbackState, dispatchFeedback] = useReducer(loadStateReducer, createLoadState());
  const [dashboardMetaState, dispatchDashboardMeta] = useReducer(loadStateReducer, createLoadState());
  const [feedbackRefreshToken, setFeedbackRefreshToken] = useState(0);

  const accessLoading = accessState.loading;
  const accessError = accessState.error;
  const runs = runsState.data;
  const runsLoading = runsState.loading;
  const runsError = runsState.error;
  const latestSummary = latestSummaryState.data;
  const latestSummaryError = latestSummaryState.error;
  const feedbackSummary = feedbackState.data;
  const feedbackLoading = feedbackState.loading;
  const feedbackError = feedbackState.error;
  const dashboardMeta = dashboardMetaState.data;
  const dashboardMetaError = dashboardMetaState.error;

  useEffect(() => {
    let isMounted = true;
    dispatchAccess({ type: "start", keepData: true });

    checkDashboardAccess()
      .then(() => {
        if (!isMounted) return;
        dispatchAccess({ type: "success", data: true });
      })
      .catch((err) => {
        if (!isMounted) return;
        dispatchAccess({ type: "error", error: err.message });
      });

    return () => {
      isMounted = false;
    };
  }, []);

  useEffect(() => {
    if (accessLoading || accessError) return;
    let isMounted = true;
    dispatchRuns({ type: "start", data: [] });

    getRuns()
      .then((data) => {
        if (!isMounted) return;
        const list = data.runs || [];
        dispatchRuns({ type: "success", data: list });
        setSelectedRunId((current) => {
          if (current && list.some((run) => run.run_id === current)) return current;
          return list[0]?.run_id || null;
        });
      })
      .catch((err) => {
        if (!isMounted) return;
        dispatchRuns({ type: "error", error: err.message, data: [] });
      });

    return () => {
      isMounted = false;
    };
  }, [accessLoading, accessError]);

  useEffect(() => {
    if (accessLoading || accessError) return;
    let isMounted = true;
    dispatchDashboardMeta({ type: "start" });

    getDashboardMeta()
      .then((data) => {
        if (!isMounted) return;
        dispatchDashboardMeta({ type: "success", data });
      })
      .catch((err) => {
        if (!isMounted) return;
        dispatchDashboardMeta({ type: "error", error: err.message });
      });

    return () => {
      isMounted = false;
    };
  }, [accessLoading, accessError]);

  const latestRunId = useMemo(() => runs[0]?.run_id || null, [runs]);

  useEffect(() => {
    if (accessLoading || accessError) return;
    if (!latestRunId) {
      dispatchLatestSummary({ type: "reset" });
      return;
    }

    let isMounted = true;
    dispatchLatestSummary({ type: "start" });

    getRunSummary(latestRunId)
      .then((data) => {
        if (!isMounted) return;
        dispatchLatestSummary({ type: "success", data });
      })
      .catch((err) => {
        if (!isMounted) return;
        dispatchLatestSummary({ type: "error", error: err.message });
      });

    return () => {
      isMounted = false;
    };
  }, [latestRunId, accessLoading, accessError]);

  useEffect(() => {
    if (accessLoading || accessError) return;
    let isMounted = true;
    dispatchFeedback({ type: "start", keepData: true });

    getFeedbackSummary()
      .then((data) => {
        if (!isMounted) return;
        dispatchFeedback({ type: "success", data });
      })
      .catch((err) => {
        if (!isMounted) return;
        dispatchFeedback({ type: "error", error: err.message, keepData: true });
      });

    return () => {
      isMounted = false;
    };
  }, [accessLoading, accessError, feedbackRefreshToken]);

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
        <FeedbackHealth
          summary={feedbackSummary}
          error={feedbackError}
          loading={feedbackLoading}
          onRefresh={() => setFeedbackRefreshToken((value) => value + 1)}
        />
        <RunComparison runs={runs} />
        <RunBuilder />

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
