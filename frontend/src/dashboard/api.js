// Dashboard API helpers.

async function handleResponse(res, defaultMessage) {
  if (res.status === 404) {
    throw new Error("Dashboard access is not enabled.");
  }

  if (!res.ok) {
    let data = {};
    try {
      data = await res.json();
    } catch {
      data = {};
    }
    throw new Error(data.error || defaultMessage);
  }

  return res.json();
}

export async function getRuns() {
  const res = await fetch("/api/eval/runs");
  const data = await handleResponse(res, "Failed to load evaluation runs.");
  return {
    ...data,
    runs: Array.isArray(data?.runs) ? data.runs : [],
  };
}

export async function getRunSummary(runId) {
  const res = await fetch(`/api/eval/runs/${encodeURIComponent(runId)}/summary`);
  return handleResponse(res, "Failed to load run summary.");
}

export async function getFeedbackSummary() {
  const res = await fetch("/api/eval/feedback/summary");
  return handleResponse(res, "Failed to load feedback summary.");
}

export async function getDashboardMeta() {
  const res = await fetch("/api/eval/meta");
  return handleResponse(res, "Failed to load dashboard reference data.");
}

export async function checkDashboardAccess() {
  const res = await fetch("/api/eval/health");
  await handleResponse(res, "Dashboard access check failed.");
  return true;
}

export function getRunModeLabel(run) {
  if (!run || !run.with_chat) return "retrieval-only";
  if (run.with_judge) return "chat+judge";
  return "chat";
}
