# Core Metrics: Step-by-Step

This project now uses a **strict core metric set**.
We removed proxy/heuristic metrics that were noisy or hard to defend.

## One Example Case Used Below

Question:
`For an RFP under a supply arrangement, when are reciprocal procurement declarations required and what must buyers verify before contract award?`

Case labels (simplified):
- `gold_relevant_doc_prefixes = [A, B]`
- `claim_evidence` has 3 claim entries
- `required_claims` has 3 claims
- `forbidden_claims` has 1 claim
- `expect_abstain = false`
- `reference_answer` exists

Retrieved top-k doc prefixes:
- `[A, B, A, N, A]` where `N` is labeled noise.

Answer summary:
- Covers 2 of 3 required claims
- Does not violate forbidden claim
- Includes citations

---

## Retrieval Metrics (Primary)

### 1) `retrieval_gold_doc_recall_at_k_mean`
What it asks:
- Did retrieval include the gold source docs?

Per-case formula:
- `gold_doc_recall_at_k = (# gold doc prefixes retrieved) / (# gold doc prefixes in case)`

Example:
- Gold docs `[A, B]`
- Retrieved includes both `A` and `B`
- Recall = `2 / 2 = 1.0`

Overall formula:
- Average this per-case recall across all cases with labels.

---

### 2) `retrieval_claim_evidence_coverage_mean`
What it asks:
- For each expected claim, did retrieval include at least one labeled evidence source?

Per-case formula:
- `claim_evidence_coverage = (# claim_evidence items covered) / (total claim_evidence items)`

Example:
- 3 claim-evidence items, all 3 covered
- Coverage = `3 / 3 = 1.0`

Overall:
- Mean across cases.

---

### 3) `retrieval_noise_rate_mean` (if labels exist)
What it asks:
- How much retrieved top-k content is known off-topic/noisy?

Per-case formula:
- `noise_rate = (# retrieved chunks whose doc prefix is in noise_doc_prefixes) / k`

Example:
- 1 noisy item in top-5
- Noise rate = `1 / 5 = 0.2`

Overall:
- Mean across labeled cases.

---

### 4) `retrieval_contradiction_rate_mean` (if labels exist)
What it asks:
- How much retrieved top-k content is known contradictory content?

Per-case formula:
- `contradiction_rate = (# retrieved chunks in contradiction_doc_prefixes) / k`

Example:
- If 0 contradictions in top-5: `0 / 5 = 0.0`

Overall:
- Mean across labeled cases.

---

## Answer Metrics (Primary)

### 5) `answer_required_claim_recall_mean`
What it asks:
- Did the answer include the required policy content?

Per-case:
- For each required claim, evaluator checks match using lexical+semantic rules.
- `required_claim_recall = (# required claims hit) / (total required claims)`

Example:
- 2 hits out of 3 required claims
- Recall = `2 / 3 = 0.6667`

Overall:
- Mean across cases.

---

### 6) `answer_forbidden_violation_rate`
What it asks:
- How often does answer assert forbidden claims?

Per-case:
- `forbidden_claim_violation_rate = (# forbidden claims violated) / (total forbidden claims)`

Example:
- 0 violations out of 1 forbidden claim
- Rate = `0 / 1 = 0.0`

Overall:
- Mean across cases that define forbidden claims.

---

### 7) `answer_citation_support_rate_mean`
What it asks:
- When answer cites chunks, does each cited sentence align with cited chunk text?

Per-case:
- For each sentence+citation pair, compute semantic similarity.
- Mark supported if similarity >= threshold.
- `citation_support_rate = supported_pairs / total_pairs`

Example:
- 3 supported out of 4 scored citation pairs
- Rate = `3 / 4 = 0.75`

Overall:
- Mean across cases with scorable citation pairs.

---

### 8) `answer_abstention_accuracy`
What it asks:
- Does the answer abstain when it should, and avoid abstaining when it should not?

Per-case:
- Detect abstain phrase in answer (for example: “insufficient information”).
- Compare to `expect_abstain`.
- `abstention_correct = true/false`

Example:
- `expect_abstain = false`, answer does not abstain -> `true`

Overall:
- Boolean converted to 1/0, then averaged.

---

### 9) `answer_reference_similarity_mean` (if `reference_answer` exists)
What it asks:
- How close is answer meaning to canonical reference answer?

Per-case:
- Embed full answer and `reference_answer`.
- Compute embedding similarity (dot product on normalized vectors).

Example:
- Similarity = `0.71`

Overall:
- Mean across cases with reference answers.

---

## Judge Metrics (Secondary)

Judge metrics run only with `--with-judge`.
Judge is useful, but **must be calibrated with human review**.

### 10) `judge_decision_correctness_mean`
Judge gives per-case score:
- 0 = wrong
- 1 = partial
- 2 = correct

Overall:
- Mean score across judged cases.

---

### 11) `judge_required_claim_recall_mean`
Judge checks required claims and returns recall in `[0,1]`.

Overall:
- Mean judge recall across cases.

---

### 12) `judge_forbidden_claim_violation_mean`
Judge returns 0/1 violation flag.

Overall:
- Mean across cases (interpretable as violation rate).

---

### 13) `judge_reference_alignment_mean` (if `reference_answer` exists)
Judge scores answer alignment vs canonical reference:
- 0/1/2

Overall:
- Mean across cases with reference answers.

---

## Timing Metrics

Reported under `timing_summary_ms`:
- `retrieval` p50/p95/mean
- `chat` p50/p95/mean (if chat enabled)
- `judge` p50/p95/mean (if judge enabled)
- `total` p50/p95/mean

Use these to catch latency regressions.

---

## Minimal Scoreboard to Watch

For weekly quality tracking, watch these 8:
1. `retrieval_gold_doc_recall_at_k_mean`
2. `retrieval_claim_evidence_coverage_mean`
3. `retrieval_noise_rate_mean`
4. `answer_required_claim_recall_mean`
5. `answer_forbidden_violation_rate`
6. `answer_citation_support_rate_mean`
7. `answer_abstention_accuracy`
8. `timing_summary_ms.total.p50`

