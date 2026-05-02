# State-of-the-Art Reliability Design for Agentic RAG Diabetes Assistant

## 1. Problem and Goals
The current system needs a major reliability and quality upgrade, with **groundedness/faithfulness** as the primary success metric.

Primary goals:
1. Increase answer faithfulness and reduce unsupported claims.
2. Improve retrieval quality, especially on currently missed relevant evidence.
3. Enforce conservative medical-safety behavior under uncertainty.

Non-goals:
1. Broad product/UI redesign.
2. Unrelated refactors not tied to reliability and retrieval quality.

## 2. Proposed Architecture
The design introduces a two-lane reliability architecture:
1. **Deterministic Retrieval Lane** for high-quality evidence collection.
2. **Guarded Generation Lane** for strictly grounded answer production with reliability checks.

Policy layer behavior:
1. If evidence is weak or confidence is low, the assistant explicitly states uncertainty.
2. It advises clinician follow-up for safety-critical or weak-evidence outputs.

## 3. Component Design
### 3.1 Query Understanding Agent
Responsibilities:
1. Query rewrite for retrieval optimization.
2. Query decomposition into focused sub-queries.
3. Ambiguity tagging for downstream clarification logic.

Inputs: user query, chat history.  
Outputs: rewritten query, sub-query list, ambiguity flags.

### 3.2 Retrieval Orchestrator
Responsibilities:
1. Execute dense and lexical retrieval in parallel for each sub-query.
2. Merge and deduplicate candidates.
3. Balance source diversity (local corpus and optional web fallback).

Inputs: rewritten query, sub-queries.  
Outputs: candidate evidence pool with source metadata.

### 3.3 Rerank & Evidence Filter
Responsibilities:
1. Cross-encoder rerank of candidate chunks.
2. Apply evidence-quality thresholds.
3. Emit citation-ready snippets and provenance metadata.

Inputs: candidate evidence pool.  
Outputs: approved evidence set, evidence score, dropped-candidate diagnostics.

### 3.4 Answer Composer
Responsibilities:
1. Generate answer strictly from approved evidence.
2. Attach explicit citation mapping to supporting chunks.
3. Avoid unsupported claims by construction.

Inputs: approved evidence set, original user query, chat history.  
Outputs: grounded answer draft with citations.

### 3.5 Reliability Judge
Responsibilities:
1. Check faithfulness against cited evidence.
2. Detect contradictions and unsupported claims.
3. Return pass/fail with reason codes.

Inputs: answer draft, approved evidence set.  
Outputs: reliability verdict and issue taxonomy.

### 3.6 Medical Safety Guard
Responsibilities:
1. Enforce conservative communication under uncertainty.
2. Apply escalation templates for clinician follow-up when needed.
3. Block confident phrasing when reliability is not sufficient.

Inputs: reliability verdict, answer draft, evidence score.  
Outputs: final user response policy envelope.

### 3.7 Observability and Evaluation Layer
Responsibilities:
1. Capture per-stage metrics and failure categories.
2. Support regression evaluation for prompts/models/retrieval changes.
3. Provide traceability for missed-evidence diagnosis.

Inputs: runtime traces and evaluation datasets.  
Outputs: metric reports and regression gate decisions.

## 4. End-to-End Data Flow
1. User query enters Query Understanding Agent.
2. Rewritten/decomposed queries go to Retrieval Orchestrator.
3. Orchestrator returns merged candidates to Rerank & Evidence Filter.
4. If evidence score is below threshold:
   1. Trigger targeted recovery retrieval.
   2. If still weak, emit clarification or conservative uncertainty response.
5. If evidence score passes threshold:
   1. Answer Composer creates citation-grounded draft.
   2. Reliability Judge validates faithfulness and contradictions.
   3. Medical Safety Guard enforces conservative policy.
6. Final response includes confidence framing and evidence citations.

## 5. Error Handling and Reliability Policy
1. No silent fallbacks to empty-context generation.
2. Retrieval failure paths are explicit and user-visible.
3. Judge or safety failures force conservative output templates.
4. External tool failures (e.g., web search) degrade gracefully:
   1. Continue local path when available.
   2. Downgrade confidence and communicate limitations.
5. Maximum retry limits are explicit, with deterministic terminal behavior.

## 6. Testing and Success Criteria
Primary KPI:
1. **Faithfulness/groundedness score on the existing evaluation set**.

Secondary KPIs:
1. Context precision and context recall.
2. Unsupported-claim rate.
3. Clarification-trigger quality and rate.
4. Conservative-fallback correctness.

Evaluation strategy:
1. Add failure-mode slices:
   1. Missing evidence.
   2. Noisy evidence.
   3. Conflicting evidence.
   4. Ambiguous user query.
2. Run regression checks before accepting retrieval/prompt/model changes.
3. Block merges when faithfulness regresses beyond agreed thresholds.

## 7. Implementation Boundaries
In scope:
1. Retrieval pipeline upgrade (decomposition, hybrid retrieval, reranking, thresholds).
2. Grounded generation controls and citation mapping.
3. Reliability judge + medical safety guard integration.
4. Observability and evaluation gates aligned to faithfulness.

Out of scope:
1. Feature work unrelated to answer reliability and retrieval quality.
2. Broad interface redesign not required for these reliability controls.
