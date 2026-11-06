# m12 — Reasoning traces from a teacher  🟡 preview

**Paper:** §3 (Eqs. 11–12), Appendices C–D, Table 5 rows "targeted" vs "rejection".
**Status:** lesson preview. Stubs and tests come when you get here.

## Two ways to get good explanations

**Targeted sampling (Eq. 11):** show the teacher the answer and ask it to reason toward it.
One call per example, but the reasoning can be *post-hoc rationalization* or hallucinated.

**Rejection sampling (Eq. 12):** don't show the answer. Sample, and **keep the trace only if the
teacher's top-1 is the true item.** More calls (the acceptance rate might be ~25–35 %), but the reasoning
actually *led* to the answer. The paper finds rejection traces give the best final model.

## Teacher options on your machine

The paper's teacher is Qwen3-32B. You'll write a small `TeacherClient` interface over the
**OpenAI-compatible chat API**, so any of these works without changing code:

| Teacher | Cost | Speed / quality |
|---|---|---|
| Ollama locally, `qwen3:4b` | free | ~10 s per trace, decent |
| Ollama locally, `qwen3:8b` (4-bit, ~5 GB) | free | slower (partly offloaded to CPU on 6 GB), better |
| A hosted API model | a few $ for ~1k traces | fast, parallel, best quality |

We'll decide together when you get here.

## Key ideas you'll learn

Resumable JSONL generation (crash-safe, so you never redo finished examples), retry and timeout
handling, acceptance-rate bookkeeping, and **trace filtering**: reject explanations that cite SIDs that
aren't in the prompt (hallucinated items).
