# m12 — Reasoning traces from a teacher

**Paper:** §3 (Eqs. 11–12), Appendices C–D, Table 5 rows "targeted" vs "rejection".
**Time:** ~half a day of coding + 1–3 h of generation. **Files you implement:** `teacher.py`, `traces.py`, `run.py`.

## Two ways to get good explanations

**Targeted sampling (Eq. 11):** show the teacher the answer (the m09 `"targeted"` prompt) and ask it
to reason toward it. One call per example, but the reasoning can be post-hoc rationalization.

**Rejection sampling (Eq. 12):** don't show the answer (the `"rejection"` prompt). Sample up to
`max_attempts` times, and **keep a trace only if the teacher's top-1 is the true item.** It costs more
calls, but the reasoning actually led to the answer. In the paper, rejection traces give the best model.

## Checking a trace (`check_trace`)

A trace is accepted only if:
1. m10 parses a ranking from it
2. its top-1 is the target (`ranking[0] == target_rank − 1`). Even targeted traces must pass this,
   because teachers sometimes ignore the hint.
3. the explanation is non-empty
4. **no hallucinated SIDs:** every SID the explanation mentions appears in the prompt (`candidate_sids`)

## What gets saved: the student's view

Each JSONL record stores the conversation **the student will train on**: the **policy** prompt (no hint!)
plus the teacher's answer rewritten in canonical form, `assistant_json(reasoning, ranking_1based)`.
Training on the targeted prompt would teach the student to expect a hint it never gets at test time.

```json
{"user": 17, "mode": "rejection", "accepted": true, "attempts": 2, "reason": "ok",
 "messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", "content": "{\"explanation\": ...}"}],
 "raw": "<the teacher's last raw output>"}
```

`generate_traces` is **resumable**. It reads the output file first and skips users already present for
that mode, then appends one line per example and flushes immediately. If it crashes after 3 hours,
rerunning continues where it stopped.

## The teacher: any OpenAI-compatible server

`OpenAICompatClient` POSTs to `{base_url}/chat/completions`. The same code talks to:

| Option | `traces.base_url` | `traces.model` | cost |
|---|---|---|---|
| Ollama (local, free): `ollama pull qwen3:4b` | `http://localhost:11434/v1` | `qwen3:4b` | free, ~10–20 s/trace on your GPU |
| Ollama, bigger | same | `qwen3:8b` (≈5 GB, partly on CPU) | free, slower, better |
| LM Studio (local) | `http://localhost:1234/v1` | whatever you loaded | free |
| A hosted API | the provider's OpenAI-compatible URL | their model id | a few $ for ~1k traces |

For a hosted API, set `traces.api_key_env` to the *name* of an environment variable holding the key.
Never put the key itself in the config. `n` samples are sent as `n` separate requests (not every server
supports `n > 1`). Qwen3 teachers may emit `<think>` blocks: m10 strips them.

## Implementation order & test

`teacher.py`: `OpenAICompatClient` → `traces.py`: `check_trace` → `sample_trace` → `generate_traces` → `run.py`

```powershell
python -m pytest gr2mini/m12_traces -v       # uses a fake teacher; no server needed
python -m gr2mini.m12_traces.run --config configs/tiny.yaml --mode rejection
python -m gr2mini.m12_traces.run --config configs/tiny.yaml --mode targeted
```

Expect rejection acceptance of ~20–40 % per attempt with a 4B teacher (random guessing would give 10 %).
**Report it**: it measures how good the teacher is at the task.
