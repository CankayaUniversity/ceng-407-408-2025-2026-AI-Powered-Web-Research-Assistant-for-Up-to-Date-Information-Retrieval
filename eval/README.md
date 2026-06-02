# Accuracy eval

Two layers, smallest first.

## 1. Offline logic checks (no Ollama, no server)

Pins the deterministic primitives that decide correctness: the synthesis and
verifier answer-acceptance gates, and the source/recency-weighted conflict
resolvers. Fast, dependency-light (only needs the project venv), and the right
thing to run after touching `llm_passes.py`.

```bash
python eval/offline_checks.py
```

Exit code is non-zero if any check fails.

## 2. End-to-end eval (needs Ollama + the running API)

Sends each case in `cases.json` to `/ask_agent` and grades the answer by
expected substrings / regex, then prints accuracy and a per-signal comparison
of passing vs failing cases.

Accuracy escalation is automatic server-side, so there is no mode flag to
pass — every request runs the adaptive pipeline and escalates itself when the
draft looks weak.

```bash
# In one terminal:
ollama serve
uvicorn main:app

# In another:
python eval/run_eval.py
python eval/run_eval.py --model qwen
python eval/run_eval.py --min-accuracy 0.7   # exit non-zero below the gate
```

### Case format (`cases.json`)

```json
{
  "id": "short-label",
  "question": "the question to ask",
  "model": "llama",            // optional, overridden by --model
  "expect_all": ["substr", ...],// every substring must appear (case-insensitive)
  "expect_regex": "pattern",    // optional, must match
  "must_not": ["substr", ...]   // optional, none may appear
}
```

Grading is intentionally simple substring/regex matching. Live web answers are
time-sensitive, so seed `cases.json` with facts that are stable (capitals,
chemistry, history) and use `must_not` to catch hedging on volatile questions
you can't pin to an exact answer yet.

### Calibration

The trust-signal table at the end of a run shows the mean of each numeric
signal for passing vs failing cases, with the gap between them. Signals with a
large positive gap track correctness; signals with ~zero gap do not. That gap
is the evidence to tune tier/signal weights against, instead of guessing.
