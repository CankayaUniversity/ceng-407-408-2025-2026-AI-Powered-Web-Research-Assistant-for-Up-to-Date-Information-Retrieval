"""End-to-end accuracy eval against a running server.

Sends each case in cases.json to the legacy non-streaming endpoint
(/ask_agent) and grades the answer by expected substrings / regex. It also
aggregates the numeric trust signals for passing vs failing cases, so you can
see which signals actually track correctness — the input you need before
calibrating tier weights instead of guessing them.

Prerequisites: Ollama running and `uvicorn main:app` serving the API.

Examples:
  python eval/run_eval.py
  python eval/run_eval.py --model qwen --high-accuracy
  python eval/run_eval.py --base-url http://127.0.0.1:8000 --min-accuracy 0.7
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))


def load_cases(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        cases = json.load(handle)
    if not isinstance(cases, list):
        raise ValueError("cases file must contain a JSON array")
    return cases


def ask(base_url: str, question: str, model: str | None, timeout: float) -> dict:
    params = {"question": question}
    if model:
        params["model"] = model
    url = f"{base_url.rstrip('/')}/ask_agent?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def grade(answer: str, case: dict) -> tuple[bool, str]:
    text = (answer or "").lower()
    for needle in case.get("expect_all", []):
        if str(needle).lower() not in text:
            return False, f"missing expected text: {needle!r}"
    expect_regex = case.get("expect_regex")
    if expect_regex and not re.search(expect_regex, answer or "", re.IGNORECASE):
        return False, f"regex did not match: {expect_regex!r}"
    for needle in case.get("must_not", []):
        if str(needle).lower() in text:
            return False, f"contains forbidden text: {needle!r}"
    return True, "ok"


def _numeric_signals(signals: dict) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in (signals or {}).items():
        if isinstance(value, bool):
            out[key] = 1.0 if value else 0.0
        elif isinstance(value, (int, float)):
            out[key] = float(value)
    return out


def _mean_signals(rows: list[dict]) -> dict[str, float]:
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for row in rows:
        for key, value in _numeric_signals(row).items():
            totals[key] = totals.get(key, 0.0) + value
            counts[key] = counts.get(key, 0) + 1
    return {key: totals[key] / counts[key] for key in totals}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the accuracy eval against a running server.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--cases", default=os.path.join(HERE, "cases.json"))
    parser.add_argument("--model", default=None, help="model key override (e.g. llama, qwen, llama32)")
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--min-accuracy", type=float, default=None, help="exit non-zero if accuracy is below this")
    args = parser.parse_args()

    cases = load_cases(args.cases)
    passed = 0
    pass_signals: list[dict] = []
    fail_signals: list[dict] = []

    for index, case in enumerate(cases, start=1):
        question = case["question"]
        model = case.get("model", args.model)
        label = case.get("id") or question
        started = time.time()
        try:
            result = ask(args.base_url, question, model, args.timeout)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            print(f"[{index:02d}] ERROR  {label}\n        request failed: {exc}")
            fail_signals.append({})
            continue

        if "error" in result:
            print(f"[{index:02d}] ERROR  {label}\n        server error: {result['error']}")
            fail_signals.append({})
            continue

        answer = result.get("agent_answer", "")
        ok, reason = grade(answer, case)
        elapsed = time.time() - started
        signals = result.get("trust_signals", {})
        (pass_signals if ok else fail_signals).append(signals)
        passed += int(ok)
        status = "PASS " if ok else "FAIL "
        print(f"[{index:02d}] {status} {label}  ({elapsed:.1f}s)")
        if not ok:
            print(f"        {reason}")
            print(f"        answer: {answer[:200]}")

    total = len(cases)
    accuracy = passed / total if total else 0.0
    print("\n" + "=" * 60)
    print(f"Accuracy: {passed}/{total} = {accuracy:.0%}")

    pass_means = _mean_signals(pass_signals)
    fail_means = _mean_signals(fail_signals)
    shared = sorted(set(pass_means) & set(fail_means))
    if shared:
        print("\nMean trust signal — passing vs failing (gap shows what tracks correctness):")
        print(f"  {'signal':<34}{'pass':>8}{'fail':>8}{'gap':>8}")
        for key in shared:
            p, f = pass_means[key], fail_means[key]
            print(f"  {key:<34}{p:>8.2f}{f:>8.2f}{p - f:>8.2f}")

    if args.min_accuracy is not None and accuracy < args.min_accuracy:
        print(f"\nAccuracy {accuracy:.0%} is below the --min-accuracy gate of {args.min_accuracy:.0%}.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
