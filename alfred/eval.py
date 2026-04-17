"""These scenarios are a seed eval set, not demo examples.

We treat correctness as a measurable property — this runner is the harness you'd
extend with adversarial red-team cases, per-user-policy variants, and production
traces as the system evolves.
"""
from __future__ import annotations

import os
import sys

from pydantic import BaseModel

from alfred.decide import decide
from alfred.scenarios import SCENARIOS
from alfred.types import DecisionType


class EvalResult(BaseModel):
    scenario_name: str
    category: str
    must_pass: bool
    expected_decision: DecisionType
    actual_decision: DecisionType
    decision_source: str
    passed: bool
    notes: str
    timing_ms: float
    error: str | None = None


def run_evals() -> list[EvalResult]:
    results: list[EvalResult] = []
    for scenario in SCENARIOS:
        error: str | None = None
        try:
            result = decide(scenario.input)
            actual = result.final_decision
            source = result.decision_source
            timing = result.timings_ms.get("total_ms", 0.0)
        except Exception as exc:  # noqa: BLE001 — harness reports any failure
            error = f"{type(exc).__name__}: {exc}"
            actual = DecisionType.REFUSE_OR_ESCALATE
            source = "fallback"
            timing = 0.0

        passed = error is None and actual == scenario.expected_decision
        results.append(
            EvalResult(
                scenario_name=scenario.name,
                category=scenario.category,
                must_pass=scenario.must_pass,
                expected_decision=scenario.expected_decision,
                actual_decision=actual,
                decision_source=source,
                passed=passed,
                notes=scenario.notes,
                timing_ms=timing,
                error=error,
            )
        )
    return results


def _truncate(text: str, width: int) -> str:
    if len(text) <= width:
        return text
    if width <= 1:
        return text[:width]
    return text[: width - 1] + "\u2026"


def print_eval_table(results: list[EvalResult]) -> None:
    widths = {
        "num": 2,
        "name": 40,
        "category": 16,
        "expected": 26,
        "actual": 26,
        "source": 14,
        "pass": 4,
        "time": 7,
        "notes": 60,
    }

    header = (
        f"{'#':>{widths['num']}}  "
        f"{'Name':<{widths['name']}}  "
        f"{'Category':<{widths['category']}}  "
        f"{'Expected':<{widths['expected']}}  "
        f"{'Actual':<{widths['actual']}}  "
        f"{'Source':<{widths['source']}}  "
        f"{'Pass':<{widths['pass']}}  "
        f"{'Time(ms)':>{widths['time']}}  "
        f"{'Notes':<{widths['notes']}}"
    )
    divider = "-" * len(header)

    print(header)
    print(divider)

    for i, r in enumerate(results, start=1):
        if r.error is not None:
            pass_marker = "ERR"
        elif r.passed:
            pass_marker = "YES"
        else:
            pass_marker = "NO"

        row = (
            f"{i:>{widths['num']}}  "
            f"{_truncate(r.scenario_name, widths['name']):<{widths['name']}}  "
            f"{_truncate(r.category, widths['category']):<{widths['category']}}  "
            f"{_truncate(r.expected_decision.value, widths['expected']):<{widths['expected']}}  "
            f"{_truncate(r.actual_decision.value, widths['actual']):<{widths['actual']}}  "
            f"{_truncate(r.decision_source, widths['source']):<{widths['source']}}  "
            f"{pass_marker:<{widths['pass']}}  "
            f"{round(r.timing_ms):>{widths['time']}}  "
            f"{_truncate(r.notes, widths['notes']):<{widths['notes']}}"
        )
        print(row)
        if r.error is not None:
            print(f"    error: {r.error}")

    total = len(results)
    passed_count = sum(1 for r in results if r.passed)
    must_pass_total = sum(1 for r in results if r.must_pass)
    must_pass_passed = sum(1 for r in results if r.must_pass and r.passed)
    print(divider)
    print(
        f"PASS: {passed_count}/{total} "
        f"(must_pass: {must_pass_passed}/{must_pass_total})"
    )


def _main() -> int:
    try:
        from dotenv import load_dotenv  # type: ignore[import-not-found]

        load_dotenv()
    except ImportError:
        pass

    if not os.getenv("ANTHROPIC_API_KEY"):
        print(
            "ANTHROPIC_API_KEY is not set. The eval harness calls the real "
            "Anthropic API — please export a key (or populate .env) before "
            "running `python -m alfred.eval`.",
            file=sys.stderr,
        )
        return 2

    results = run_evals()
    print_eval_table(results)

    must_pass_failures = [r for r in results if r.must_pass and not r.passed]
    return 0 if not must_pass_failures else 1


if __name__ == "__main__":
    sys.exit(_main())
