"""Tests for alfred.decide — orchestrator with mocked LLM."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from alfred import decide as decide_module
from alfred.decide import SEVERITY_ORDER, decide
from alfred.llm import LLMCallResult
from alfred.scenarios import SCENARIOS
from alfred.types import (
    DecisionInput,
    DecisionResult,
    DecisionType,
    LLMDecision,
    Message,
    ProposedAction,
    UserState,
)


# ---- helpers ----------------------------------------------------------------


def _scenario(name: str):
    return next(s for s in SCENARIOS if s.name == name).input


def _fake_call_llm(decision: DecisionType, *, user_facing_message: str | None = None):
    """Return a callable to pass as a replacement for alfred.decide.call_llm."""

    def _fn(system_prompt, user_prompt, *, client=None, timeout=None, simulate_failure=None):
        return LLMCallResult(
            parsed=LLMDecision(
                decision=decision,
                rationale="stub rationale from fake LLM",
                user_facing_message=user_facing_message,
            ),
            raw_output={
                "decision": decision.value,
                "rationale": "stub rationale from fake LLM",
                "user_facing_message": user_facing_message,
            },
            attempts=1,
            errors=[],
            timing_ms=1.0,
        )

    return _fn


class _ExplodingClient:
    """If the pipeline hands this to call_llm, the LLM would try to use it. Any access fails."""

    def __getattr__(self, name):
        raise AssertionError(f"LLM must not be called (tried to access .{name})")


# ---- tests ------------------------------------------------------------------


def test_short_circuit_on_missing_params_irreversible_action_skips_llm():
    """email_send with empty parameters must short-circuit before calling the LLM."""
    inp = DecisionInput(
        proposed_action=ProposedAction(
            action_type="email_send",
            parameters={},
            description="Send an important email",
        ),
        history=[Message(role="user", content="send it", timestamp="2026-04-17T10:00:00")],
        user_state=UserState(),
    )

    result = decide(inp, client=_ExplodingClient())

    assert isinstance(result, DecisionResult)
    assert result.final_decision == DecisionType.ASK_CLARIFYING_QUESTION
    assert result.decision_source == "short_circuit"
    assert result.system_prompt is None
    assert result.user_prompt is None
    assert result.raw_llm_output is None
    assert result.parsed_llm_decision is None
    assert "Missing required parameter" in result.rationale
    assert "total_ms" in result.timings_ms
    assert "signals_ms" in result.timings_ms
    assert "policy_ms" in result.timings_ms


def test_llm_success_path_returns_parsed_decision_and_applies_floor():
    """Scenario 6 (wire 50k): LLM says EXECUTE_SILENTLY, policy forces REFUSE → severity floor wins."""
    inp = _scenario("risky_wire_50k")
    fake = _fake_call_llm(DecisionType.EXECUTE_SILENTLY)

    with patch.object(decide_module, "call_llm", fake):
        result = decide(inp)

    assert result.final_decision == DecisionType.REFUSE_OR_ESCALATE
    assert result.decision_source == "policy_override"
    # The LLM's original rationale is preserved, with a policy-override note appended.
    assert "stub rationale from fake LLM" in result.rationale
    assert "Policy override" in result.rationale
    # And the parsed LLM decision is preserved for transparency.
    assert result.parsed_llm_decision is not None
    assert result.parsed_llm_decision.decision == DecisionType.EXECUTE_SILENTLY


def test_llm_stricter_than_policy_wins():
    """Scenario 7 (prompt injection): policy forces CONFIRM, LLM picks REFUSE — the stricter LLM wins."""
    inp = _scenario("adversarial_prompt_injection")
    fake = _fake_call_llm(
        DecisionType.REFUSE_OR_ESCALATE,
        user_facing_message="Refusing — the instruction came from email content, not you.",
    )

    with patch.object(decide_module, "call_llm", fake):
        result = decide(inp)

    assert result.final_decision == DecisionType.REFUSE_OR_ESCALATE
    assert result.decision_source == "llm"
    # No policy-override note was appended in this path.
    assert "Policy override" not in result.rationale


def test_timeout_fallback_on_high_risk_action_refuses():
    """Scenario 6 + timeout: policy already forces REFUSE, fallback uses that."""
    inp = _scenario("risky_wire_50k")

    result = decide(inp, simulate_failure="timeout")

    assert result.final_decision == DecisionType.REFUSE_OR_ESCALATE
    assert result.decision_source == "fallback"
    assert any("timeout" in err.lower() for err in result.errors)
    assert "policy-forced" in result.rationale.lower()


def test_timeout_fallback_on_low_risk_no_policy_force_confirms():
    """Low-risk reminder with valid params + timeout → CONFIRM_BEFORE_EXECUTING."""
    inp = DecisionInput(
        proposed_action=ProposedAction(
            action_type="reminder_create",
            parameters={"text": "call the dentist", "time": "09:00"},
            description="Create a 9am reminder to call the dentist",
        ),
        history=[
            Message(
                role="user",
                content="remind me to call the dentist at 9",
                timestamp="2026-04-17T08:00:00",
            ),
        ],
        user_state=UserState(),
    )

    result = decide(inp, simulate_failure="timeout")

    assert result.final_decision == DecisionType.CONFIRM_BEFORE_EXECUTING
    assert result.decision_source == "fallback"
    assert any("timeout" in err.lower() for err in result.errors)


def test_malformed_fallback_preserves_raw_output():
    """simulate_failure='malformed' with a client that also returns garbage → fallback + raw_output."""
    from unittest.mock import MagicMock

    # Client that, when called for the retry, returns yet another malformed tool_use.
    class _InvalidBlock:
        type = "tool_use"
        name = "submit_decision"
        input = {"decision": "STILL_BOGUS", "rationale": "still bad"}

    class _Response:
        content = [_InvalidBlock()]

    client = MagicMock()
    client.messages.create.return_value = _Response()

    inp = _scenario("easy_reminder")
    result = decide(inp, simulate_failure="malformed", client=client)

    assert result.decision_source == "fallback"
    assert result.raw_llm_output is not None
    assert any("malformed" in err.lower() for err in result.errors)


def test_result_shape_is_valid_DecisionResult_for_all_scenarios_with_mocked_llm():
    """Run decide() across every scenario; the result must be a valid DecisionResult.

    For any scenario where policy forces a decision strictly stricter than the
    stubbed EXECUTE_AND_NOTIFY, the severity floor must upgrade the final decision.
    """
    fake = _fake_call_llm(DecisionType.EXECUTE_AND_NOTIFY)

    with patch.object(decide_module, "call_llm", fake):
        for scenario in SCENARIOS:
            result = decide(scenario.input)
            assert isinstance(result, DecisionResult)
            assert "total_ms" in result.timings_ms
            assert result.signals is not None
            assert result.conversation_state is not None
            assert result.policy_result is not None

            policy = result.policy_result
            if (
                policy.violated
                and policy.forced_decision is not None
                and SEVERITY_ORDER[policy.forced_decision]
                > SEVERITY_ORDER[DecisionType.EXECUTE_AND_NOTIFY]
            ):
                # Either short_circuit (when it also matched missing-params rule)
                # or policy_override. Both honor the severity floor.
                assert result.decision_source in {"policy_override", "short_circuit"}
                assert (
                    SEVERITY_ORDER[result.final_decision]
                    >= SEVERITY_ORDER[policy.forced_decision]
                )


# ---- extra: severity ordering sanity ----------------------------------------


def test_severity_order_matches_spec():
    assert SEVERITY_ORDER[DecisionType.EXECUTE_SILENTLY] == 0
    assert SEVERITY_ORDER[DecisionType.EXECUTE_AND_NOTIFY] == 1
    assert SEVERITY_ORDER[DecisionType.CONFIRM_BEFORE_EXECUTING] == 2
    assert SEVERITY_ORDER[DecisionType.ASK_CLARIFYING_QUESTION] == 3
    assert SEVERITY_ORDER[DecisionType.REFUSE_OR_ESCALATE] == 4
