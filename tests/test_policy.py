"""Tests for alfred.policy — hard deterministic policy rules."""
from alfred.conversation_state import derive_conversation_state
from alfred.policy import check_policy
from alfred.scenarios import SCENARIOS
from alfred.signals import compute_signals
from alfred.types import (
    DecisionInput,
    DecisionType,
    Message,
    PolicyResult,
    ProposedAction,
    UserState,
)


def _build(
    action_type: str,
    parameters: dict,
    user_state: UserState | None = None,
    history: list[Message] | None = None,
) -> DecisionInput:
    return DecisionInput(
        proposed_action=ProposedAction(
            action_type=action_type,
            parameters=parameters,
            description=f"test {action_type}",
        ),
        history=history or [],
        user_state=user_state or UserState(),
    )


def _run(inp: DecisionInput) -> PolicyResult:
    signals = compute_signals(inp)
    conv_state = derive_conversation_state(inp.history)
    return check_policy(inp, signals, conv_state)


def test_wire_over_10x_threshold_refuses():
    scenario = next(s for s in SCENARIOS if s.name == "risky_wire_50k")
    result = _run(scenario.input)
    assert result.violated is True
    assert result.forced_decision == DecisionType.REFUSE_OR_ESCALATE


def test_wire_over_threshold_confirms():
    inp = _build(
        "wire_transfer",
        {"recipient": "Acme", "amount": 5000, "currency": "USD"},
        user_state=UserState(financial_confirmation_threshold_usd=1000.0),
    )
    result = _run(inp)
    assert result.violated is True
    assert result.forced_decision == DecisionType.CONFIRM_BEFORE_EXECUTING


def test_wire_under_threshold_no_violation():
    inp = _build(
        "wire_transfer",
        {"recipient": "Acme", "amount": 500, "currency": "USD"},
        user_state=UserState(financial_confirmation_threshold_usd=1000.0),
    )
    result = _run(inp)
    assert result.violated is False
    assert result.forced_decision is None


def test_bulk_email_to_all_confirms():
    inp = _build(
        "email_send",
        {"to": "all", "subject": "hi", "body": "hello everyone"},
    )
    result = _run(inp)
    assert result.violated is True
    assert result.forced_decision == DecisionType.CONFIRM_BEFORE_EXECUTING


def test_bulk_email_many_recipients_confirms():
    inp = _build(
        "email_send",
        {
            "to": ["a@x.com", "b@x.com", "c@x.com", "d@x.com", "e@x.com", "f@x.com"],
            "subject": "hi",
            "body": "hello",
        },
    )
    result = _run(inp)
    assert result.violated is True
    assert result.forced_decision == DecisionType.CONFIRM_BEFORE_EXECUTING


def test_bulk_email_five_recipients_is_boundary_no_violation():
    inp = _build(
        "email_send",
        {
            "to": ["a@x.com", "b@x.com", "c@x.com", "d@x.com", "e@x.com"],
            "subject": "hi",
            "body": "hello",
        },
    )
    result = _run(inp)
    assert result.violated is False


def test_missing_params_on_email_send_is_ask_clarifying():
    inp = _build("email_send", {})
    result = _run(inp)
    assert result.violated is True
    assert result.forced_decision == DecisionType.ASK_CLARIFYING_QUESTION


def test_missing_params_on_reminder_not_violation():
    inp = _build("reminder_create", {})
    result = _run(inp)
    assert result.violated is False


def test_silent_send_disabled_is_reason_not_violation():
    inp = _build(
        "email_send",
        {"to": "a@b.com", "subject": "hi", "body": "hey"},
        user_state=UserState(silent_send_enabled=False),
    )
    result = _run(inp)
    assert result.violated is False
    assert any("silent" in r.lower() for r in result.reasons)


def test_silent_send_enabled_omits_silent_path_reason():
    inp = _build(
        "email_send",
        {"to": "a@b.com", "subject": "hi", "body": "hey"},
        user_state=UserState(silent_send_enabled=True),
    )
    result = _run(inp)
    assert result.violated is False
    assert not any("silent" in r.lower() for r in result.reasons)


def test_rule_1_wins_over_rule_4_when_both_would_fire():
    # Wire over 10x threshold AND missing the `currency` param (irreversible).
    # Rule 1 (refuse) must win over Rule 4 (ask_clarifying).
    inp = _build(
        "wire_transfer",
        {"recipient": "Acme", "amount": 50000},  # missing currency
        user_state=UserState(financial_confirmation_threshold_usd=1000.0),
    )
    result = _run(inp)
    assert result.forced_decision == DecisionType.REFUSE_OR_ESCALATE


def test_check_policy_runs_on_all_scenarios():
    for scenario in SCENARIOS:
        result = _run(scenario.input)
        assert isinstance(result, PolicyResult)
