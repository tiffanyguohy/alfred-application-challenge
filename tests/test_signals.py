"""Tests for alfred.signals — deterministic signal computation."""
from alfred.scenarios import SCENARIOS
from alfred.signals import compute_signals
from alfred.types import (
    ComputedSignals,
    Contact,
    DecisionInput,
    Message,
    ProposedAction,
    UserState,
)


def _make_input(
    action_type: str,
    parameters: dict,
    history: list[Message] | None = None,
    user_state: UserState | None = None,
    description: str = "test action",
) -> DecisionInput:
    return DecisionInput(
        proposed_action=ProposedAction(
            action_type=action_type,
            parameters=parameters,
            description=description,
        ),
        history=history or [],
        user_state=user_state or UserState(),
    )


def test_reversibility_email_send_is_irreversible():
    inp = _make_input(
        "email_send",
        {"to": "a@b.com", "subject": "hi", "body": "hello"},
    )
    assert compute_signals(inp).reversibility == "irreversible"


def test_reversibility_reminder_is_reversible():
    inp = _make_input("reminder_create", {"text": "call mom", "time": "18:00"})
    assert compute_signals(inp).reversibility == "reversible"


def test_blast_radius_email_send_is_external():
    inp = _make_input(
        "email_send",
        {"to": "a@b.com", "subject": "hi", "body": "hello"},
    )
    assert compute_signals(inp).blast_radius == "external"


def test_blast_radius_calendar_read_is_self():
    inp = _make_input("calendar_read", {"date_range": "tomorrow"})
    assert compute_signals(inp).blast_radius == "self"


def test_blast_radius_wire_transfer_is_financial():
    inp = _make_input(
        "wire_transfer",
        {"recipient": "Pixel Partners", "amount": 5000, "currency": "USD"},
    )
    assert compute_signals(inp).blast_radius == "financial"


def test_missing_parameters_detects_all_missing():
    inp = _make_input("email_send", {})
    missing = compute_signals(inp).missing_parameters
    assert set(missing) == {"to", "subject", "body"}


def test_missing_parameters_empty_when_all_present():
    inp = _make_input(
        "email_send",
        {"to": "a@b.com", "subject": "hi", "body": "hello"},
    )
    assert compute_signals(inp).missing_parameters == []


def test_entity_ambiguity_three_sarahs():
    scenario = next(s for s in SCENARIOS if s.name == "ambiguous_three_sarahs")
    sig = compute_signals(scenario.input)
    assert "Sarah" in sig.entity_ambiguities
    assert len(sig.entity_ambiguities["Sarah"]) == 3


def test_entity_ambiguity_empty_when_full_email_given():
    user_state = UserState(
        contacts=[
            Contact(name="Sarah Chen", email="sarah.chen@acme.com"),
            Contact(name="Sarah Park", email="spark@internal.co"),
        ]
    )
    inp = _make_input(
        "email_send",
        {"to": "sarah.chen@acme.com", "subject": "hi", "body": "hey"},
        user_state=user_state,
    )
    assert compute_signals(inp).entity_ambiguities == {}


def test_risk_factors_include_irreversible_for_email_send():
    inp = _make_input(
        "email_send",
        {"to": "a@b.com", "subject": "hi", "body": "hello"},
    )
    assert "irreversible action" in compute_signals(inp).risk_contributing_factors


def test_risk_factors_include_financial_for_wire():
    inp = _make_input(
        "wire_transfer",
        {"recipient": "Vendor", "amount": 5000, "currency": "USD"},
    )
    factors = compute_signals(inp).risk_contributing_factors
    assert any("financial amount over $1,000" in f for f in factors)


def test_risk_factors_include_emotional_for_quit_scenario():
    scenario = next(s for s in SCENARIOS if s.name == "risky_quit_email")
    factors = compute_signals(scenario.input).risk_contributing_factors
    assert "emotionally charged recent messages" in factors


def test_intent_clarity_score_full_when_clear():
    inp = _make_input("reminder_create", {"text": "call mom", "time": "18:00"})
    assert compute_signals(inp).intent_clarity_score == 1.0


def test_intent_clarity_score_drops_with_missing_params():
    inp = _make_input("email_send", {})
    assert compute_signals(inp).intent_clarity_score < 1.0


def test_intent_clarity_score_drops_with_entity_ambiguity():
    scenario = next(s for s in SCENARIOS if s.name == "ambiguous_three_sarahs")
    assert compute_signals(scenario.input).intent_clarity_score < 1.0


def test_compute_signals_runs_on_all_scenarios():
    for scenario in SCENARIOS:
        sig = compute_signals(scenario.input)
        assert isinstance(sig, ComputedSignals)
