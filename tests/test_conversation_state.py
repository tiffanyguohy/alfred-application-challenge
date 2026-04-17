"""Tests for alfred.conversation_state — deriving state from chat history."""
from alfred.conversation_state import derive_conversation_state
from alfred.scenarios import SCENARIOS
from alfred.types import ConversationState, Message


def _msg(role: str, content: str, ts: str = "2026-04-17T09:00:00") -> Message:
    return Message(role=role, content=content, timestamp=ts)


def test_pending_constraints_detects_hold_off():
    history = [_msg("user", "hold off on that email for now")]
    state = derive_conversation_state(history)
    assert len(state.pending_constraints) == 1


def test_pending_constraints_detects_wait_until():
    history = [_msg("user", "wait until legal approves before sending")]
    state = derive_conversation_state(history)
    assert len(state.pending_constraints) == 1


def test_pending_constraints_empty_on_plain_history():
    history = [_msg("user", "Remind me to call mom at 6pm")]
    state = derive_conversation_state(history)
    assert state.pending_constraints == []


def test_pending_constraints_not_resolved_by_generic_yep():
    scenario = next(s for s in SCENARIOS if s.name == "ambiguous_yep_send_it")
    state = derive_conversation_state(scenario.input.history)
    assert state.pending_constraints, "Generic 'yep send it' must not resolve the legal hold"


def test_pending_constraints_persist_despite_legal_approved():
    # Resolution is the LLM's job (per CLAUDE.md code/LLM split). The deterministic
    # layer surfaces the hold; even "legal approved" must not auto-clear it here.
    scenario = next(s for s in SCENARIOS if s.name == "ambiguous_yep_send_it")
    extended = list(scenario.input.history) + [
        _msg("user", "legal approved the terms", "2026-04-17T15:00:00")
    ]
    state = derive_conversation_state(extended)
    assert state.pending_constraints, "Resolution judgment belongs to the LLM, not code"


def test_pause_is_not_triggered_by_pause_my_subscription():
    # Bare "pause " was too loose; now we require "pause on/that/the".
    history = [_msg("user", "I want to pause my subscription when it renews")]
    state = derive_conversation_state(history)
    assert state.pending_constraints == []


def test_unresolved_references_at_seven_word_boundary_flags():
    history = [_msg("user", "please send it to them right now")]  # 7 words, short
    state = derive_conversation_state(history)
    assert state.unresolved_references, "7-word message with 'it' should flag unresolved"


def test_unresolved_references_at_eight_word_boundary_does_not_flag():
    history = [_msg("user", "please send it to the finance team right now")]  # 8 words
    state = derive_conversation_state(history)
    assert state.unresolved_references == []


def test_unresolved_references_detects_short_it_message():
    history = [_msg("user", "send it")]
    state = derive_conversation_state(history)
    assert state.unresolved_references, "Short 'send it' should flag an unresolved reference"


def test_unresolved_references_empty_on_long_message():
    history = [
        _msg(
            "user",
            "Please send the quarterly revenue report to the finance team today",
        )
    ]
    state = derive_conversation_state(history)
    assert state.unresolved_references == []


def test_last_drafted_artifact_detects_draft_mention():
    scenario = next(s for s in SCENARIOS if s.name == "ambiguous_yep_send_it")
    state = derive_conversation_state(scenario.input.history)
    assert state.last_drafted_artifact is not None
    assert len(state.last_drafted_artifact) <= 100
    # Should latch onto the 09:16 message (the actual draft with Subject line),
    # not a later message that merely references the draft.
    assert "09:16" in state.last_drafted_artifact
    assert "Acme" in state.last_drafted_artifact


def test_awaiting_confirmation_detects_trailing_question():
    history = [
        _msg("user", "Draft the email", "2026-04-17T09:00:00"),
        _msg("alfred", "Here's the draft. Want me to send it?", "2026-04-17T09:01:00"),
    ]
    state = derive_conversation_state(history)
    assert state.awaiting_confirmation is not None


def test_awaiting_confirmation_none_when_user_has_replied():
    history = [
        _msg("user", "Draft the email", "2026-04-17T09:00:00"),
        _msg("alfred", "Here's the draft. Want me to send it?", "2026-04-17T09:01:00"),
        _msg("user", "not yet", "2026-04-17T09:02:00"),
    ]
    state = derive_conversation_state(history)
    assert state.awaiting_confirmation is None


def test_derive_runs_on_all_scenarios():
    for scenario in SCENARIOS:
        state = derive_conversation_state(scenario.input.history)
        assert isinstance(state, ConversationState)
