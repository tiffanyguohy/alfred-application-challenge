"""Main orchestrator for the Execution Decision Layer.

Pipeline:
  1. Compute deterministic signals from the action.
  2. Derive conversation state from history.
  3. Check policy (for a floor, not yet applied).
  4. Short-circuit if required params are missing on an irreversible action.
  5. Build prompts and call the LLM.
  6. Apply severity floor: policy can upgrade the LLM's decision, never downgrade.
  7. On LLM failure, fall back to the safest decision for the action's risk tier.
"""
from __future__ import annotations

import time
from typing import Any

from alfred import llm as llm_module
from alfred.conversation_state import derive_conversation_state
from alfred.llm import (
    LLMCallResult,
    LLMMalformedError,
    LLMTimeoutError,
    call_llm,
)
from alfred.policy import check_policy
from alfred.prompt import build_system_prompt, build_user_prompt
from alfred.signals import compute_signals
from alfred.types import (
    ComputedSignals,
    ConversationState,
    DecisionInput,
    DecisionResult,
    DecisionType,
    LLMDecision,
    PolicyResult,
)


# Severity ordering: 0 = most permissive, 4 = most restrictive.
# Policy-forced decisions act as a *floor* (minimum caution). The LLM can
# upgrade severity beyond the floor but cannot go below it.
SEVERITY_ORDER: dict[DecisionType, int] = {
    DecisionType.EXECUTE_SILENTLY: 0,
    DecisionType.EXECUTE_AND_NOTIFY: 1,
    DecisionType.CONFIRM_BEFORE_EXECUTING: 2,
    DecisionType.ASK_CLARIFYING_QUESTION: 3,
    DecisionType.REFUSE_OR_ESCALATE: 4,
}


# Anything in these sets gets the REFUSE fallback when the LLM fails with no
# policy-forced decision to fall back to.
_HIGH_RISK_BLAST_RADII = {"external", "financial", "legal"}


_FALLBACK_USER_MESSAGES: dict[DecisionType, str] = {
    DecisionType.REFUSE_OR_ESCALATE: (
        "I can't proceed with this action right now. Please try again or handle it manually."
    ),
    DecisionType.ASK_CLARIFYING_QUESTION: (
        "I need a bit more information before I can take that action."
    ),
    DecisionType.CONFIRM_BEFORE_EXECUTING: (
        "I'd like to confirm with you before taking this action. Proceed?"
    ),
}


def decide(
    input: DecisionInput,
    *,
    simulate_failure: str | None = None,
    client: Any = None,
    timeout: float | None = None,
) -> DecisionResult:
    """Run the full decision pipeline and return a DecisionResult."""
    t0 = time.perf_counter()
    timings: dict[str, float] = {}
    errors: list[str] = []

    # Step 2: signals.
    t_signals = time.perf_counter()
    signals = compute_signals(input)
    timings["signals_ms"] = _elapsed_ms(t_signals)

    # Step 3: conversation state.
    t_state = time.perf_counter()
    conv_state = derive_conversation_state(input.history)
    timings["conversation_state_ms"] = _elapsed_ms(t_state)

    # Step 4: policy (compute only; apply later).
    t_policy = time.perf_counter()
    policy_result = check_policy(input, signals, conv_state)
    timings["policy_ms"] = _elapsed_ms(t_policy)

    # Step 5: short-circuit when the action is irreversible and we don't have
    # the required parameters. No LLM call needed — we know we must ask.
    if signals.missing_parameters and signals.reversibility == "irreversible":
        timings["total_ms"] = _elapsed_ms(t0)
        return _short_circuit_result(
            input=input,
            signals=signals,
            conv_state=conv_state,
            policy_result=policy_result,
            timings=timings,
            errors=errors,
        )

    # Step 6: prompts.
    t_prompt = time.perf_counter()
    system_prompt = build_system_prompt()
    user_prompt = build_user_prompt(input, signals, conv_state)
    timings["prompt_ms"] = _elapsed_ms(t_prompt)

    # Step 7: LLM call. Capture the timing regardless of success/failure.
    effective_timeout = timeout if timeout is not None else llm_module.DEFAULT_TIMEOUT_SECONDS
    llm_result: LLMCallResult | None = None
    raw_llm_output: dict | list | str | None = None
    parsed_llm_decision: LLMDecision | None = None

    t_llm = time.perf_counter()
    try:
        llm_result = call_llm(
            system_prompt,
            user_prompt,
            client=client,
            timeout=effective_timeout,
            simulate_failure=simulate_failure,
        )
    except LLMTimeoutError:
        errors.append("LLM timeout")
    except LLMMalformedError as exc:
        errors.append("LLM malformed after retry")
        raw_llm_output = _serialize_raw_output(exc.raw_output)
    timings["llm_ms"] = _elapsed_ms(t_llm)

    if llm_result is not None:
        raw_llm_output = _serialize_raw_output(llm_result.raw_output)
        parsed_llm_decision = llm_result.parsed
        final_decision, decision_source, rationale, user_facing_message = _apply_policy_floor(
            llm_result=llm_result,
            policy_result=policy_result,
        )
    else:
        final_decision, decision_source, rationale, user_facing_message = _fallback(
            signals=signals,
            policy_result=policy_result,
            errors=errors,
        )

    timings["total_ms"] = _elapsed_ms(t0)

    return DecisionResult(
        final_decision=final_decision,
        decision_source=decision_source,
        rationale=rationale,
        user_facing_message=user_facing_message,
        signals=signals,
        conversation_state=conv_state,
        policy_result=policy_result,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        raw_llm_output=raw_llm_output,
        parsed_llm_decision=parsed_llm_decision,
        timings_ms=timings,
        errors=errors,
    )


# ---- pipeline branches -------------------------------------------------------


def _short_circuit_result(
    *,
    input: DecisionInput,
    signals: ComputedSignals,
    conv_state: ConversationState,
    policy_result: PolicyResult,
    timings: dict[str, float],
    errors: list[str],
) -> DecisionResult:
    missing = signals.missing_parameters
    description = input.proposed_action.description or input.proposed_action.action_type
    rationale = (
        "Missing required parameter(s) on an irreversible action: "
        f"{', '.join(missing)}. Short-circuited without LLM call."
    )
    user_facing_message = (
        f"I need {_humanize_list(missing)} before I can {description}."
    )
    return DecisionResult(
        final_decision=DecisionType.ASK_CLARIFYING_QUESTION,
        decision_source="short_circuit",
        rationale=rationale,
        user_facing_message=user_facing_message,
        signals=signals,
        conversation_state=conv_state,
        policy_result=policy_result,
        system_prompt=None,
        user_prompt=None,
        raw_llm_output=None,
        parsed_llm_decision=None,
        timings_ms=timings,
        errors=errors,
    )


def _apply_policy_floor(
    *,
    llm_result: LLMCallResult,
    policy_result: PolicyResult,
) -> tuple[DecisionType, str, str, str | None]:
    """Return (final_decision, decision_source, rationale, user_facing_message)."""
    parsed = llm_result.parsed
    assert parsed is not None, "call_llm returned no parsed decision on the success path"

    llm_decision = parsed.decision
    if (
        policy_result.violated
        and policy_result.forced_decision is not None
        and SEVERITY_ORDER[policy_result.forced_decision] > SEVERITY_ORDER[llm_decision]
    ):
        forced = policy_result.forced_decision
        first_reason = policy_result.reasons[0] if policy_result.reasons else "policy violation"
        rationale = (
            f"{parsed.rationale}\n\n"
            f"[Policy override: forced to {forced.value} because {first_reason}.]"
        )
        return forced, "policy_override", rationale, parsed.user_facing_message

    return llm_decision, "llm", parsed.rationale, parsed.user_facing_message


def _fallback(
    *,
    signals: ComputedSignals,
    policy_result: PolicyResult,
    errors: list[str],
) -> tuple[DecisionType, str, str, str | None]:
    """Degrade safely when the LLM didn't produce a usable answer."""
    error_summary = errors[-1] if errors else "unknown error"

    if policy_result.violated and policy_result.forced_decision is not None:
        forced = policy_result.forced_decision
        reasons_block = "; ".join(policy_result.reasons) if policy_result.reasons else ""
        rationale = "LLM unavailable; policy-forced decision applied."
        if reasons_block:
            rationale += f" Reasons: {reasons_block}"
        user_message = _FALLBACK_USER_MESSAGES.get(forced)
        return forced, "fallback", rationale, user_message

    decision = _fallback_decision_by_risk(signals)
    rationale = (
        f"LLM failure ({error_summary}). "
        f"Fallback to {decision.value} based on risk tier."
    )
    return decision, "fallback", rationale, _FALLBACK_USER_MESSAGES.get(decision)


def _fallback_decision_by_risk(signals: ComputedSignals) -> DecisionType:
    if (
        signals.reversibility == "irreversible"
        or signals.blast_radius in _HIGH_RISK_BLAST_RADII
    ):
        return DecisionType.REFUSE_OR_ESCALATE
    if signals.missing_parameters:
        return DecisionType.ASK_CLARIFYING_QUESTION
    return DecisionType.CONFIRM_BEFORE_EXECUTING


# ---- small utilities --------------------------------------------------------


def _elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


def _humanize_list(items: list[str]) -> str:
    if not items:
        return "more information"
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _serialize_raw_output(
    raw: dict | list | str | None,
) -> dict | list | str | None:
    """Convert SDK content blocks (non-pickleable objects) into plain dicts.

    On success, `raw_output` from the LLM is already the tool_input dict. On
    malformed paths we may be handed the raw list of content block objects,
    which Pydantic can't serialize as-is.
    """
    if not isinstance(raw, list):
        return raw
    return [
        {
            "type": getattr(block, "type", "unknown"),
            "name": getattr(block, "name", None),
            "input": getattr(block, "input", None),
        }
        for block in raw
    ]
