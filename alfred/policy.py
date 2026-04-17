"""Hard policy tripwires. Deterministic; can force-override the LLM decision."""
from typing import Any

from alfred.types import (
    ComputedSignals,
    ConversationState,
    DecisionInput,
    DecisionType,
    PolicyResult,
)


FINANCIAL_ACTION_TYPES = {"wire_transfer", "payment"}
SEND_ACTION_TYPES = {"email_send", "wire_transfer", "payment"}
BULK_RECIPIENT_TOKENS = {"all", "everyone", "@all"}
BULK_RECIPIENT_THRESHOLD = 5


def check_policy(
    input: DecisionInput,
    signals: ComputedSignals,
    conv_state: ConversationState,
) -> PolicyResult:
    """Apply hard rules in order; first violating rule sets forced_decision."""
    action = input.proposed_action
    user_state = input.user_state
    reasons: list[str] = []
    forced_decision: DecisionType | None = None

    # Rule 1 & 2: financial amount thresholds.
    amount = _get_amount(action.parameters)
    threshold = user_state.financial_confirmation_threshold_usd
    if action.action_type in FINANCIAL_ACTION_TYPES and amount is not None and threshold > 0:
        if amount > 10 * threshold:
            reasons.append(
                f"Amount ${_format_usd(amount)} exceeds 10x user threshold "
                f"(${_format_usd(10 * threshold)})"
            )
            forced_decision = DecisionType.REFUSE_OR_ESCALATE
        elif amount > threshold:
            reasons.append(
                f"Amount ${_format_usd(amount)} exceeds user confirmation threshold "
                f"(${_format_usd(threshold)})"
            )
            forced_decision = DecisionType.CONFIRM_BEFORE_EXECUTING

    # Rule 3: bulk external email.
    if action.action_type == "email_send" and _is_bulk_recipients(action.parameters):
        reasons.append("Bulk external action")
        if forced_decision is None:
            forced_decision = DecisionType.CONFIRM_BEFORE_EXECUTING

    # Rule 4: missing required params on an irreversible action.
    if signals.missing_parameters and signals.reversibility == "irreversible":
        reasons.append(
            "Missing required parameter(s) on irreversible action: "
            + ", ".join(signals.missing_parameters)
        )
        if forced_decision is None:
            forced_decision = DecisionType.ASK_CLARIFYING_QUESTION

    violated = forced_decision is not None

    # Rule 5: silent_send disabled on a send action — soft signal, not a violation.
    if action.action_type in SEND_ACTION_TYPES and not user_state.silent_send_enabled:
        reasons.append("User has silent_send_enabled=False; silent path blocked")

    return PolicyResult(violated=violated, forced_decision=forced_decision, reasons=reasons)


def _get_amount(parameters: dict[str, Any]) -> float | None:
    value = parameters.get("amount")
    return float(value) if isinstance(value, (int, float)) else None


def _is_bulk_recipients(parameters: dict[str, Any]) -> bool:
    to_value = parameters.get("to")
    if isinstance(to_value, str) and to_value.strip().lower() in BULK_RECIPIENT_TOKENS:
        return True
    return isinstance(to_value, list) and len(to_value) > BULK_RECIPIENT_THRESHOLD


def _format_usd(amount: float) -> str:
    if amount == int(amount):
        return f"{int(amount):,}"
    return f"{amount:,.2f}"
