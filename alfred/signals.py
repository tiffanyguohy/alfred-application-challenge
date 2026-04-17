"""Deterministic computation of action signals for the decision layer."""
from typing import Any

from alfred.conversation_state import extract_unresolved_references
from alfred.types import (
    ComputedSignals,
    DecisionInput,
    Message,
    ProposedAction,
    UserState,
)


REVERSIBILITY_BY_ACTION_TYPE: dict[str, str] = {
    "reminder_create": "reversible",
    "calendar_read": "reversible",
    "calendar_event_create": "reversible",
    "calendar_event_update": "reversible",
    "email_draft": "reversible",
    "email_send": "irreversible",
    "wire_transfer": "irreversible",
    "payment": "irreversible",
}


BLAST_RADIUS_BY_ACTION_TYPE: dict[str, str] = {
    "reminder_create": "self",
    "calendar_read": "self",
    "calendar_event_create": "internal",
    "calendar_event_update": "internal",
    "email_draft": "self",
    "email_send": "external",
    "wire_transfer": "financial",
    "payment": "financial",
}


REQUIRED_PARAMS_BY_ACTION_TYPE: dict[str, list[str]] = {
    "reminder_create": ["text", "time"],
    "calendar_read": ["date_range"],
    "calendar_event_create": ["title", "start_time", "end_time"],
    "calendar_event_update": ["event_id"],
    "email_draft": ["to", "subject", "body"],
    "email_send": ["to", "subject", "body"],
    "wire_transfer": ["recipient", "amount", "currency"],
    "payment": ["recipient", "amount", "currency"],
}


EMOTIONAL_MARKERS = (
    "had it",
    "fed up",
    "i'm done",
    "i am done",
    "quit",
    "furious",
    "hate",
    "goddamn",
    "f***",
)

NOTEWORTHY_FINANCIAL_USD = 1000

FIELDS_TO_SCAN_FOR_NAMES = ("to", "recipient", "title", "body", "subject")


def compute_signals(input: DecisionInput) -> ComputedSignals:
    """Compute deterministic signals (no LLM, no I/O) for an incoming action."""
    action = input.proposed_action
    reversibility = REVERSIBILITY_BY_ACTION_TYPE.get(action.action_type, "irreversible")
    blast_radius = BLAST_RADIUS_BY_ACTION_TYPE.get(action.action_type, "external")
    missing_parameters = find_missing_parameters(action)
    entity_ambiguities = find_entity_ambiguities(action, input.user_state)

    risk_contributing_factors = compute_risk_contributing_factors(
        action=action,
        reversibility=reversibility,
        blast_radius=blast_radius,
        missing_parameters=missing_parameters,
        entity_ambiguities=entity_ambiguities,
        history=input.history,
        user_state=input.user_state,
    )

    intent_clarity_score = compute_intent_clarity_score(
        missing_parameters=missing_parameters,
        entity_ambiguities=entity_ambiguities,
        history=input.history,
    )

    return ComputedSignals(
        reversibility=reversibility,
        blast_radius=blast_radius,
        missing_parameters=missing_parameters,
        entity_ambiguities=entity_ambiguities,
        risk_contributing_factors=risk_contributing_factors,
        intent_clarity_score=intent_clarity_score,
    )


def find_missing_parameters(action: ProposedAction) -> list[str]:
    required = REQUIRED_PARAMS_BY_ACTION_TYPE.get(action.action_type, [])
    return [p for p in required if not _has_value(action.parameters, p)]


def _has_value(params: dict[str, Any], key: str) -> bool:
    value = params.get(key)
    if value is None:
        return False
    if isinstance(value, str) and not value.strip():
        return False
    return True


def find_entity_ambiguities(
    action: ProposedAction, user_state: UserState
) -> dict[str, list[str]]:
    first_name_index: dict[str, list[str]] = {}
    for contact in user_state.contacts:
        first = contact.name.split()[0] if contact.name else ""
        if first:
            first_name_index.setdefault(first, []).append(contact.name)

    ambiguous_first_names = {
        name: matches for name, matches in first_name_index.items() if len(matches) >= 2
    }
    if not ambiguous_first_names:
        return {}

    haystack = _collect_scannable_text(action).lower()
    if not haystack:
        return {}

    found: dict[str, list[str]] = {}
    for first_name, matches in ambiguous_first_names.items():
        if first_name.lower() not in haystack:
            continue
        # Skip if the user already disambiguated with a full email belonging to one contact.
        if _already_disambiguated(first_name, action, user_state):
            continue
        found[first_name] = matches
    return found


def _collect_scannable_text(action: ProposedAction) -> str:
    pieces: list[str] = [action.description]
    for field in FIELDS_TO_SCAN_FOR_NAMES:
        val = action.parameters.get(field)
        if isinstance(val, str):
            pieces.append(val)
    return " ".join(pieces)


def _already_disambiguated(
    first_name: str, action: ProposedAction, user_state: UserState
) -> bool:
    addresses = [
        v.lower()
        for v in (action.parameters.get("to"), action.parameters.get("recipient"))
        if isinstance(v, str)
    ]
    if not addresses:
        return False
    matching_contacts = [
        c for c in user_state.contacts if c.name.split()[:1] == [first_name]
    ]
    for contact in matching_contacts:
        if not contact.email:
            continue
        email = contact.email.lower()
        if any(email in addr for addr in addresses):
            return True
    return False


def compute_risk_contributing_factors(
    *,
    action: ProposedAction,
    reversibility: str,
    blast_radius: str,
    missing_parameters: list[str],
    entity_ambiguities: dict[str, list[str]],
    history: list[Message],
    user_state: UserState,
) -> list[str]:
    factors: list[str] = []

    if reversibility == "irreversible":
        factors.append("irreversible action")

    if blast_radius == "external":
        factors.append("external recipient")

    amount = action.parameters.get("amount")
    if isinstance(amount, (int, float)):
        threshold = user_state.financial_confirmation_threshold_usd
        if amount > NOTEWORTHY_FINANCIAL_USD:
            factors.append(f"financial amount over ${NOTEWORTHY_FINANCIAL_USD:,}")
        if threshold > 0 and amount > 10 * threshold:
            factors.append("financial amount over 10x user threshold")

    if _has_emotional_charge(history):
        factors.append("emotionally charged recent messages")

    if missing_parameters:
        factors.append("missing required parameters")

    if entity_ambiguities:
        factors.append("entity ambiguity")

    return factors


def _has_emotional_charge(history: list[Message]) -> bool:
    recent_user_texts = [m.content.lower() for m in history if m.role == "user"][-3:]
    return any(
        marker in text for text in recent_user_texts for marker in EMOTIONAL_MARKERS
    )


def compute_intent_clarity_score(
    *,
    missing_parameters: list[str],
    entity_ambiguities: dict[str, list[str]],
    history: list[Message],
) -> float:
    score = 1.0
    score -= 0.3 * min(len(missing_parameters), 3)
    if entity_ambiguities:
        score -= 0.4
    score -= 0.2 * len(extract_unresolved_references(history))
    return max(0.0, round(score, 3))
