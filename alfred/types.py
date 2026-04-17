"""Pydantic v2 models for the alfred_ Execution Decision Layer."""
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class DecisionType(str, Enum):
    EXECUTE_SILENTLY = "execute_silently"
    EXECUTE_AND_NOTIFY = "execute_and_notify"
    CONFIRM_BEFORE_EXECUTING = "confirm_before_executing"
    ASK_CLARIFYING_QUESTION = "ask_clarifying_question"
    REFUSE_OR_ESCALATE = "refuse_or_escalate"


ActionType = Literal[
    "reminder_create",
    "calendar_read",
    "calendar_event_create",
    "calendar_event_update",
    "email_send",
    "email_draft",
    "wire_transfer",
    "payment",
]

Role = Literal["user", "alfred"]


class Contact(BaseModel):
    name: str
    email: str


class Message(BaseModel):
    role: Role
    content: str
    timestamp: str


class ProposedAction(BaseModel):
    action_type: ActionType
    parameters: dict[str, Any]
    description: str


class UserState(BaseModel):
    silent_send_enabled: bool = False
    financial_confirmation_threshold_usd: float = 1000.0
    preferred_calendar_autonomy: Literal["low", "medium", "high"] = "medium"
    external_email_default: Literal["confirm", "notify_after"] = "confirm"
    contacts: list[Contact] = Field(default_factory=list)


class DecisionInput(BaseModel):
    proposed_action: ProposedAction
    history: list[Message]
    user_state: UserState


class ConversationState(BaseModel):
    pending_constraints: list[str] = Field(default_factory=list)
    unresolved_references: list[str] = Field(default_factory=list)
    last_drafted_artifact: str | None = None
    awaiting_confirmation: str | None = None


class ComputedSignals(BaseModel):
    reversibility: Literal["irreversible", "partially_reversible", "reversible"]
    blast_radius: Literal["self", "internal", "external", "financial", "legal"]
    missing_parameters: list[str] = Field(default_factory=list)
    entity_ambiguities: dict[str, list[str]] = Field(default_factory=dict)
    risk_contributing_factors: list[str] = Field(default_factory=list)
    intent_clarity_score: float = Field(ge=0.0, le=1.0)


class PolicyResult(BaseModel):
    violated: bool
    forced_decision: DecisionType | None = None
    reasons: list[str] = Field(default_factory=list)


class LLMDecision(BaseModel):
    decision: DecisionType
    rationale: str
    user_facing_message: str | None = None


class DecisionResult(BaseModel):
    final_decision: DecisionType
    decision_source: Literal["llm", "policy_override", "short_circuit", "fallback"]
    rationale: str
    user_facing_message: str | None = None

    signals: ComputedSignals
    conversation_state: ConversationState
    policy_result: PolicyResult

    system_prompt: str | None = None
    user_prompt: str | None = None
    raw_llm_output: dict | list | str | None = None
    parsed_llm_decision: LLMDecision | None = None

    timings_ms: dict[str, float] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)
