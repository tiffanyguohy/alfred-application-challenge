"""Derive a compact ConversationState from raw chat history."""
import re

from alfred.types import ConversationState, Message


# Whether a constraint has been *resolved* is a context judgment we leave to the LLM
# (per the code/LLM split in CLAUDE.md). The deterministic layer only surfaces the
# hold; generic affirmations like "yep" must not be silently auto-resolved in code.
HOLD_PATTERNS = (
    "hold off",
    "hold the",
    "wait until",
    "wait for",
    "don't send",
    "do not send",
    "don't do",
    "do not do",
    "pause on",
    "pause that",
    "pause the",
    "not yet",
    "on hold",
)


# Strong markers indicate the alfred message IS the draft (vs. merely mentioning
# a previous draft — e.g. "I'll hold the draft until legal signs off"). Prefer
# strong markers first so we latch onto the actual drafted artifact.
STRONG_DRAFT_MARKERS = ("here's the draft", "here is the draft", "i've prepared", "i have prepared")
WEAK_DRAFT_MARKERS = ("draft",)


AWAITING_MARKERS = ("shall i", "want me to", "should i", "confirm")


PRONOUN_TOKENS = ("it", "that", "this")
DEICTIC_PHRASES = ("the email", "the draft", "the message", "the transfer")

UNRESOLVED_REF_MIN_WORDS = 8


def derive_conversation_state(history: list[Message]) -> ConversationState:
    """Walk history to extract holds, unresolved references, last draft, awaiting confirm."""
    return ConversationState(
        pending_constraints=_extract_pending_constraints(history),
        unresolved_references=extract_unresolved_references(history),
        last_drafted_artifact=_extract_last_drafted_artifact(history),
        awaiting_confirmation=_extract_awaiting_confirmation(history),
    )


def _extract_pending_constraints(history: list[Message]) -> list[str]:
    constraints: list[str] = []
    for msg in history:
        if msg.role != "user":
            continue
        lower = msg.content.lower()
        for pattern in HOLD_PATTERNS:
            if pattern in lower:
                constraints.append(_extract_sentence(msg.content, pattern))
                break
    return constraints


def _extract_sentence(text: str, pattern: str) -> str:
    """Return the sentence containing `pattern` (case-insensitive), or the full text."""
    if pattern not in text.lower():
        return text.strip()
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        if pattern in sentence.lower():
            return sentence.strip()
    return text.strip()


def extract_unresolved_references(history: list[Message]) -> list[str]:
    """Scan the latest user message only; longer messages typically self-resolve."""
    latest_user = next((m for m in reversed(history) if m.role == "user"), None)
    if latest_user is None:
        return []
    text = latest_user.content.strip()
    if len(text.split()) >= UNRESOLVED_REF_MIN_WORDS:
        return []

    lower = text.lower()
    tokens = set(re.findall(r"[a-zA-Z']+", lower))
    found: list[str] = [p for p in PRONOUN_TOKENS if p in tokens]
    found.extend(phrase for phrase in DEICTIC_PHRASES if phrase in lower)
    return found


def _extract_last_drafted_artifact(history: list[Message]) -> str | None:
    strong_match: Message | None = None
    weak_match: Message | None = None
    for msg in history:
        if msg.role != "alfred":
            continue
        lower = msg.content.lower()
        if any(marker in lower for marker in STRONG_DRAFT_MARKERS):
            strong_match = msg
        elif any(marker in lower for marker in WEAK_DRAFT_MARKERS):
            weak_match = msg

    latest_draft = strong_match or weak_match
    if latest_draft is None:
        return None

    topic = _infer_topic(latest_draft.content)
    time_part = _short_timestamp(latest_draft.timestamp)
    descriptor = f"{topic} draft from {time_part}" if topic else f"draft from {time_part}"
    return descriptor[:100]


def _infer_topic(draft_content: str) -> str:
    subject = re.search(r"Subject:\s*([^\n]+)", draft_content, re.IGNORECASE)
    if subject:
        return subject.group(1).strip()
    first_line = draft_content.strip().splitlines()[0] if draft_content.strip() else ""
    return first_line[:60].strip()


def _short_timestamp(timestamp: str) -> str:
    if "T" in timestamp:
        return timestamp.split("T", 1)[1][:5]
    return timestamp


def _extract_awaiting_confirmation(history: list[Message]) -> str | None:
    last_ask: Message | None = None
    last_ask_index = -1
    for i, msg in enumerate(history):
        if msg.role == "alfred" and _looks_like_confirmation_ask(msg.content):
            last_ask = msg
            last_ask_index = i

    if last_ask is None:
        return None

    user_replied_after = any(m.role == "user" for m in history[last_ask_index + 1 :])
    if user_replied_after:
        return None

    return _extract_question_text(last_ask.content)[:100]


def _looks_like_confirmation_ask(content: str) -> bool:
    stripped = content.strip()
    if stripped.endswith("?"):
        return True
    lower = stripped.lower()
    return any(marker in lower for marker in AWAITING_MARKERS)


def _extract_question_text(content: str) -> str:
    for sentence in reversed(re.split(r"(?<=[.!?])\s+", content.strip())):
        s = sentence.strip()
        if not s:
            continue
        if s.endswith("?") or any(marker in s.lower() for marker in AWAITING_MARKERS):
            return s
    return content.strip()
