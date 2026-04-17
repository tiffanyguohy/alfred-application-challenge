"""Anthropic client wrapper: tool-use call, retry on malformed, failure simulation.

All network activity is funneled through a single call site so tests can inject
a mock. `call_llm` is the only public function; it returns `LLMCallResult`
(parsed + raw + diagnostics) or raises `LLMTimeoutError` / `LLMMalformedError`.
"""
from __future__ import annotations

import time
from typing import Any

import anthropic
import httpx
from pydantic import BaseModel, ValidationError

from alfred.types import DecisionType, LLMDecision


CLAUDE_MODEL: str = "claude-sonnet-4-5"
DEFAULT_TIMEOUT_SECONDS: float = 15.0
MAX_TOKENS: int = 1024
MAX_ATTEMPTS: int = 2


TOOL_SCHEMA: dict = {
    "name": "submit_decision",
    "description": "Submit the final decision for the proposed action.",
    "input_schema": {
        "type": "object",
        "properties": {
            "decision": {
                "type": "string",
                "enum": [d.value for d in DecisionType],
            },
            "rationale": {
                "type": "string",
                "description": "2-4 sentences on why this decision, grounded in signals and conversation state.",
            },
            "user_facing_message": {
                "type": ["string", "null"],
                "description": "The message to show the user for confirm/ask/refuse decisions. Null for silent execution.",
            },
        },
        "required": ["decision", "rationale"],
    },
}


CORRECTION_NOTE = (
    "\n\nCORRECTION: Your previous output failed schema validation. Return the "
    "tool call with one of the exact enum strings for 'decision' and non-empty "
    "'rationale'."
)


class LLMCallResult(BaseModel):
    parsed: LLMDecision | None
    raw_output: dict | list | str | None
    attempts: int
    errors: list[str]
    timing_ms: float


class LLMTimeoutError(Exception):
    """Raised when the LLM call exceeds the configured timeout."""


class LLMMalformedError(Exception):
    """Raised after retry fails to produce schema-valid output.

    Carries `raw_output` from the last attempt so the UI can surface it.
    """

    def __init__(self, message: str, raw_output: dict | list | str | None = None) -> None:
        super().__init__(message)
        self.raw_output = raw_output


def call_llm(
    system_prompt: str,
    user_prompt: str,
    *,
    client: Any = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    simulate_failure: str | None = None,
) -> LLMCallResult:
    """Call the LLM with tool-use structured output, retrying once on malformed.

    On success returns LLMCallResult with `parsed` populated. On timeout raises
    LLMTimeoutError. On two consecutive malformed responses raises
    LLMMalformedError (carrying the last raw output).
    """
    start = time.perf_counter()

    if simulate_failure == "timeout":
        # Bounded stall so the UI shows a nonzero latency, but never the full 15s.
        time.sleep(0.1)
        raise LLMTimeoutError("Simulated timeout")

    # Only construct a real client when we actually need one. Malformed simulation
    # short-circuits both attempts locally (see below), so it must work with no key.
    if client is None and simulate_failure != "malformed":
        client = anthropic.Anthropic()

    errors: list[str] = []
    raw_output: dict | list | str | None = None
    attempts = 0

    for attempt_index in range(MAX_ATTEMPTS):
        attempts = attempt_index + 1
        use_correction = attempt_index > 0
        current_system = system_prompt + CORRECTION_NOTE if use_correction else system_prompt

        # Malformed sim: both attempts return canned bad output, so the retry
        # path fires and then LLMMalformedError is raised for the fallback demo.
        if simulate_failure == "malformed":
            response_content = _synthesize_malformed_response()
        else:
            try:
                response = _invoke_client(
                    client=client,
                    system_prompt=current_system,
                    user_prompt=user_prompt,
                    timeout=timeout,
                )
            except (anthropic.APITimeoutError, httpx.TimeoutException) as exc:
                raise LLMTimeoutError(f"LLM call timed out after {timeout}s: {exc}") from exc
            response_content = _content_to_blocks(response)

        tool_input = _extract_tool_input(response_content)
        raw_output = tool_input if tool_input is not None else response_content

        if tool_input is None:
            errors.append(f"attempt {attempts}: no submit_decision tool_use block found")
            continue

        try:
            parsed = LLMDecision.model_validate(tool_input)
        except ValidationError as exc:
            errors.append(f"attempt {attempts}: schema validation failed: {exc.errors()}")
            continue

        timing_ms = (time.perf_counter() - start) * 1000.0
        return LLMCallResult(
            parsed=parsed,
            raw_output=raw_output,
            attempts=attempts,
            errors=errors,
            timing_ms=timing_ms,
        )

    # Both attempts failed. Raise with the last raw output attached for the UI.
    raise LLMMalformedError(
        f"LLM output failed schema validation after {attempts} attempts: {errors}",
        raw_output=raw_output,
    )


# ---- helpers -----------------------------------------------------------------


def _invoke_client(
    *,
    client: Any,
    system_prompt: str,
    user_prompt: str,
    timeout: float,
) -> Any:
    """Single point of network contact. Isolated so tests can mock cleanly."""
    return client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=MAX_TOKENS,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
        tools=[TOOL_SCHEMA],
        tool_choice={"type": "tool", "name": "submit_decision"},
        timeout=timeout,
    )


def _content_to_blocks(response: Any) -> list:
    """Return the content blocks list from an SDK response (or any mock that exposes .content)."""
    content = getattr(response, "content", None)
    if content is None:
        return []
    return list(content)


def _extract_tool_input(content_blocks: Any) -> dict | None:
    """Find the first submit_decision tool_use block and return its `.input` dict."""
    if not isinstance(content_blocks, list):
        return None
    for block in content_blocks:
        block_type = getattr(block, "type", None)
        block_name = getattr(block, "name", None)
        if block_type == "tool_use" and block_name == "submit_decision":
            raw_input = getattr(block, "input", None)
            if isinstance(raw_input, dict):
                return raw_input
    return None


def _synthesize_malformed_response() -> list:
    """Canned bad tool_use block used when simulate_failure='malformed'.

    Shaped like the real SDK's content blocks: an object with `.type`, `.name`,
    and `.input`. The enum value is deliberately invalid so LLMDecision
    validation fails and the retry path fires.
    """

    class _SimulatedToolUseBlock:
        def __init__(self) -> None:
            self.type = "tool_use"
            self.name = "submit_decision"
            self.input = {
                "decision": "YOLO_SEND",
                "rationale": "Simulated malformed output for failure mode testing.",
            }

    return [_SimulatedToolUseBlock()]
