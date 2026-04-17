"""Tests for alfred.llm — mocked Anthropic client, no real network calls."""
from __future__ import annotations

from unittest.mock import MagicMock

import httpx
import pytest

from alfred.llm import (
    CLAUDE_MODEL,
    TOOL_SCHEMA,
    LLMCallResult,
    LLMMalformedError,
    LLMTimeoutError,
    call_llm,
)
from alfred.types import DecisionType


# ----- SDK-shaped fixtures ---------------------------------------------------


class _ToolUse:
    def __init__(self, input_dict: dict) -> None:
        self.type = "tool_use"
        self.name = "submit_decision"
        self.input = input_dict


class _Response:
    def __init__(self, content_blocks: list) -> None:
        self.content = content_blocks
        self.stop_reason = "tool_use"


def _valid_tool_use(decision: str = DecisionType.EXECUTE_SILENTLY.value) -> _ToolUse:
    return _ToolUse(
        {
            "decision": decision,
            "rationale": "Reversible self-only action with clear intent.",
            "user_facing_message": None,
        }
    )


def _invalid_tool_use() -> _ToolUse:
    return _ToolUse(
        {
            "decision": "YOLO_SEND",  # not a valid DecisionType enum value
            "rationale": "Bogus output used to exercise the retry path.",
        }
    )


def _make_mock_client(responses: list) -> MagicMock:
    """Build a mock Anthropic client whose messages.create() returns each response in turn."""
    client = MagicMock()
    client.messages.create.side_effect = responses
    return client


# ----- tests ------------------------------------------------------------------


def test_happy_path_returns_parsed_decision() -> None:
    """Valid tool_use on first attempt -> parsed decision, attempts=1, no errors."""
    client = _make_mock_client([_Response([_valid_tool_use()])])

    result = call_llm("sys", "usr", client=client)

    assert isinstance(result, LLMCallResult)
    assert result.parsed is not None
    assert result.parsed.decision == DecisionType.EXECUTE_SILENTLY
    assert result.attempts == 1
    assert result.errors == []
    assert client.messages.create.call_count == 1

    # Spot-check the call arguments match the documented invocation contract.
    kwargs = client.messages.create.call_args.kwargs
    assert kwargs["model"] == CLAUDE_MODEL
    assert kwargs["tools"] == [TOOL_SCHEMA]
    assert kwargs["tool_choice"] == {"type": "tool", "name": "submit_decision"}


def test_malformed_then_valid_on_retry() -> None:
    """Invalid on attempt 1, valid on attempt 2 -> parsed set, attempts=2, one error recorded."""
    client = _make_mock_client(
        [
            _Response([_invalid_tool_use()]),
            _Response([_valid_tool_use(DecisionType.CONFIRM_BEFORE_EXECUTING.value)]),
        ]
    )

    result = call_llm("sys", "usr", client=client)

    assert result.parsed is not None
    assert result.parsed.decision == DecisionType.CONFIRM_BEFORE_EXECUTING
    assert result.attempts == 2
    assert len(result.errors) == 1
    assert "schema validation failed" in result.errors[0]
    assert client.messages.create.call_count == 2

    # The retry must carry the CORRECTION note in its system prompt.
    second_call_kwargs = client.messages.create.call_args_list[1].kwargs
    assert "CORRECTION" in second_call_kwargs["system"]


def test_malformed_twice_raises() -> None:
    """Both attempts malformed -> LLMMalformedError, raw_output from last attempt on exception."""
    client = _make_mock_client(
        [
            _Response([_invalid_tool_use()]),
            _Response([_invalid_tool_use()]),
        ]
    )

    with pytest.raises(LLMMalformedError) as excinfo:
        call_llm("sys", "usr", client=client)

    assert excinfo.value.raw_output is not None
    # raw_output is the last attempt's tool_use input dict.
    assert isinstance(excinfo.value.raw_output, dict)
    assert excinfo.value.raw_output.get("decision") == "YOLO_SEND"
    assert client.messages.create.call_count == 2


def test_simulate_timeout_raises_LLMTimeoutError() -> None:
    """simulate_failure='timeout' short-circuits before the network; client can be None."""
    with pytest.raises(LLMTimeoutError):
        call_llm("sys", "usr", client=None, simulate_failure="timeout")


def test_simulate_malformed_raises_after_two_canned_failures() -> None:
    """simulate_failure='malformed' must demo the full retry-then-fallback flow
    without hitting the network (so the UI can show it with no API key set).
    Both attempts use canned bad output; LLMMalformedError is raised.
    """
    with pytest.raises(LLMMalformedError) as excinfo:
        call_llm("sys", "usr", client=None, simulate_failure="malformed")

    assert excinfo.value.raw_output is not None
    assert isinstance(excinfo.value.raw_output, dict)
    assert excinfo.value.raw_output.get("decision") == "YOLO_SEND"


def test_timing_ms_is_populated() -> None:
    """Any successful call records positive wall-clock timing in ms."""
    client = _make_mock_client([_Response([_valid_tool_use()])])

    result = call_llm("sys", "usr", client=client)

    assert result.timing_ms > 0


# ----- extra coverage for real-network timeout wrapping -----------------------


def test_httpx_timeout_is_wrapped_as_LLMTimeoutError() -> None:
    """If the underlying SDK surfaces a timeout, callers see LLMTimeoutError."""
    client = MagicMock()
    client.messages.create.side_effect = httpx.TimeoutException("boom")

    with pytest.raises(LLMTimeoutError):
        call_llm("sys", "usr", client=client)
