"""Streamlit UI for the alfred_ Execution Decision Layer.

Layout:
  Left column  — scenario input (picker, action, history, user state, failure sim, run).
  Right column — decision outcome and "under the hood" expanders.

All logic lives in alfred/*. This module is a thin presentational wrapper.
"""
from __future__ import annotations

import json
import os
from typing import Any, Callable

import streamlit as st
from dotenv import load_dotenv

from alfred.decide import decide
from alfred.scenarios import SCENARIOS
from alfred.types import DecisionResult, DecisionType

load_dotenv()

st.set_page_config(
    page_title="alfred_ Decision Layer",
    layout="wide",
    page_icon="🤝",
)

_FAILURE_SIM_MAP: dict[str, str | None] = {
    "None": None,
    "Timeout": "timeout",
    "Malformed output": "malformed",
}

_DECISION_RENDERER: dict[DecisionType, Callable[[str], Any]] = {
    DecisionType.EXECUTE_SILENTLY: st.success,
    DecisionType.EXECUTE_AND_NOTIFY: st.info,
    DecisionType.CONFIRM_BEFORE_EXECUTING: st.warning,
    DecisionType.ASK_CLARIFYING_QUESTION: st.warning,
    DecisionType.REFUSE_OR_ESCALATE: st.error,
}

_ROLE_COLOR = {"user": "#1f77b4", "alfred": "#6c757d"}


def _has_api_key() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def _render_expander(label: str, builder: Callable[[], None], *, expanded: bool = False) -> None:
    with st.expander(label, expanded=expanded):
        builder()


def _init_session_state() -> None:
    st.session_state.setdefault("scenario_idx", 0)
    st.session_state.setdefault("last_result", None)
    st.session_state.setdefault("failure_sim", "None")


def _render_sidebar() -> None:
    st.sidebar.markdown(
        "### The five decisions\n"
        "`execute_silently` · `execute_and_notify` · `confirm_before_executing` · "
        "`ask_clarifying_question` · `refuse_or_escalate`.\n\n"
        "Policy can force a floor; LLM provides contextual judgment."
    )
    st.sidebar.markdown("[README]() | [GitHub]()")
    if _has_api_key():
        st.sidebar.success("API key detected")
    else:
        st.sidebar.error("No API key — only failure-sim paths work")


def _render_left_column() -> tuple[int, str, bool]:
    st.header("Scenario")

    scenario_idx = st.selectbox(
        "Pick a scenario",
        options=list(range(len(SCENARIOS))),
        format_func=lambda i: f"{i + 1}. {SCENARIOS[i].name}",
        key="scenario_idx",
    )
    scenario = SCENARIOS[scenario_idx]

    meta_cols = st.columns(3)
    meta_cols[0].markdown(f"**Category:** `{scenario.category}`")
    meta_cols[1].markdown(f"**Must pass:** {'✓' if scenario.must_pass else '—'}")
    meta_cols[2].caption(f"Expected: `{scenario.expected_decision.value}`")

    action = scenario.input.proposed_action
    with st.expander("Proposed action", expanded=True):
        st.markdown(f"**{action.action_type}**")
        st.write(action.description)
        st.code(json.dumps(action.parameters, indent=2), language="json")

    with st.expander("Conversation history", expanded=True):
        if not scenario.input.history:
            st.caption("(no prior messages)")
        for msg in scenario.input.history:
            color = _ROLE_COLOR.get(msg.role, "#333")
            st.markdown(
                f"<span style='color:{color}; font-weight:600'>[{msg.role} @ {msg.timestamp}]</span>",
                unsafe_allow_html=True,
            )
            st.text(msg.content)

    with st.expander("User state", expanded=False):
        us = scenario.input.user_state
        st.markdown(f"- `silent_send_enabled`: **{us.silent_send_enabled}**")
        st.markdown(
            f"- `financial_confirmation_threshold_usd`: **{us.financial_confirmation_threshold_usd}**"
        )
        st.markdown(f"- `preferred_calendar_autonomy`: **{us.preferred_calendar_autonomy}**")
        st.markdown(f"- `external_email_default`: **{us.external_email_default}**")
        st.markdown(f"- `contact_count`: **{len(us.contacts)}**")
        if us.contacts:
            st.markdown("**Contacts:**")
            for c in us.contacts:
                st.markdown(f"  - {c.name} — `{c.email}`")

    st.markdown(f"*Notes: {scenario.notes}*")

    failure_sim = st.radio(
        "Simulate LLM failure",
        options=list(_FAILURE_SIM_MAP.keys()),
        horizontal=True,
        key="failure_sim",
    )

    needs_network = _FAILURE_SIM_MAP[failure_sim] is None
    disabled = needs_network and not _has_api_key()
    run_clicked = st.button(
        "Run decision pipeline",
        type="primary",
        disabled=disabled,
        help="Set ANTHROPIC_API_KEY or use a failure simulation." if disabled else None,
    )

    return scenario_idx, failure_sim, run_clicked


def _render_decision_badge(result: DecisionResult) -> None:
    renderer = _DECISION_RENDERER.get(result.final_decision, st.info)
    label = f"**{result.final_decision.value}**  ·  source: `{result.decision_source}`"
    renderer(label)


def _render_match_badge(result: DecisionResult, expected: DecisionType) -> None:
    if result.final_decision == expected:
        st.markdown(":green[✓ matches expected decision]")
    else:
        st.markdown(
            f":red[✗ mismatch] — expected `{expected.value}`, "
            f"got `{result.final_decision.value}`"
        )


def _render_under_the_hood(result: DecisionResult) -> None:
    def conversation_state() -> None:
        cs = result.conversation_state
        st.markdown(f"**pending_constraints:** {cs.pending_constraints or '_(none)_'}")
        st.markdown(f"**unresolved_references:** {cs.unresolved_references or '_(none)_'}")
        st.markdown(f"**last_drafted_artifact:** {cs.last_drafted_artifact or '_(none)_'}")
        st.markdown(f"**awaiting_confirmation:** {cs.awaiting_confirmation or '_(none)_'}")

    def signals() -> None:
        s = result.signals
        st.markdown(f"- **reversibility:** `{s.reversibility}`")
        st.markdown(f"- **blast_radius:** `{s.blast_radius}`")
        st.markdown(f"- **intent_clarity_score:** `{s.intent_clarity_score}`")
        st.markdown(f"- **missing_parameters:** {s.missing_parameters or '_(none)_'}")
        st.markdown(f"- **entity_ambiguities:** {s.entity_ambiguities or '_(none)_'}")
        st.markdown(
            f"- **risk_contributing_factors:** {s.risk_contributing_factors or '_(none)_'}"
        )

    def policy() -> None:
        p = result.policy_result
        st.markdown(f"- **violated:** {'yes' if p.violated else 'no'}")
        forced = p.forced_decision.value if p.forced_decision else "_(none)_"
        st.markdown(f"- **forced_decision:** `{forced}`")
        if p.reasons:
            st.markdown("**reasons:**")
            for r in p.reasons:
                st.markdown(f"  - {r}")
        else:
            st.markdown("**reasons:** _(none)_")

    def system_prompt() -> None:
        st.code(result.system_prompt or "(prompt not built — short-circuited)", language="markdown")

    def user_prompt() -> None:
        st.code(result.user_prompt or "(prompt not built — short-circuited)", language="markdown")

    def raw_llm() -> None:
        if result.raw_llm_output is None:
            st.caption("(none — LLM was not called or returned nothing)")
            return
        st.code(
            json.dumps(result.raw_llm_output, indent=2, default=str),
            language="json",
        )

    def parsed() -> None:
        if result.parsed_llm_decision is None:
            st.caption("(none)")
            return
        st.code(
            json.dumps(result.parsed_llm_decision.model_dump(), indent=2, default=str),
            language="json",
        )

    def decision_source() -> None:
        st.markdown(f"Source: `{result.decision_source}`")

    def timings() -> None:
        rows = [{"stage": k, "ms": round(v, 2)} for k, v in result.timings_ms.items()]
        st.table(rows)

    def errors() -> None:
        if not result.errors:
            st.caption("(no errors)")
            return
        for err in result.errors:
            st.warning(err)

    _render_expander("🧭 Conversation State (key context layer)", conversation_state, expanded=True)
    _render_expander("📊 Computed Signals", signals)
    _render_expander("🛡️ Policy Check", policy)
    _render_expander("📝 System Prompt", system_prompt)
    _render_expander("📝 User Prompt", user_prompt)
    _render_expander("🤖 Raw LLM Output", raw_llm)
    _render_expander("🧾 Parsed LLM Decision", parsed)
    _render_expander("⚙️ Decision Source", decision_source)
    _render_expander("⏱️ Timings", timings)
    _render_expander("⚠️ Errors", errors)


def _render_right_column(scenario_idx: int) -> None:
    result: DecisionResult | None = st.session_state.get("last_result")
    if result is None:
        st.markdown(":grey[Click Run to evaluate this scenario.]")
        return

    _render_decision_badge(result)
    st.markdown("### Rationale")
    st.markdown(result.rationale)

    if result.user_facing_message:
        st.info(f"**Message alfred_ would send:**\n\n*{result.user_facing_message}*")

    _render_match_badge(result, SCENARIOS[scenario_idx].expected_decision)
    st.divider()
    _render_under_the_hood(result)


def _run_pipeline(scenario_idx: int, failure_sim: str) -> None:
    scenario = SCENARIOS[scenario_idx]
    sim_value = _FAILURE_SIM_MAP[failure_sim]
    with st.spinner("Running decision pipeline…"):
        try:
            result = decide(scenario.input, simulate_failure=sim_value)
            st.session_state["last_result"] = result
        except Exception as exc:  # noqa: BLE001 — surface any crash to the UI
            st.session_state["last_result"] = None
            st.error(f"Decision pipeline crashed: {exc!r}")


def main() -> None:
    _init_session_state()

    st.title("alfred_ Execution Decision Layer")
    st.caption(
        "Trust is the product. Prototype demonstrating the code/LLM split for action decisions."
    )

    _render_sidebar()

    left, right = st.columns([1, 1.3])
    with left:
        scenario_idx, failure_sim, run_clicked = _render_left_column()

    if run_clicked:
        _run_pipeline(scenario_idx, failure_sim)

    with right:
        _render_right_column(scenario_idx)


main()
