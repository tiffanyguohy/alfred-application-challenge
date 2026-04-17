"""Streamlit UI for the alfred_ Execution Decision Layer.

Layout:
  Left column  — tabs: Preloaded scenarios | Custom scenario builder.
  Right column — decision outcome and under-the-hood expanders.

All decision logic lives in alfred/*. This module is a thin presentational wrapper.
"""
from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from typing import Any, Callable, get_args

import streamlit as st
from dotenv import load_dotenv

from alfred.decide import decide
from alfred.scenarios import SCENARIOS
from alfred.types import (
    ActionType,
    Contact,
    DecisionInput,
    DecisionResult,
    DecisionType,
    Message,
    ProposedAction,
    UserState,
)

load_dotenv()

st.set_page_config(
    page_title="alfred_ Decision Layer",
    layout="wide",
)


_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500;600&family=Fira+Sans:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"], .stMarkdown, .stTextInput, .stSelectbox, .stTextArea, .stButton, .stRadio, .stExpander {
    font-family: 'Fira Sans', system-ui, sans-serif !important;
}
code, pre, .stCode, .stCode pre, .stMarkdown code {
    font-family: 'Fira Code', ui-monospace, monospace !important;
    font-size: 0.85rem !important;
}

/* Decision badge — color-coded left border, slate surface */
.alfred-badge {
    border-left: 4px solid var(--badge-color, #64748B);
    background: #1E293B;
    border-radius: 6px;
    padding: 14px 16px;
    margin: 4px 0 12px 0;
    font-family: 'Fira Code', ui-monospace, monospace;
}
.alfred-badge .decision {
    font-size: 1.05rem;
    font-weight: 600;
    color: #F8FAFC;
}
.alfred-badge .source {
    font-size: 0.78rem;
    color: #94A3B8;
    margin-top: 4px;
}

/* Match badges */
.alfred-match-pass {
    color: #22C55E;
    font-size: 0.85rem;
    font-weight: 500;
}
.alfred-match-fail {
    color: #EF4444;
    font-size: 0.85rem;
    font-weight: 500;
}

/* Section labels */
.alfred-section-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    color: #94A3B8;
    text-transform: uppercase;
    margin: 18px 0 8px 0;
}

/* Conversation turn lines */
.alfred-turn-meta {
    font-family: 'Fira Code', monospace;
    font-size: 0.78rem;
    color: #94A3B8;
    margin-bottom: 2px;
}
.alfred-turn-content {
    color: #E2E8F0;
    font-size: 0.92rem;
    margin-bottom: 10px;
    white-space: pre-wrap;
}

/* Tab labels */
.stTabs [data-baseweb="tab-list"] button {
    font-family: 'Fira Sans', sans-serif;
    font-weight: 500;
}

/* Buttons — subtle hover */
.stButton button {
    transition: background-color 200ms ease, border-color 200ms ease;
}

/* Tighten dividers */
hr {
    border-color: #1E293B !important;
    opacity: 0.6;
}
</style>
"""

st.markdown(_CSS, unsafe_allow_html=True)


_FAILURE_SIM_MAP: dict[str, str | None] = {
    "None": None,
    "Timeout": "timeout",
    "Malformed output": "malformed",
}

_DECISION_COLORS: dict[DecisionType, str] = {
    DecisionType.EXECUTE_SILENTLY: "#22C55E",
    DecisionType.EXECUTE_AND_NOTIFY: "#3B82F6",
    DecisionType.CONFIRM_BEFORE_EXECUTING: "#F59E0B",
    DecisionType.ASK_CLARIFYING_QUESTION: "#F97316",
    DecisionType.REFUSE_OR_ESCALATE: "#EF4444",
}

_ACTION_TYPES: list[str] = list(get_args(ActionType))

_PARAM_PLACEHOLDERS: dict[str, str] = {
    "reminder_create": '{"text": "call mom", "time": "18:00"}',
    "calendar_read": '{"date": "2026-04-18"}',
    "calendar_event_create": '{"title": "Lunch with Sarah", "start": "2026-04-22T12:00:00", "duration_min": 60}',
    "calendar_event_update": '{"event_id": "evt_123", "new_start": "2026-04-17T16:00:00"}',
    "email_send": '{"to": "boss@company.com", "subject": "Q3 update", "body": "Here is the summary..."}',
    "email_draft": '{"to": "boss@company.com", "subject": "Q3 update", "body": "Here is the summary..."}',
    "wire_transfer": '{"to_account": "Acme Inc", "amount_usd": 5000, "memo": "Invoice #42"}',
    "payment": '{"to": "Acme", "amount_usd": 50, "memo": "Subscription"}',
}


def _has_api_key() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _init_session_state() -> None:
    st.session_state.setdefault("scenario_idx", 0)
    st.session_state.setdefault("last_result", None)
    st.session_state.setdefault("last_expected", None)
    st.session_state.setdefault("custom_turn_ids", [])


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def _render_sidebar() -> None:
    st.sidebar.markdown("### The five decisions")
    st.sidebar.markdown(
        "`execute_silently`\n\n"
        "`execute_and_notify`\n\n"
        "`confirm_before_executing`\n\n"
        "`ask_clarifying_question`\n\n"
        "`refuse_or_escalate`"
    )
    st.sidebar.caption("Policy rules can override the model decision.")
    st.sidebar.divider()
    if _has_api_key():
        st.sidebar.success("API key detected")
    else:
        st.sidebar.warning("No API key — only failure-sim paths work")


# ---------------------------------------------------------------------------
# Preloaded tab
# ---------------------------------------------------------------------------

def _render_preloaded_tab() -> tuple[int, str, bool]:
    scenario_idx = st.selectbox(
        "Pick a scenario",
        options=list(range(len(SCENARIOS))),
        format_func=lambda i: f"{i + 1}. {SCENARIOS[i].name}",
        key="scenario_idx",
    )
    scenario = SCENARIOS[scenario_idx]

    meta_cols = st.columns(3)
    meta_cols[0].markdown(f"**Category** &nbsp;`{scenario.category}`")
    meta_cols[1].markdown(f"**Must pass** &nbsp;{'yes' if scenario.must_pass else 'no'}")
    meta_cols[2].markdown(f"**Expected** &nbsp;`{scenario.expected_decision.value}`")

    action = scenario.input.proposed_action
    with st.expander("Proposed action", expanded=True):
        st.markdown(f"**{action.action_type}**")
        st.write(action.description)
        st.code(json.dumps(action.parameters, indent=2), language="json")

    with st.expander("Conversation history", expanded=True):
        if not scenario.input.history:
            st.caption("(no prior messages)")
        for msg in scenario.input.history:
            st.markdown(
                f"<div class='alfred-turn-meta'>{msg.role} &middot; {msg.timestamp}</div>"
                f"<div class='alfred-turn-content'>{msg.content}</div>",
                unsafe_allow_html=True,
            )

    with st.expander("User state", expanded=False):
        us = scenario.input.user_state
        st.markdown(f"- `silent_send_enabled`: **{us.silent_send_enabled}**")
        st.markdown(
            f"- `financial_confirmation_threshold_usd`: **{us.financial_confirmation_threshold_usd}**"
        )
        st.markdown(f"- `preferred_calendar_autonomy`: **{us.preferred_calendar_autonomy}**")
        st.markdown(f"- `external_email_default`: **{us.external_email_default}**")
        st.markdown(f"- `contacts`: **{len(us.contacts)}**")
        if us.contacts:
            for c in us.contacts:
                st.markdown(f"  - {c.name} — `{c.email}`")

    if scenario.notes:
        st.caption(scenario.notes)

    failure_sim = st.radio(
        "Simulate LLM failure",
        options=list(_FAILURE_SIM_MAP.keys()),
        horizontal=True,
        key="failure_sim_preloaded",
    )

    needs_network = _FAILURE_SIM_MAP[failure_sim] is None
    disabled = needs_network and not _has_api_key()
    run_clicked = st.button(
        "Run decision pipeline",
        type="primary",
        disabled=disabled,
        help="Set ANTHROPIC_API_KEY or use a failure simulation." if disabled else None,
        key="run_preloaded",
    )

    return scenario_idx, failure_sim, run_clicked


# ---------------------------------------------------------------------------
# Custom tab
# ---------------------------------------------------------------------------

def _add_custom_turn(role: str = "user", content: str = "") -> None:
    new_id = uuid.uuid4().hex[:8]
    st.session_state["custom_turn_ids"].append(new_id)
    st.session_state[f"custom_role_{new_id}"] = role
    st.session_state[f"custom_content_{new_id}"] = content
    st.session_state[f"custom_ts_{new_id}"] = _now_iso()


def _remove_custom_turn(turn_id: str) -> None:
    if turn_id in st.session_state["custom_turn_ids"]:
        st.session_state["custom_turn_ids"].remove(turn_id)
    for prefix in ("custom_role_", "custom_content_", "custom_ts_"):
        st.session_state.pop(f"{prefix}{turn_id}", None)


def _render_custom_tab() -> tuple[str, bool, dict[str, Any]]:
    st.markdown("Submit any action plus context. The pipeline runs unchanged.")

    action_type = st.selectbox(
        "Action type",
        options=_ACTION_TYPES,
        key="custom_action_type",
    )
    description = st.text_input(
        "Description",
        placeholder="One-line summary of what alfred_ would do",
        key="custom_description",
    )
    parameters_json = st.text_area(
        "Parameters (JSON)",
        value=st.session_state.get(
            "custom_parameters_json", _PARAM_PLACEHOLDERS.get(action_type, "{}")
        ),
        height=110,
        key="custom_parameters_json",
        help="Action-dependent JSON object. Switch action type for example shapes.",
    )

    st.markdown("<div class='alfred-section-label'>Conversation history</div>", unsafe_allow_html=True)

    turn_ids = list(st.session_state["custom_turn_ids"])
    if not turn_ids:
        st.caption("No turns yet. Add one below.")

    for turn_id in turn_ids:
        cols = st.columns([1.3, 5, 2.5, 0.7])
        cols[0].selectbox(
            "Role",
            ["user", "alfred"],
            key=f"custom_role_{turn_id}",
            label_visibility="collapsed",
        )
        cols[1].text_area(
            "Content",
            key=f"custom_content_{turn_id}",
            label_visibility="collapsed",
            height=70,
            placeholder="Message content",
        )
        cols[2].text_input(
            "Timestamp",
            key=f"custom_ts_{turn_id}",
            label_visibility="collapsed",
            placeholder="ISO timestamp",
        )
        if cols[3].button("Remove", key=f"custom_rm_{turn_id}"):
            _remove_custom_turn(turn_id)
            st.rerun()

    add_col, _ = st.columns([1, 4])
    if add_col.button("Add turn", key="custom_add_turn"):
        _add_custom_turn()
        st.rerun()

    with st.expander("User state", expanded=False):
        silent_send = st.toggle(
            "silent_send_enabled",
            value=False,
            key="custom_silent_send",
        )
        threshold = st.number_input(
            "financial_confirmation_threshold_usd",
            min_value=0.0,
            value=1000.0,
            step=100.0,
            key="custom_threshold",
        )
        autonomy = st.selectbox(
            "preferred_calendar_autonomy",
            ["low", "medium", "high"],
            index=1,
            key="custom_autonomy",
        )
        email_default = st.selectbox(
            "external_email_default",
            ["confirm", "notify_after"],
            index=0,
            key="custom_email_default",
        )
        contacts_json = st.text_area(
            "Contacts (JSON list)",
            value="[]",
            height=70,
            key="custom_contacts_json",
            help='Example: [{"name": "Sarah Chen", "email": "sarah@x.com"}]',
        )

    failure_sim = st.radio(
        "Simulate LLM failure",
        options=list(_FAILURE_SIM_MAP.keys()),
        horizontal=True,
        key="failure_sim_custom",
    )

    needs_network = _FAILURE_SIM_MAP[failure_sim] is None
    disabled = needs_network and not _has_api_key()
    run_clicked = st.button(
        "Run decision pipeline",
        type="primary",
        disabled=disabled,
        help="Set ANTHROPIC_API_KEY or use a failure simulation." if disabled else None,
        key="run_custom",
    )

    form_data = {
        "action_type": action_type,
        "description": description,
        "parameters_json": parameters_json,
        "turn_ids": turn_ids,
        "user_state": {
            "silent_send_enabled": silent_send,
            "financial_confirmation_threshold_usd": float(threshold),
            "preferred_calendar_autonomy": autonomy,
            "external_email_default": email_default,
            "contacts_json": contacts_json,
        },
    }
    return failure_sim, run_clicked, form_data


# ---------------------------------------------------------------------------
# Right column — decision rendering
# ---------------------------------------------------------------------------

def _render_decision_badge(result: DecisionResult) -> None:
    color = _DECISION_COLORS.get(result.final_decision, "#64748B")
    st.markdown(
        f"<div class='alfred-badge' style='--badge-color: {color}'>"
        f"<div class='decision'>{result.final_decision.value}</div>"
        f"<div class='source'>source: {result.decision_source}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


def _render_match_badge(result: DecisionResult, expected: DecisionType) -> None:
    if result.final_decision == expected:
        st.markdown(
            "<div class='alfred-match-pass'>matches expected decision</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div class='alfred-match-fail'>mismatch — expected "
            f"<code>{expected.value}</code>, got <code>{result.final_decision.value}</code></div>",
            unsafe_allow_html=True,
        )


def _render_under_the_hood(result: DecisionResult) -> None:
    with st.expander("Conversation State", expanded=True):
        cs = result.conversation_state
        st.markdown(f"**pending_constraints:** {cs.pending_constraints or '_(none)_'}")
        st.markdown(f"**unresolved_references:** {cs.unresolved_references or '_(none)_'}")
        st.markdown(f"**last_drafted_artifact:** {cs.last_drafted_artifact or '_(none)_'}")
        st.markdown(f"**awaiting_confirmation:** {cs.awaiting_confirmation or '_(none)_'}")

    with st.expander("Computed Signals"):
        s = result.signals
        st.markdown(f"- **reversibility:** `{s.reversibility}`")
        st.markdown(f"- **blast_radius:** `{s.blast_radius}`")
        st.markdown(f"- **intent_clarity_score:** `{s.intent_clarity_score}`")
        st.markdown(f"- **missing_parameters:** {s.missing_parameters or '_(none)_'}")
        st.markdown(f"- **entity_ambiguities:** {s.entity_ambiguities or '_(none)_'}")
        st.markdown(
            f"- **risk_contributing_factors:** {s.risk_contributing_factors or '_(none)_'}"
        )

    with st.expander("Policy Check"):
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

    with st.expander("System Prompt"):
        st.code(result.system_prompt or "(prompt not built — short-circuited)", language="markdown")

    with st.expander("User Prompt"):
        st.code(result.user_prompt or "(prompt not built — short-circuited)", language="markdown")

    with st.expander("Raw LLM Output"):
        if result.raw_llm_output is None:
            st.caption("(none — LLM was not called or returned nothing)")
        else:
            st.code(json.dumps(result.raw_llm_output, indent=2, default=str), language="json")

    with st.expander("Parsed LLM Decision"):
        if result.parsed_llm_decision is None:
            st.caption("(none)")
        else:
            st.code(
                json.dumps(result.parsed_llm_decision.model_dump(), indent=2, default=str),
                language="json",
            )

    with st.expander("Decision Source"):
        st.markdown(f"Source: `{result.decision_source}`")

    with st.expander("Timings"):
        rows = [{"stage": k, "ms": round(v, 2)} for k, v in result.timings_ms.items()]
        st.table(rows)

    with st.expander("Errors"):
        if not result.errors:
            st.caption("(no errors)")
        else:
            for err in result.errors:
                st.warning(err)


def _render_right_column() -> None:
    result: DecisionResult | None = st.session_state.get("last_result")
    if result is None:
        st.markdown(
            "<div style='color:#64748B; padding: 24px 0;'>"
            "Click <b>Run decision pipeline</b> on the left to evaluate."
            "</div>",
            unsafe_allow_html=True,
        )
        return

    _render_decision_badge(result)

    st.markdown("<div class='alfred-section-label'>Rationale</div>", unsafe_allow_html=True)
    st.markdown(result.rationale)

    if result.user_facing_message:
        st.info(f"**Message alfred_ would send:**\n\n*{result.user_facing_message}*")

    expected: DecisionType | None = st.session_state.get("last_expected")
    if expected is not None:
        _render_match_badge(result, expected)

    st.divider()
    _render_under_the_hood(result)


# ---------------------------------------------------------------------------
# Pipeline runners
# ---------------------------------------------------------------------------

def _run_pipeline_preloaded(scenario_idx: int, failure_sim: str) -> None:
    scenario = SCENARIOS[scenario_idx]
    sim_value = _FAILURE_SIM_MAP[failure_sim]
    with st.spinner("Running decision pipeline…"):
        try:
            result = decide(scenario.input, simulate_failure=sim_value)
            st.session_state["last_result"] = result
            st.session_state["last_expected"] = scenario.expected_decision
        except Exception as exc:  # noqa: BLE001
            st.session_state["last_result"] = None
            st.session_state["last_expected"] = None
            st.error(f"Decision pipeline crashed: {exc!r}")


def _build_custom_input(form_data: dict[str, Any]) -> DecisionInput:
    parameters = json.loads(form_data["parameters_json"] or "{}")
    if not isinstance(parameters, dict):
        raise ValueError("Parameters must be a JSON object")

    contacts_raw = json.loads(form_data["user_state"]["contacts_json"] or "[]")
    if not isinstance(contacts_raw, list):
        raise ValueError("Contacts must be a JSON list")
    contacts = [Contact(**c) for c in contacts_raw]

    history: list[Message] = []
    for tid in form_data["turn_ids"]:
        role = st.session_state.get(f"custom_role_{tid}", "user")
        content = st.session_state.get(f"custom_content_{tid}", "")
        timestamp = st.session_state.get(f"custom_ts_{tid}", _now_iso())
        history.append(Message(role=role, content=content, timestamp=timestamp))

    user_state = UserState(
        silent_send_enabled=form_data["user_state"]["silent_send_enabled"],
        financial_confirmation_threshold_usd=form_data["user_state"][
            "financial_confirmation_threshold_usd"
        ],
        preferred_calendar_autonomy=form_data["user_state"]["preferred_calendar_autonomy"],
        external_email_default=form_data["user_state"]["external_email_default"],
        contacts=contacts,
    )

    return DecisionInput(
        proposed_action=ProposedAction(
            action_type=form_data["action_type"],
            parameters=parameters,
            description=form_data["description"],
        ),
        history=history,
        user_state=user_state,
    )


def _run_pipeline_custom(form_data: dict[str, Any], failure_sim: str) -> None:
    if not form_data["description"].strip():
        st.error("Description is required.")
        return
    try:
        decision_input = _build_custom_input(form_data)
    except json.JSONDecodeError as exc:
        st.error(f"Invalid JSON: {exc.msg} (line {exc.lineno}, col {exc.colno})")
        return
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not build scenario: {exc}")
        return

    sim_value = _FAILURE_SIM_MAP[failure_sim]
    with st.spinner("Running decision pipeline…"):
        try:
            result = decide(decision_input, simulate_failure=sim_value)
            st.session_state["last_result"] = result
            st.session_state["last_expected"] = None
        except Exception as exc:  # noqa: BLE001
            st.session_state["last_result"] = None
            st.session_state["last_expected"] = None
            st.error(f"Decision pipeline crashed: {exc!r}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _init_session_state()

    st.title("alfred_ Decision Layer")
    st.caption("Action decision pipeline inspector — preloaded scenarios + custom submission.")

    _render_sidebar()

    left, right = st.columns([1, 1.3])
    with left:
        preloaded_tab, custom_tab = st.tabs(["Preloaded", "Custom"])

        run_preloaded = False
        run_custom = False
        scenario_idx = 0
        sim_preloaded = "None"
        sim_custom = "None"
        custom_form_data: dict[str, Any] | None = None

        with preloaded_tab:
            scenario_idx, sim_preloaded, run_preloaded = _render_preloaded_tab()
        with custom_tab:
            sim_custom, run_custom, custom_form_data = _render_custom_tab()

    if run_preloaded:
        _run_pipeline_preloaded(scenario_idx, sim_preloaded)
    elif run_custom and custom_form_data is not None:
        _run_pipeline_custom(custom_form_data, sim_custom)

    with right:
        _render_right_column()


main()
