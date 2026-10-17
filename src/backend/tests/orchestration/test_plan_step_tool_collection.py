"""Plan steps must only collect the tool calls their own reply generated.

Issue #140: every step replayed the session history into its agent and then
extracted tool calls from the **whole** context, so one historical call was
appended into the persisted message once per step (same tool-use id, N
copies). Bedrock-backed upstreams enforce tool-use id uniqueness across the
request and reject the poisoned payload with 400 REQUEST_BODY_INVALID,
permanently wedging the chat. These tests lock the invariant: calls that
existed before the step's reply never re-enter the step log, even when
mid-turn compaction swapped the context list out from under the baseline.
"""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from agentscope.message import Msg, TextBlock, ToolCallBlock, ToolResultBlock
from core.llm.message_compat import session_to_msgs

HISTORY_ROWS = [
    {"role": "user", "content": "历史问题"},
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "历史回答"},
            {"type": "tool_call", "id": "c1", "name": "search", "input": '{"q": "hist"}'},
        ],
    },
    {
        "role": "tool",
        "content": [{"type": "tool_result", "id": "c1", "name": "search", "output": "历史结果"}],
    },
]


class FakeAgent:
    """Fresh per-step agent whose reply appends only its own call."""

    def __init__(self, step_id: str) -> None:
        self.step_id = step_id
        self.state = SimpleNamespace(context=[])

    async def reply(self, inputs=None, **_kw):
        self.state.context.append(
            Msg(name="user", role="user", content=[TextBlock(type="text", text=str(inputs))])
        )
        self.state.context.append(
            Msg(
                name="assistant",
                role="assistant",
                content=[
                    TextBlock(type="text", text=f"{self.step_id} done"),
                    ToolCallBlock(
                        type="tool_call", id=f"call_{self.step_id}", name="search", input="{}"
                    ),
                ],
            )
        )
        self.state.context.append(
            Msg(
                name="tool",
                role="assistant",
                content=[
                    ToolResultBlock(
                        type="tool_result",
                        id=f"call_{self.step_id}",
                        name="search",
                        output=f"result_{self.step_id}",
                    )
                ],
            )
        )
        return SimpleNamespace(content=f"{self.step_id} done")


class CompactingFakeAgent(FakeAgent):
    """A reply under mid-turn compaction: the context list is replaced."""

    async def reply(self, inputs=None, **_kw):
        own = FakeAgent(self.step_id)
        await own.reply(inputs=inputs)
        # The replacement still carries the replayed history's call blocks —
        # compaction protected them — but it is a different list object.
        self.state.context = session_to_msgs(HISTORY_ROWS) + list(own.state.context)
        return SimpleNamespace(content=f"{self.step_id} done")


def _make_plan(step_ids):
    steps = [
        SimpleNamespace(
            step_id=step_id,
            step_order=n,
            title=step_id,
            description=f"do {step_id}",
            expected_tools=[],
            expected_skills=[],
            expected_agents=[],
        )
        for n, step_id in enumerate(step_ids, 1)
    ]
    return SimpleNamespace(
        status="approved",
        steps=steps,
        extra_data={},
        task_input="synthetic",
        title="synthetic",
        description="",
        total_steps=len(step_ids),
    )


@pytest.fixture()
def plan_harness(monkeypatch):
    from orchestration.subagents import plan_mode

    plan = _make_plan(["step_a", "step_b"])
    step_logs = []

    class Service:
        def __init__(self, _db):
            pass

        def get_plan(self, *_a, **_kw):
            return plan

        def update_plan(self, *_a, **kw):
            plan.status = kw.get("status", plan.status)

        def update_step(self, *_a, **kw):
            if "tool_calls_log" in kw:
                step_logs.append(kw["tool_calls_log"])

    agent_factory = []

    async def create(**_kw):
        agent = FakeAgent(f"step_{len(agent_factory) + 1}")
        agent_factory.append(agent)
        return agent, []

    async def history(*_a, **_kw):
        return list(HISTORY_ROWS)

    async def noop(*_a, **_kw):
        return None

    tool_log_writes = []
    monkeypatch.setattr(plan_mode, "PlanService", Service)
    monkeypatch.setattr(plan_mode, "create_agent_executor", create)
    monkeypatch.setattr(plan_mode, "_prepare_history", history)
    monkeypatch.setattr(
        plan_mode, "_resolve_plugin_capabilities", lambda _db, _uid, sids, mids: (sids, mids, [])
    )
    monkeypatch.setattr(plan_mode, "_load_visible_agents", lambda *a: [])
    monkeypatch.setattr(plan_mode, "_build_file_context", lambda *a, **kw: "")
    monkeypatch.setattr(plan_mode, "is_run_cancelled", lambda *a: False)
    monkeypatch.setattr(plan_mode.log_writer, "start_subagent_log", noop)
    monkeypatch.setattr(plan_mode.log_writer, "finish_subagent_log", noop)
    monkeypatch.setattr(plan_mode.log_writer, "subagent_scope", lambda *a, **kw: nullcontext())
    monkeypatch.setattr(
        plan_mode.log_writer,
        "schedule_tool_call_write",
        lambda write: tool_log_writes.append(write),
    )
    monkeypatch.setattr(
        "core.services.project_scope.build_project_ctx_from_chat_id", lambda *a: None
    )
    monkeypatch.setattr(
        "core.services.ontology_service.build_user_ontology_runtime", lambda **kw: (False, {})
    )
    return SimpleNamespace(
        plan_mode=plan_mode, plan=plan, step_logs=step_logs, tool_log_writes=tool_log_writes
    )


async def _run(harness):
    return [
        event
        async for event in harness.plan_mode.astream_execute_plan(
            plan_id="plan-1",
            user_id="u1",
            db=SimpleNamespace(refresh=lambda *_: None),
        )
    ]


async def test_history_calls_are_never_recollected_by_a_step(plan_harness):
    events = await _run(plan_harness)

    assert not any(event["type"] == "plan_error" for event in events)
    assert events[-1]["type"] == "plan_complete" and events[-1]["completed_steps"] == 2
    # Only the calls the steps themselves made — never the replayed "c1".
    assert [e["tool_id"] for e in events if e["type"] == "tool_call"] == [
        "call_step_1",
        "call_step_2",
    ]
    assert [e["tool_id"] for e in events if e["type"] == "tool_result"] == [
        "call_step_1",
        "call_step_2",
    ]
    assert [[entry["tool_id"] for entry in log] for log in plan_harness.step_logs] == [
        ["call_step_1"],
        ["call_step_2"],
    ]
    assert all(write["tool_call_id"] != "c1" for write in plan_harness.tool_log_writes)


async def test_collection_survives_a_mid_turn_context_replacement(plan_harness, monkeypatch):
    original_create = plan_harness.plan_mode.create_agent_executor

    async def compacting_create(**kw):
        agent, clients = await original_create(**kw)
        replacement = CompactingFakeAgent(agent.step_id)
        replacement.state.context = agent.state.context
        return replacement, clients

    monkeypatch.setattr(plan_harness.plan_mode, "create_agent_executor", compacting_create)

    events = await _run(plan_harness)

    assert not any(event["type"] == "plan_error" for event in events)
    assert [e["tool_id"] for e in events if e["type"] == "tool_call"] == [
        "call_step_1",
        "call_step_2",
    ]
