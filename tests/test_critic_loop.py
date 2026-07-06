"""End-to-end tests for the agentic critic loop (`run_tool_loop`).

These use a mock LLMClient that scripts a list of assistant turns. Each
scripted turn is a list of tool calls (dicts of {id, name, arguments}).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pytest

from pact.llm import ToolLoopResult, run_tool_loop


class MockChoice:
    def __init__(self, message):
        self.message = message


class MockResponse:
    def __init__(self, message):
        self.choices = [MockChoice(message)]


@dataclass
class MockFunction:
    name: str
    arguments: str


@dataclass
class MockToolCall:
    id: str
    function: MockFunction
    type: str = "function"


class MockMessage:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []


class ScriptedClient:
    """Replays a queue of assistant messages.

    `script` is a list of (content, tool_calls) tuples. Each element of
    `tool_calls` is a `(id, name, arguments_dict)` triple.
    """

    def __init__(self, script):
        self.script = list(script)
        self.calls: list[dict[str, Any]] = []

    def chat_raw(self, messages, tools=None, tool_choice=None):
        if not self.script:
            raise RuntimeError("script exhausted but loop still calling")
        content, tool_calls = self.script.pop(0)
        self.calls.append({
            "tool_choice": tool_choice,
            "n_messages": len(messages),
        })
        message_tool_calls = [
            MockToolCall(
                id=tc[0],
                function=MockFunction(name=tc[1], arguments=json.dumps(tc[2])),
            )
            for tc in tool_calls
        ]
        return MockMessage(content=content, tool_calls=message_tool_calls)


def _terminate_immediately_script(ops):
    return [(None, [("call_1", "submit_ops", {"ops": ops})])]


def _investigate_then_terminate_script(edit_index, ops):
    return [
        (None, [("call_1", "get_original_trajectories", {"edit_index": edit_index})]),
        (None, [("call_2", "submit_ops", {"ops": ops})]),
    ]


def _handler_returning(value):
    def handler(name, args):
        return json.dumps({"echo": args, "from": name, "value": value})
    return handler


def test_terminate_immediately():
    client = ScriptedClient(_terminate_immediately_script([]))
    result = run_tool_loop(
        client,
        system="SYS",
        initial_user="USER",
        tools=[],
        investigation_tools={"get_original_trajectories"},
        terminator_tool="submit_ops",
        investigation_handler=_handler_returning("never called"),
    )
    assert result.final_tool_call is not None
    assert result.final_tool_call["name"] == "submit_ops"
    assert result.final_tool_call["arguments"] == {"ops": []}
    assert result.loop_stats["investigation_calls"] == 0
    assert result.loop_stats["forced_terminate"] is False
    assert result.loop_stats["total_turns"] == 1


def test_one_investigation_then_terminate():
    client = ScriptedClient(_investigate_then_terminate_script(0, []))
    result = run_tool_loop(
        client,
        system="SYS",
        initial_user="USER",
        tools=[],
        investigation_tools={"get_original_trajectories"},
        terminator_tool="submit_ops",
        investigation_handler=_handler_returning("trace data"),
    )
    assert result.final_tool_call["name"] == "submit_ops"
    assert result.loop_stats["investigation_calls"] == 1
    assert result.loop_stats["total_turns"] == 2

    # Transcript shape: system + user + assistant(tool_call) + tool(result) + assistant(submit_ops)
    roles = [m["role"] for m in result.messages]
    assert roles == ["system", "user", "assistant", "tool", "assistant"]

    # Tool result message links back via tool_call_id.
    tool_msg = result.messages[3]
    assert tool_msg["tool_call_id"] == "call_1"
    assert tool_msg["name"] == "get_original_trajectories"
    payload = json.loads(tool_msg["content"])
    assert payload["from"] == "get_original_trajectories"
    assert payload["echo"] == {"edit_index": 0}


def test_max_investigations_forces_submit():
    """After 5 investigations the harness forces tool_choice=submit_ops."""
    script = []
    for i in range(5):
        script.append((None, [(f"call_inv{i}", "get_original_trajectories", {"edit_index": i})]))
    # On the 6th turn the model still tries to investigate — but harness
    # has set tool_choice=submit_ops so any well-behaved model would
    # submit. We script it to comply.
    script.append((None, [("call_final", "submit_ops", {"ops": []})]))

    client = ScriptedClient(script)
    result = run_tool_loop(
        client,
        system="SYS",
        initial_user="USER",
        tools=[],
        investigation_tools={"get_original_trajectories"},
        terminator_tool="submit_ops",
        investigation_handler=_handler_returning("ok"),
        max_investigations=5,
        max_turns=7,
    )
    assert result.loop_stats["investigation_calls"] == 5
    assert result.loop_stats["forced_terminate"] is True
    assert result.final_tool_call["name"] == "submit_ops"

    # The 6th (final) chat_raw call must have used forced tool_choice.
    last_call_choice = client.calls[-1]["tool_choice"]
    assert isinstance(last_call_choice, dict)
    assert last_call_choice["function"]["name"] == "submit_ops"


def test_max_investigations_can_skip_forced_terminator():
    client = ScriptedClient(_terminate_immediately_script([]))
    result = run_tool_loop(
        client,
        system="SYS",
        initial_user="USER",
        tools=[],
        investigation_tools=set(),
        terminator_tool="submit_ops",
        investigation_handler=_handler_returning("never"),
        max_investigations=0,
        force_terminator_after_investigations=False,
    )

    assert result.final_tool_call["name"] == "submit_ops"
    assert result.loop_stats["forced_terminate"] is False
    assert client.calls[0]["tool_choice"] == "auto"


def test_unknown_tool_returns_error_to_model():
    script = [
        (None, [("call_1", "nonexistent_tool", {"foo": 1})]),
        (None, [("call_2", "submit_ops", {"ops": []})]),
    ]
    client = ScriptedClient(script)
    result = run_tool_loop(
        client,
        system="SYS",
        initial_user="USER",
        tools=[],
        investigation_tools={"get_original_trajectories"},
        terminator_tool="submit_ops",
        investigation_handler=_handler_returning("never"),
    )
    assert result.final_tool_call["name"] == "submit_ops"
    # The unknown-tool error must appear in the transcript as a tool message.
    tool_msgs = [m for m in result.messages if m["role"] == "tool"]
    assert len(tool_msgs) == 1
    assert "unknown tool" in tool_msgs[0]["content"]


def test_no_tool_then_submit_via_nudge():
    """Model writes prose first (no tool), harness nudges, model then submits."""
    script = [
        ("Let me think about whether any rule covers this failure...", []),  # content only
        (None, [("call_1", "submit_ops", {"ops": []})]),                       # nudge -> submit
    ]
    client = ScriptedClient(script)
    result = run_tool_loop(
        client,
        system="SYS",
        initial_user="USER",
        tools=[],
        investigation_tools={"get_original_trajectories"},
        terminator_tool="submit_ops",
        investigation_handler=_handler_returning("never"),
    )
    assert result.final_tool_call is not None
    assert result.final_tool_call["name"] == "submit_ops"
    assert result.loop_stats["no_tool_nudges"] == 1
    assert result.loop_stats["total_turns"] == 2
    # Transcript: system + user + assistant(text only) + user(nudge) + assistant(submit)
    roles = [m["role"] for m in result.messages]
    assert roles == ["system", "user", "assistant", "user", "assistant"]
    # First chat used tool_choice="auto" (not "required").
    assert client.calls[0]["tool_choice"] == "auto"
    # Nudge content reaches the model.
    nudge = result.messages[3]["content"]
    assert "call exactly one tool" in nudge


def test_no_tool_nudge_omits_investigation_tool_when_unavailable():
    script = [
        ("Let me think first.", []),
        (None, [("call_1", "submit_ops", {"ops": []})]),
    ]
    client = ScriptedClient(script)
    result = run_tool_loop(
        client,
        system="SYS",
        initial_user="USER",
        tools=[],
        investigation_tools=set(),
        terminator_tool="submit_ops",
        investigation_handler=_handler_returning("never"),
    )

    nudge = result.messages[3]["content"]
    assert "get_original_trajectories" not in nudge
    assert "`submit_ops(ops=[...])`" in nudge


def test_repeated_no_tool_exhausts_nudges():
    """If the model never calls a tool, nudges cap and we surface an error."""
    # 1 initial + max_no_tool_nudges (default 2) nudge follow-ups = 3 no-tool turns.
    script = [("I'm just thinking out loud.", []) for _ in range(5)]
    client = ScriptedClient(script)
    result = run_tool_loop(
        client,
        system="SYS",
        initial_user="USER",
        tools=[],
        investigation_tools={"get_original_trajectories"},
        terminator_tool="submit_ops",
        investigation_handler=_handler_returning("never"),
        max_no_tool_nudges=2,
        max_turns=10,
    )
    assert result.final_tool_call is None
    assert result.error is not None
    assert "no tool_calls after 2" in result.error
    assert result.loop_stats["no_tool_nudges"] == 3
    assert result.loop_stats["total_turns"] == 3


def test_hard_cap_without_submit_yields_error():
    """If max_turns is hit without submit_ops, harness reports an error."""
    # Force a model that never submits: every turn investigates a new edit.
    script = [(None, [(f"call_{i}", "get_original_trajectories", {"edit_index": i})]) for i in range(10)]
    client = ScriptedClient(script)
    result = run_tool_loop(
        client,
        system="SYS",
        initial_user="USER",
        tools=[],
        investigation_tools={"get_original_trajectories"},
        terminator_tool="submit_ops",
        investigation_handler=_handler_returning("ok"),
        max_investigations=10,
        max_turns=3,
    )
    assert result.final_tool_call is None
    assert result.error is not None
    assert "hard cap" in result.error
    assert result.loop_stats["total_turns"] == 3
