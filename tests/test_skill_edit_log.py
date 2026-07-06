"""Tests for SkillState.record_edits + edit_log persistence and the
run_experiment._build_trajectory_handler bound to disk layout."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pact.io import write_json
from pact.run_experiment import _build_trajectory_handler
from pact.skill import EditRecord, SkillState
from pact.skill_ops import apply_skill_ops


def _add_op(title="T", content="When X, do Y.", why="Because."):
    return {
        "op": "add",
        "rule": {"title": title, "content": content, "why": why},
    }


def test_record_edits_appends_one_record_per_op():
    skill = SkillState(original="", current="", version=0)
    patch = json.dumps({"ops": [_add_op(title="A"), _add_op(title="B")]})
    result = apply_skill_ops(skill.current, patch)
    assert result.ok

    skill.current = result.text
    skill.version += 1
    skill.record_edits(
        result.ops,
        rules_before=result.rules_before,
        rules_after=result.rules_after,
        group_index=3,
        round_index=0,
        group_dir="evolution/groups/0003_x",
    )
    assert len(skill.edit_log) == 2
    assert [r.edit_index for r in skill.edit_log] == [0, 1]
    assert [r.rule_id for r in skill.edit_log] == ["001", "002"]
    assert [r.op for r in skill.edit_log] == ["add", "add"]
    assert skill.edit_log[0].after["title"] == "A"
    assert skill.edit_log[1].after["title"] == "B"


def test_record_edits_update_captures_before_and_after(tmp_path):
    skill = SkillState(original="", current="", version=0)

    add_patch = json.dumps({"ops": [_add_op(title="T", content="When X, do Y.")]})
    add_result = apply_skill_ops(skill.current, add_patch)
    skill.current = add_result.text
    skill.record_edits(
        add_result.ops,
        rules_before=add_result.rules_before,
        rules_after=add_result.rules_after,
        group_index=0,
        round_index=0,
        group_dir="evolution/groups/0000_a",
    )

    update_patch = json.dumps({"ops": [{
        "op": "update",
        "id": "001",
        "rule": {"title": "T2", "content": "When X, do Z.", "why": "W"},
    }]})
    update_result = apply_skill_ops(skill.current, update_patch)
    skill.current = update_result.text
    skill.record_edits(
        update_result.ops,
        rules_before=update_result.rules_before,
        rules_after=update_result.rules_after,
        group_index=1,
        round_index=0,
        group_dir="evolution/groups/0001_b",
    )

    assert len(skill.edit_log) == 2
    update_rec = skill.edit_log[1]
    assert update_rec.op == "update"
    assert update_rec.before["content"] == "When X, do Y."
    assert update_rec.after["content"] == "When X, do Z."


def test_edit_log_persistence_round_trip(tmp_path: Path):
    skill = SkillState(original="", current="", version=0)
    patch = json.dumps({"ops": [_add_op()]})
    result = apply_skill_ops(skill.current, patch)
    skill.current = result.text
    skill.record_edits(
        result.ops,
        rules_before=result.rules_before,
        rules_after=result.rules_after,
        group_index=0,
        round_index=0,
        group_dir="evolution/groups/0000_a",
    )

    path = tmp_path / "edit_log.json"
    skill.save_edit_log(path)

    reloaded = SkillState(original="", current="", version=0)
    reloaded.load_edit_log(path)
    assert len(reloaded.edit_log) == 1
    assert reloaded.edit_log[0].rule_id == "001"
    assert reloaded.edit_log[0].after["content"] == "When X, do Y."
    assert reloaded.edit_log[0].after["why"] == "Because."


def test_trajectory_handler_reads_originating_files(tmp_path: Path):
    out_dir = tmp_path / "run"
    group_dir_rel = "evolution/groups/0007_demo"
    group_dir_abs = out_dir / group_dir_rel
    group_dir_abs.mkdir(parents=True)

    write_json(group_dir_abs / "task.json", {"id": "task-42", "_path": "data/task42.json"})
    write_json(group_dir_abs / "student_initial.json", {
        "skill_used": "[RULE 001] T\ncontent: When ...\n",
        "raw": "<think>look around</think>\nuse desklamp 1",
        "final_answer": "use desklamp 1",
        "evaluation": {"success": False, "mode": "official", "reason": "lamp not located"},
    })
    write_json(group_dir_abs / "teacher.json", {
        "skill_used": "[RULE 001] T\ncontent: When ...\n",
        "raw": "<think>dresser first</think>\ngo to dresser; use desklamp",
        "final_answer": "go to dresser; use desklamp",
        "evaluation": {"success": True, "mode": "official"},
    })

    edit_log = [EditRecord(
        edit_index=0,
        group_index=7,
        round_index=0,
        op="add",
        rule_id="001",
        before=None,
        after={"title": "T", "content": "When ...", "why": "W"},
        group_dir=group_dir_rel,
    )]

    handler = _build_trajectory_handler(out_dir, edit_log)
    payload = json.loads(handler("get_original_trajectories", {"edit_index": 0}))
    assert payload["edit_index"] == 0
    assert payload["op"] == "add"
    assert payload["student_trajectory"]["skill_used"].startswith("[RULE 001]")
    # Critic now sees the full raw model output (thinking trace + answer),
    # not just the extracted final_answer. This is the key fix that lets it
    # diagnose where the agent's reasoning went wrong on single-shot QA.
    assert payload["student_trajectory"]["model_output"] == "<think>look around</think>\nuse desklamp 1"
    assert payload["student_trajectory"]["evaluation"]["success"] is False
    assert payload["student_trajectory"]["evaluation"]["reason"] == "lamp not located"
    assert payload["teacher_trajectory"]["model_output"] == "<think>dresser first</think>\ngo to dresser; use desklamp"
    assert payload["teacher_trajectory"]["evaluation"]["success"] is True


def test_trajectory_handler_rejects_bad_index(tmp_path: Path):
    handler = _build_trajectory_handler(tmp_path, [])
    payload = json.loads(handler("get_original_trajectories", {"edit_index": 99}))
    assert "error" in payload
    assert "out of range" in payload["error"]


def test_trajectory_handler_rejects_unknown_tool(tmp_path: Path):
    handler = _build_trajectory_handler(tmp_path, [])
    payload = json.loads(handler("foo", {}))
    assert "error" in payload
    assert "unknown tool" in payload["error"]
