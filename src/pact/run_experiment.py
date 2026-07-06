from __future__ import annotations

import argparse
import csv
import os
import re
import time
from pathlib import Path
from typing import Any

from .adapters import BenchmarkAdapter, load_adapter
from .io import read_json, write_json, write_text
from .llm import LLMClient, LLMConfig, load_env_file, merge_usage, run_tool_loop, usage_delta
from .prompts import (
    CRITIC_SYSTEM,
    CRITIC_TOOLS,
    ROLLOUT_SYSTEM,
    critic_initial_user,
)
from .skill import SkillState
from .skill_ops import (
    apply_skill_ops,
    canonicalize_skill_text,
    parse_skill_rules,
    render_for_agent,
    render_for_critic,
)


ROOT = Path(__file__).resolve().parents[2]


def _slim_eval_for_critic(ev: dict[str, Any]) -> dict[str, Any]:
    """Trim a full benchmark evaluation dict down to the fields a skill
    curator actually needs to decide an op.

    Drops duplicates of ``final_answer`` (``prediction``, ``solution_code``),
    backend metadata, absolute paths, and per-case spreadsheet previews —
    these add tens of KB per trajectory and the critic ignores them.
    Keeps per-case pass/fail + the diff message and any stderr/non-zero
    exit code that indicates a crash, so the critic can still reason
    about *why* the rollout failed.
    """
    drop_top = {
        "prediction",
        "solution_code",
        "requires_recalculation_backend",
        "recalculation_backend_available",
        "game_file",
    }
    out = {
        k: v
        for k, v in ev.items()
        if k not in drop_top and k not in {"cases", "steps"}
    }
    # SpreadsheetBench: per-case execution results.
    cases = ev.get("cases")
    if isinstance(cases, list):
        out["cases"] = [_slim_case_for_critic(c) for c in cases]
    # ALFWorld: per-step trace. final_answer only joins the actions and
    # drops every observation, so the critic needs the full step list to
    # see what the agent actually faced at each turn. The slim function
    # also dedupes consecutive observations (action with no env change).
    steps = ev.get("steps")
    if isinstance(steps, list):
        out["steps"] = _slim_steps_for_critic(steps)
    return out


def _slim_case_for_critic(c: dict[str, Any]) -> dict[str, Any]:
    out = {
        "case": c.get("case"),
        "success": c.get("success"),
        "error": c.get("error"),
    }
    exe = c.get("execution") or {}
    rc = exe.get("returncode")
    if rc not in (0, None):
        out["returncode"] = rc
    stderr = (exe.get("stderr") or "").strip()
    if stderr:
        out["stderr"] = stderr if len(stderr) <= 400 else stderr[:400] + "…"
    return out


_STEP_OBS_CHAR_CAP = 280


def _slim_step_for_critic(s: dict[str, Any], *, prev_obs: str | None = None) -> dict[str, Any]:
    """Slim one ALFWorld step. Strategy:

    - `action`: always emit (each step's identity).
    - `observation`: emit only when it differs from the previous step
      (most non-movement actions produce identical state); collapse
      duplicates to the sentinel `"(unchanged)"` so the critic still
      knows the action ran without re-spending tokens. Length-cap a
      changed obs at 280 chars — enough for the type-level cues the
      critic needs (room kind, present object classes, action result)
      without dumping the full room inventory on every step.
    - `admissible` / `won` / `done`: emit ONLY when non-default
      (admissible=False, won=True, done=True). On typical trajectories
      these are at their default on every step and just inflate JSON
      noise by ~50 chars/step. Setting them to defaults conveys nothing
      the critic can act on.
    - `stop_reason`: emit only when present (last step at most).
    """
    out: dict[str, Any] = {"action": s.get("action")}
    obs = (s.get("observation") or "").strip()
    if obs and obs == prev_obs:
        out["observation"] = "(unchanged)"
    elif obs:
        out["observation"] = obs if len(obs) <= _STEP_OBS_CHAR_CAP else obs[:_STEP_OBS_CHAR_CAP] + "…"
    if s.get("admissible") is False:
        out["admissible"] = False
    if s.get("won"):
        out["won"] = True
    if s.get("done"):
        out["done"] = True
    if s.get("stop_reason"):
        out["stop_reason"] = s["stop_reason"]
    return out


def _slim_steps_for_critic(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Apply per-step slimming with cross-step dedupe of identical observations."""
    out: list[dict[str, Any]] = []
    prev_obs: str | None = None
    for s in steps:
        item = _slim_step_for_critic(s, prev_obs=prev_obs)
        obs_raw = (s.get("observation") or "").strip()
        if obs_raw:
            prev_obs = obs_raw
        out.append(item)
    return out


def _skill_for_agent(skill_text: str) -> str:
    """Render the agent's runtime view: header + content only.
    `why` (student-vs-teacher narrative) is omitted to keep meta-commentary
    out of the agent's context."""
    if not skill_text.strip():
        return ""
    try:
        rules = parse_skill_rules(canonicalize_skill_text(skill_text))
    except ValueError:
        return skill_text
    return render_for_agent(rules)


def _gold_path_for_task(adapter: BenchmarkAdapter, task: dict[str, Any]) -> str:
    """Return the gold answer file path for spreadsheetbench tasks.

    During training (evolve mode) we pass this to the rollout so the
    model can receive cell-level feedback when its code executes
    successfully but produces wrong output.  Returns empty string for
    non-spreadsheet benchmarks or when the file doesn't exist.
    """
    if adapter.name != "spreadsheetbench":
        return ""
    task_id = str(task.get("id", ""))
    dataset_root = task.get("dataset_root", "")
    spreadsheet_path = task.get("spreadsheet_path", f"spreadsheet/{task_id}")
    gold_file = f"1_{task_id}_answer.xlsx"
    gold_path = Path(dataset_root) / spreadsheet_path / gold_file
    return str(gold_path) if gold_path.exists() else ""


def _trajectory_for_critic(
    name: str, rollout: dict[str, Any], evaluation: dict[str, Any]
) -> dict[str, Any]:
    """Compact trajectory shape consumed by the critic.

    A trajectory is exactly: skills the agent saw + the agent's full
    `model_output` (raw reply with thinking trace and the answer, not just
    the extracted answer) + the slimmed evaluation. The full prompt is
    dropped because the task text appears elsewhere in the critic's user
    message. ``raw`` is preferred over ``final_answer`` so the critic can
    diagnose *where* the agent's reasoning went wrong on single-shot QA
    benchmarks; multi-turn benchmarks (alfworld, spreadsheet) still expose
    their trajectory via ``evaluation.steps`` / code, and their ``raw`` is
    just the last action / code block.
    """
    model_output = rollout.get("raw") or rollout.get("final_answer", "")
    return {
        "name": name,
        "skill_used": rollout.get("skill_used", "") or "[None]",
        "model_output": model_output,
        "evaluation": _slim_eval_for_critic(evaluation),
    }


def run_rollout(
    llm: Any,
    adapter: BenchmarkAdapter,
    role: str,
    task: dict[str, Any],
    skill_text: str,
    output_dir: Path | None = None,
    gold_path: str = "",
) -> dict[str, Any]:
    skill_text_for_agent = _skill_for_agent(skill_text)
    # Snapshot usage before the rollout so we can compute the delta after.
    usage_before = llm.get_usage() if hasattr(llm, "get_usage") else None

    if adapter.name == "alfworld":
        from .alfworld_official import run_interactive_alfworld_rollout
        result = run_interactive_alfworld_rollout(task, skill_text_for_agent, llm, role=role)
    elif adapter.name == "spreadsheetbench":
        from .spreadsheet_rollout import run_codegen_multi
        result = run_codegen_multi(
            task, skill_text_for_agent, llm, role=role, output_dir=output_dir,
            gold_path=gold_path
        )
    else:
        messages = adapter.build_messages(task, skill_text_for_agent, output_dir=output_dir)
        # If the adapter didn't supply a system message, prefix the framework default
        # (empty string for the legacy single-user-message path). Adapters that ship
        # their own SkillOpt-aligned system message return it as messages[0] already.
        if not messages or messages[0].get("role") != "system":
            messages = [{"role": "system", "content": ROLLOUT_SYSTEM}] + messages
        raw = llm.chat(messages)
        cleaned = llm.clean_content(raw)
        final_answer = adapter.extract_answer(cleaned)
        # Render a compact text view of the prompt for logs/debugging. Multimodal
        # content parts (e.g. docvqa image_url) are summarised as a placeholder
        # rather than dumping base64 into the trajectory JSON.
        prompt_text = _render_messages_for_log(messages)
        slim_messages = _slim_messages_for_log(messages)
        # Persist the assistant's reply as a regular message so viewers see the
        # full conversation (input + response), not just the prompt.
        slim_messages.append({"role": "assistant", "content": raw})
        result = {
            "role": role,
            "skill_used": skill_text_for_agent,
            "prompt": prompt_text,
            "messages": slim_messages,
            "raw": raw,
            "final_answer": final_answer,
        }

    # Attach usage delta if the client supports tracking.
    if usage_before is not None and hasattr(llm, "get_usage"):
        result["usage"] = usage_delta(usage_before, llm.get_usage())

    return result


def _render_messages_for_log(messages: list[dict[str, Any]]) -> str:
    """Flatten a chat message list into a single text blob for human inspection."""
    parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "?")
        content = msg.get("content", "")
        if isinstance(content, list):
            chunks: list[str] = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    chunks.append(str(part.get("text", "")))
                elif isinstance(part, dict) and part.get("type") == "image_url":
                    chunks.append("[image_url omitted]")
                else:
                    chunks.append(str(part))
            content = "\n".join(chunks)
        parts.append(f"<{role}>\n{content}")
    return "\n\n".join(parts)


def _slim_messages_for_log(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Persist messages without exploding the trajectory JSON with base64
    images. Preserves every field on the message (tool_calls, tool_call_id,
    name, role, …) — only the `content` value is rewritten when it's a
    multimodal parts list, swapping `image_url` data URIs for a sentinel.
    Critic transcripts and rollout transcripts both flow through here, so
    dropping tool-related fields would silently corrupt critic.json."""
    slim: list[dict[str, Any]] = []
    for msg in messages:
        new_msg = dict(msg)
        content = msg.get("content", "")
        if isinstance(content, list):
            new_parts: list[Any] = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    new_parts.append({"type": "image_url", "image_url": "[omitted for log]"})
                else:
                    new_parts.append(part)
            new_msg["content"] = new_parts
        slim.append(new_msg)
    return slim


def _finalize_failed_task(
    group_dir: Path,
    group_summary: dict[str, Any],
    step: str,
    exc: BaseException,
    skill_version: int,
) -> dict[str, Any]:
    """Persist a group as `failed` so resume can detect and retry it.

    `failed=True` is the load-bearing flag: `_load_resumed_skill` filters
    out groups with this set, so they are *not* counted as completed and
    `evolve_on_task` runs again on them next time. The per-step `failed_at`
    is for triage only.
    """
    group_summary["failed"] = True
    group_summary["failure_reason"] = "llm_error"
    group_summary["failed_at"] = step
    group_summary["error"] = f"{type(exc).__name__}: {exc}"
    group_summary["skill_version_end"] = skill_version
    write_json(group_dir / "summary.json", group_summary)
    return group_summary


def _log_task(
    phase: str,
    bench: str,
    idx: int,
    total: int,
    task_id: Any,
    status: str,
    detail: str,
    elapsed: float,
) -> None:
    """One-line per-task progress log. Formats:
        [evolve docvqa  3/10 q63700] OK   round 1 v3->v4 (27.1s)
        [evolve docvqa  4/10 q12550] FAIL skill v4 (42.3s)
        [evolve docvqa  5/10 q47064] ERR  teacher_rollout APIConnectionError (12.1s)
    """
    tid = str(task_id) if task_id is not None else "?"
    if len(tid) > 24:
        tid = tid[:21] + "…"
    width = max(2, len(str(total)))
    head = f"[{phase} {bench} {idx:>{width}}/{total} {tid}]"
    print(f"{head} {status:<4} {detail} ({elapsed:.1f}s)", flush=True)


def evolve_on_task(
    adapter: BenchmarkAdapter,
    task: dict[str, Any],
    task_index: int,
    total_tasks: int,
    skill: SkillState,
    student: Any,
    teacher: Any,
    critic: Any,
    judge: Any,
    out_dir: Path,
    max_iters: int,
    skip_self_evolution_if_teacher_fails: bool,
) -> dict[str, Any]:
    started = time.time()
    group_dir = out_dir / "evolution" / "groups" / f"{task_index:04d}_{safe_id(task)}"
    group_dir.mkdir(parents=True, exist_ok=True)
    write_json(group_dir / "task.json", task)

    task_text = adapter.render_task(task)
    group_summary: dict[str, Any] = {
        "task_index": task_index,
        "task_id": task.get("id") or task.get("_path"),
        "success": False,
        "rounds": [],
        "skill_version_start": skill.version,
    }

    # Per-role usage accumulator for this group.
    group_usage: dict[str, Any] = {}

    skill_v_start = skill.version

    try:
        student_initial = run_rollout(student, adapter, "student", task, skill.current, output_dir=group_dir / "student_initial_outputs", gold_path=_gold_path_for_task(adapter, task))
    except Exception as exc:
        _finalize_failed_task(group_dir, group_summary, "student_initial", exc, skill.version)
        _log_task("evolve", adapter.name, task_index + 1, total_tasks, group_summary["task_id"],
                  "ERR", f"student_initial {type(exc).__name__}", time.time() - started)
        return group_summary
    try:
        eval_initial = adapter.evaluate(
            task,
            student_initial["final_answer"],
            judge=judge,
            trajectory=student_initial,
            output_dir=group_dir / "student_initial_outputs",
        )
    except Exception as exc:
        _finalize_failed_task(group_dir, group_summary, "eval_student_initial", exc, skill.version)
        _log_task("evolve", adapter.name, task_index + 1, total_tasks, group_summary["task_id"],
                  "ERR", f"eval_student_initial {type(exc).__name__}", time.time() - started)
        return group_summary
    write_json(group_dir / "student_initial.json", {**student_initial, "evaluation": eval_initial})
    if student_initial.get("usage"):
        group_usage["student"] = merge_usage(group_usage.get("student"), student_initial["usage"])
    if eval_initial["success"]:
        group_summary["success"] = True
        group_summary["skill_version_end"] = skill.version
        if group_usage:
            group_summary["usage"] = group_usage
        write_json(group_dir / "summary.json", group_summary)
        _log_task("evolve", adapter.name, task_index + 1, total_tasks, group_summary["task_id"],
                  "OK", f"first-try v{skill_v_start}", time.time() - started)
        return group_summary

    latest_student = student_initial
    latest_eval = eval_initial
    task_start_skill = skill.current

    history_trajectories: list[dict[str, Any]] = [
        _trajectory_for_critic("student_initial", student_initial, eval_initial)
    ]

    try:
        teacher_rollout = run_rollout(teacher, adapter, "teacher", task, task_start_skill, output_dir=group_dir / "teacher_outputs", gold_path=_gold_path_for_task(adapter, task))
    except Exception as exc:
        _finalize_failed_task(group_dir, group_summary, "teacher", exc, skill.version)
        _log_task("evolve", adapter.name, task_index + 1, total_tasks, group_summary["task_id"],
                  "ERR", f"teacher {type(exc).__name__}", time.time() - started)
        return group_summary
    try:
        teacher_eval = adapter.evaluate(
            task,
            teacher_rollout["final_answer"],
            judge=judge,
            trajectory=teacher_rollout,
            output_dir=group_dir / "teacher_outputs",
        )
    except Exception as exc:
        _finalize_failed_task(group_dir, group_summary, "eval_teacher", exc, skill.version)
        _log_task("evolve", adapter.name, task_index + 1, total_tasks, group_summary["task_id"],
                  "ERR", f"eval_teacher {type(exc).__name__}", time.time() - started)
        return group_summary
    write_json(group_dir / "teacher.json", {**teacher_rollout, "evaluation": teacher_eval})
    if teacher_rollout.get("usage"):
        group_usage["teacher"] = merge_usage(group_usage.get("teacher"), teacher_rollout["usage"])

    if skip_self_evolution_if_teacher_fails and not teacher_eval["success"]:
        group_summary["teacher_failed"] = True
        group_summary["skipped_self_evolution"] = True
        group_summary["skill_version_end"] = skill.version
        write_json(group_dir / "summary.json", group_summary)
        _log_task("evolve", adapter.name, task_index + 1, total_tasks, group_summary["task_id"],
                  "FAIL", f"teacher-also-failed v{skill_v_start}", time.time() - started)
        return group_summary

    history_trajectories.append(
        _trajectory_for_critic("teacher", teacher_rollout, teacher_eval)
    )

    group_dir_rel = f"evolution/groups/{group_dir.name}"

    for round_idx in range(max_iters):
        round_dir = group_dir / f"round_{round_idx:02d}"
        round_dir.mkdir(parents=True, exist_ok=True)

        edit_log_payload = [record.to_dict() for record in skill.edit_log]
        critic_user = critic_initial_user(
            adapter.name,
            render_for_critic(parse_skill_rules(canonicalize_skill_text(task_start_skill))),
            edit_log_payload,
            task_text,
            history_trajectories,
            attachments=adapter.critic_attachments(task),
        )

        # Snapshot critic usage before the tool loop.
        critic_usage_before = critic.get_usage() if hasattr(critic, "get_usage") else None

        loop_result = run_tool_loop(
            critic,
            system=CRITIC_SYSTEM,
            initial_user=critic_user,
            tools=CRITIC_TOOLS,
            investigation_tools={"get_original_trajectories"},
            terminator_tool="submit_ops",
            investigation_handler=_build_trajectory_handler(out_dir, skill.edit_log),
            max_investigations=10,
            max_turns=12,
        )

        # Capture critic usage delta for this round.
        critic_round_usage = None
        if critic_usage_before is not None and hasattr(critic, "get_usage"):
            critic_round_usage = usage_delta(critic_usage_before, critic.get_usage())
            group_usage["critic"] = merge_usage(group_usage.get("critic"), critic_round_usage)

        ops_list: list[dict[str, Any]] = []
        if loop_result.final_tool_call is not None:
            raw_ops = loop_result.final_tool_call.get("arguments", {}).get("ops")
            if isinstance(raw_ops, list):
                ops_list = raw_ops

        ops_result = apply_skill_ops(task_start_skill, ops_list)
        candidate_skill = ops_result.text if ops_result.ok else task_start_skill

        critic_record: dict[str, Any] = {
            "round": round_idx,
            "messages": _slim_messages_for_log(loop_result.messages),
            "ops": ops_result.ops or [],
            "ops_valid": ops_result.ok,
            "ops_error": ops_result.error,
            "skill_before": task_start_skill,
            "candidate_skill": candidate_skill,
            "accepted": False,
            "investigations": loop_result.investigations,
            "loop_stats": loop_result.loop_stats,
            "loop_error": loop_result.error,
        }
        if critic_round_usage:
            critic_record["usage"] = critic_round_usage
        round_summary: dict[str, Any] = {
            "round": round_idx,
            "ops_valid": ops_result.ok,
            "ops_error": ops_result.error,
            "ops_committed": False,
            "skill_version_before_round": skill.version,
            "ops": ops_result.ops or [],
            "loop_stats": loop_result.loop_stats,
        }

        if ops_result.ok:
            write_text(round_dir / "candidate_skill.md", candidate_skill)
        else:
            write_text(round_dir / "rejected_skill.md", task_start_skill)

        # Critic actively decided no skill change is justified. The candidate
        # skill is identical to the one the student just failed on, so a retry
        # would be wasted; subsequent rounds would re-feed the same context
        # and almost certainly return the same {"ops":[]}. Skip both.
        if ops_result.ok and not ops_result.ops:
            round_summary["skipped_retry"] = "critic_returned_empty_ops"
            critic_record["skipped_retry"] = True
            group_summary["rounds"].append(round_summary)
            write_json(round_dir / "critic.json", critic_record)
            break

        try:
            retry = run_rollout(student, adapter, "student_retry", task, candidate_skill, output_dir=round_dir / "student_retry_outputs", gold_path=_gold_path_for_task(adapter, task))
        except Exception as exc:
            _finalize_failed_task(group_dir, group_summary, f"student_retry_round_{round_idx}", exc, skill.version)
            _log_task("evolve", adapter.name, task_index + 1, total_tasks, group_summary["task_id"],
                      "ERR", f"retry r{round_idx} {type(exc).__name__}", time.time() - started)
            return group_summary
        try:
            retry_eval = adapter.evaluate(
                task,
                retry["final_answer"],
                judge=judge,
                trajectory=retry,
                output_dir=round_dir / "student_retry_outputs",
            )
        except Exception as exc:
            _finalize_failed_task(group_dir, group_summary, f"eval_student_retry_round_{round_idx}", exc, skill.version)
            _log_task("evolve", adapter.name, task_index + 1, total_tasks, group_summary["task_id"],
                      "ERR", f"eval_retry r{round_idx} {type(exc).__name__}", time.time() - started)
            return group_summary
        write_json(round_dir / "student_retry.json", {**retry, "evaluation": retry_eval})
        if retry.get("usage"):
            group_usage["student"] = merge_usage(group_usage.get("student"), retry["usage"])

        round_summary["student_retry_success"] = retry_eval["success"]
        group_summary["rounds"].append(round_summary)
        latest_student = retry
        latest_eval = retry_eval

        history_trajectories.append(
            _trajectory_for_critic(f"student_retry_round_{round_idx}", retry, retry_eval)
        )

        if retry_eval["success"]:
            if ops_result.ok and ops_result.ops:
                skill.current = candidate_skill
                skill.version += 1
                round_summary["ops_committed"] = True
                round_summary["skill_version_after_commit"] = skill.version
                critic_record["accepted"] = True
                appended = skill.record_edits(
                    ops_result.ops,
                    rules_before=ops_result.rules_before,
                    rules_after=ops_result.rules_after,
                    group_index=task_index,
                    round_index=round_idx,
                    group_dir=group_dir_rel,
                )
                round_summary["edit_indexes"] = [r.edit_index for r in appended]
                skill_path = out_dir / "skills" / f"{skill.version:04d}_after_group_{task_index:04d}_round_{round_idx:02d}.md"
                skill.save_snapshot(skill_path)
                skill.save_snapshot(round_dir / "accepted_skill.md")
                skill.save_edit_log(out_dir / "evolution" / "edit_log.json")
            write_json(round_dir / "critic.json", critic_record)
            group_summary["success"] = True
            break

        write_json(round_dir / "critic.json", critic_record)

    group_summary["skill_version_end"] = skill.version
    if group_usage:
        group_summary["usage"] = group_usage
    write_json(group_dir / "summary.json", group_summary)
    rounds_run = len(group_summary.get("rounds") or [])
    if group_summary["success"]:
        skill_change = (
            f"v{skill_v_start}->v{skill.version}" if skill.version != skill_v_start else f"v{skill.version}"
        )
        detail = f"round {rounds_run} {skill_change}"
        status = "OK"
    else:
        detail = f"no-improvement v{skill.version}"
        status = "FAIL"
    _log_task("evolve", adapter.name, task_index + 1, total_tasks, group_summary["task_id"],
              status, detail, time.time() - started)
    return group_summary


def _build_trajectory_handler(out_dir: Path, edit_log):
    """Build a `get_original_trajectories` tool handler bound to this run.

    The handler maps `edit_index` to the originating group's
    `student_initial.json` and `teacher.json` via the EditRecord's
    `group_dir` pointer."""
    import json as _json

    def handler(name: str, args: dict[str, Any]) -> str:
        if name != "get_original_trajectories":
            return _json.dumps({"error": f"unknown tool {name!r}"})
        idx = args.get("edit_index")
        try:
            idx_int = int(idx)
        except (TypeError, ValueError):
            return _json.dumps({"error": f"edit_index must be an integer, got {idx!r}"})
        if idx_int < 0 or idx_int >= len(edit_log):
            return _json.dumps({
                "error": f"edit_index {idx_int} out of range (have {len(edit_log)})",
            })
        record = edit_log[idx_int]
        group_dir_abs = out_dir / record.group_dir
        result: dict[str, Any] = {
            "edit_index": idx_int,
            "group_index": record.group_index,
            "round_index": record.round_index,
            "op": record.op,
            "rule_id": record.rule_id,
        }
        student_path = group_dir_abs / "student_initial.json"
        teacher_path = group_dir_abs / "teacher.json"
        task_path = group_dir_abs / "task.json"
        if task_path.exists():
            try:
                task_obj = read_json(task_path)
                result["task_text"] = task_obj.get("_path") or task_obj.get("id") or ""
            except Exception:
                pass
        if student_path.exists():
            try:
                student_obj = read_json(student_path)
                result["student_trajectory"] = _summarize_trajectory(student_obj)
            except Exception as exc:
                result["student_trajectory_error"] = str(exc)
        if teacher_path.exists():
            try:
                teacher_obj = read_json(teacher_path)
                result["teacher_trajectory"] = _summarize_trajectory(teacher_obj)
            except Exception as exc:
                result["teacher_trajectory_error"] = str(exc)
        return _json.dumps(result, ensure_ascii=False)

    return handler


def _summarize_trajectory(traj: dict[str, Any]) -> dict[str, Any]:
    """Trim a persisted rollout record (student_initial.json / teacher.json)
    into the same compact shape the in-loop history uses: the skill the
    agent saw + full model_output (raw reply, thinking included) + slimmed
    evaluation. Drops verbose fields (prompt, env_steps) the critic does
    not need. Falls back to final_answer for older records that didn't
    persist the raw reply."""
    out: dict[str, Any] = {
        "skill_used": traj.get("skill_used", "") or "[None]",
    }
    raw = traj.get("raw") or traj.get("final_answer", "")
    if raw:
        s = str(raw)
        out["model_output"] = s if len(s) <= 6000 else s[:6000] + "…"
    ev = traj.get("evaluation") or {}
    if ev:
        out["evaluation"] = _slim_eval_for_critic(ev)
    return out


def _run_one_eval_task(
    idx: int,
    task: dict[str, Any],
    adapter: BenchmarkAdapter,
    split: str,
    split_dir: Path,
    skill: SkillState,
    student: Any,
    judge: Any,
    total: int,
    resume: bool,
) -> dict[str, Any]:
    """Evaluate a single eval task. Always returns a record dict — never
    raises — so a ThreadPoolExecutor wrapper can collect results uniformly.
    Resume-skip and failure-marker logic mirrors the original sequential
    loop exactly, just hoisted into a callable.
    """
    task_output_dir = split_dir / f"{idx:04d}_{safe_id(task)}_outputs"
    task_record_path = split_dir / f"{idx:04d}_{safe_id(task)}.json"
    if resume and task_record_path.exists():
        try:
            record = read_json(task_record_path)
        except Exception:
            record = None
        # Resume policy: skip records that finished cleanly (success or
        # honest model-failure), but re-run records marked as `failed`
        # (LLM call exhausted retries last time).
        if record is not None and not record.get("failed"):
            print(f"[resume] skip eval task {idx} ({safe_id(task)})", flush=True)
            return record
    started = time.time()
    try:
        rollout = run_rollout(student, adapter, "student_eval", task, skill.current, output_dir=task_output_dir)
        evaluation = adapter.evaluate(
            task,
            rollout["final_answer"],
            judge=judge,
            trajectory=rollout,
            output_dir=task_output_dir,
        )
        record = {
            "index": idx,
            "task_id": task.get("id") or task.get("_path"),
            # Persist the original task so saved eval trajectories keep the
            # benchmark context (e.g. docvqa image_path). Evolution groups
            # already save task.json next to each group.
            "task": task,
            **rollout,
            "evaluation": evaluation,
        }
    except Exception as exc:
        # The LLM call (or, less commonly, adapter.evaluate) raised after
        # SDK retries were exhausted. Record this as `failed=true` so
        # resume re-runs it instead of skipping past a hole in the data.
        err_msg = f"{type(exc).__name__}: {exc}"
        print(f"[eval-exception] task {idx} ({safe_id(task)}): {err_msg}", flush=True)
        record = {
            "index": idx,
            "task_id": task.get("id") or task.get("_path"),
            "task": task,
            "failed": True,
            "failure_reason": "llm_error",
            "error": err_msg,
            "evaluation": {
                "success": False,
                "mode": "infra_error",
                "error": err_msg,
            },
        }
    write_json(task_record_path, record)
    elapsed = time.time() - started
    if record.get("failed"):
        status, detail = "ERR", f"{record.get('error', 'unknown')[:40]}"
    else:
        ev = record.get("evaluation", {})
        status = "OK" if ev.get("success") else "FAIL"
        extras = []
        for k in ("anls_mean", "f1", "anls", "sub_em"):
            if k in ev and isinstance(ev[k], (int, float)):
                extras.append(f"{k}={ev[k]:.3f}")
                break
        detail = " ".join(extras) if extras else ev.get("mode", "")
    _log_task(f"eval/{split}", adapter.name, idx + 1, total, record["task_id"],
              status, detail, elapsed)
    return record


def evaluate_split(
    adapter: BenchmarkAdapter,
    split: str,
    tasks: list[dict[str, Any]],
    skill: SkillState,
    student: Any,
    judge: Any,
    out_dir: Path,
    resume: bool = False,
) -> dict[str, Any]:
    split_dir = out_dir / "evaluation" / split
    split_dir.mkdir(parents=True, exist_ok=True)
    total = len(tasks)

    # Per-split worker count. Default 8 = parallel by default; set
    # PACT_EVAL_WORKERS=1 to fall back to fully serial. ALFWorld is
    # force-serialized regardless because `textworld.gym.register_games`
    # (alfworld_official.py:43-72) mutates a global Gym registry and is
    # not thread-safe.
    workers_env = max(1, int(os.environ.get("PACT_EVAL_WORKERS", "8")))
    workers = min(workers_env, total) if total else 1
    if adapter.name == "alfworld" and workers > 1:
        print(
            f"[eval/{split} alfworld] forcing workers=1 (Gym registry not thread-safe), "
            f"requested {workers_env}",
            flush=True,
        )
        workers = 1

    rows_by_idx: dict[int, dict[str, Any]] = {}
    if workers <= 1:
        for idx, task in enumerate(tasks):
            rows_by_idx[idx] = _run_one_eval_task(
                idx, task, adapter, split, split_dir, skill, student, judge, total, resume,
            )
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        print(
            f"[eval/{split} {adapter.name}] dispatching {total} tasks across {workers} workers",
            flush=True,
        )
        with ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix=f"eval-{adapter.name}-{split}"
        ) as pool:
            futures = {
                pool.submit(
                    _run_one_eval_task,
                    idx, task, adapter, split, split_dir, skill, student, judge, total, resume,
                ): idx
                for idx, task in enumerate(tasks)
            }
            for fut in as_completed(futures):
                idx = futures[fut]
                rows_by_idx[idx] = fut.result()

    # Re-order by idx so downstream metric aggregation matches the
    # sequential version byte-for-byte.
    rows = [rows_by_idx[i] for i in range(total)]

    success_count = sum(1 for row in rows if row["evaluation"]["success"])
    soft_vals = [
        row["evaluation"].get("soft_restriction", float(row["evaluation"]["success"]))
        for row in rows
    ]
    hard_vals = [
        row["evaluation"].get("hard_restriction", int(row["evaluation"]["success"]))
        for row in rows
    ]
    metrics = {
        "benchmark": adapter.name,
        "split": split,
        "num_tasks": len(rows),
        "success_count": success_count,
        "success_rate": success_count / len(rows) if rows else 0.0,
        "soft_restriction_mean": sum(soft_vals) / len(soft_vals) if soft_vals else 0.0,
        "hard_restriction_mean": sum(hard_vals) / len(hard_vals) if hard_vals else 0.0,
        "evaluation_modes": sorted({row["evaluation"]["mode"] for row in rows}) if rows else [],
    }
    # Mean-aggregate any per-task numeric metric the adapter emits (f1, anls,
    # sub_em, …) so the board can show softer scores beyond pure success_rate.
    # Skips fields already accounted for above and any non-numeric values.
    _EXCLUDE = {"success", "soft_restriction", "hard_restriction"}
    extra_keys: dict[str, list[float]] = {}
    for row in rows:
        ev = row.get("evaluation", {})
        for k, v in ev.items():
            if k in _EXCLUDE:
                continue
            if isinstance(v, bool):
                v = float(v)
            elif not isinstance(v, (int, float)):
                continue
            extra_keys.setdefault(k, []).append(float(v))
    for k, vals in extra_keys.items():
        if k not in metrics:  # don't clobber the canonical aggregates above
            metrics[f"{k}_mean"] = sum(vals) / len(vals) if vals else 0.0
    # Aggregate eval usage across all tasks.
    eval_usage: dict[str, Any] | None = None
    for row in rows:
        task_usage = row.get("usage")
        if task_usage:
            eval_usage = merge_usage(eval_usage, task_usage)
    if eval_usage:
        metrics["usage"] = eval_usage
    write_json(split_dir / "metrics.json", metrics)
    return metrics


def _load_resumed_skill(
    bench_dir: Path, initial_skill_path: Path
) -> tuple[SkillState, set[str], list[dict[str, Any]]]:
    """Load skill state and completed groups from a previous run."""
    skill = SkillState.from_path(initial_skill_path)
    completed_groups: set[str] = set()
    existing_summaries: list[dict[str, Any]] = []

    if not bench_dir.exists():
        return skill, completed_groups, existing_summaries

    bench_summary_path = bench_dir / "evolution" / "summary.json"
    if bench_summary_path.exists():
        try:
            existing_summaries = read_json(bench_summary_path)
            if not isinstance(existing_summaries, list):
                existing_summaries = []
        except Exception:
            pass

    groups_dir = bench_dir / "evolution" / "groups"
    failed_task_indexes: set[int] = set()
    if groups_dir.exists():
        for gdir in groups_dir.iterdir():
            if not gdir.is_dir():
                continue
            group_summary_path = gdir / "summary.json"
            if not group_summary_path.exists():
                continue
            # Only count cleanly-finished groups as "done". Groups marked
            # `failed=true` (LLM call exhausted retries last time) are NOT
            # added to `completed_groups`, so the outer loop re-runs them.
            try:
                summary = read_json(group_summary_path)
            except Exception:
                summary = None
            if isinstance(summary, dict) and summary.get("failed"):
                ti = summary.get("task_index")
                if isinstance(ti, int):
                    failed_task_indexes.add(ti)
                print(f"[resume] retrying previously-failed task: {gdir.name}", flush=True)
                continue
            completed_groups.add(gdir.name)

    # Drop stale entries from the bench-level summary list so the retry
    # path doesn't end up with two records (old failed + new) for the
    # same task_index after the bench-level summary is re-written.
    if failed_task_indexes:
        existing_summaries = [
            s for s in existing_summaries
            if not (isinstance(s, dict) and s.get("task_index") in failed_task_indexes)
        ]

    skills_dir = bench_dir / "skills"
    if skills_dir.exists():
        snapshots = sorted(skills_dir.glob("[0-9][0-9][0-9][0-9]_after_group_*.md"))
        if snapshots:
            latest = snapshots[-1]
            skill = SkillState.from_path(latest)
            try:
                skill.version = int(latest.name.split("_")[0])
            except ValueError:
                skill.version = len(snapshots)
        elif (skills_dir / "final.md").exists():
            skill = SkillState.from_path(skills_dir / "final.md")
            skill.version = len(list(skills_dir.glob("[0-9][0-9][0-9][0-9]_after_group_*.md")))

    # Restore the edit log if one was persisted.
    edit_log_path = bench_dir / "evolution" / "edit_log.json"
    try:
        skill.load_edit_log(edit_log_path)
    except Exception as exc:
        print(f"[resume] warning: could not load edit_log.json: {exc}")

    return skill, completed_groups, existing_summaries


def run_benchmark(
    adapter: BenchmarkAdapter,
    args: argparse.Namespace,
    student: Any,
    teacher: Any,
    critic: Any,
    judge: Any,
    run_dir: Path,
    resume: bool = False,
) -> list[dict[str, Any]]:
    bench_dir = run_dir / adapter.name
    mode = args.mode

    if mode == "test" and args.skill:
        skill_path = Path(args.skill)
        if not skill_path.is_absolute():
            skill_path = ROOT / skill_path
        # If skill points to a run directory, resolve per-benchmark final.md
        if skill_path.is_dir():
            skill_path = skill_path / adapter.name / "skills" / "final.md"
        if skill_path.exists():
            skill = SkillState.from_path(skill_path)
            print(f"[test mode] loaded skill from {skill_path}")
        else:
            print(f"[warning] skill file not found: {skill_path}, using initial skill")
            skill = SkillState.from_path(adapter.initial_skill_path)
        completed_groups = set()
        train_summaries = []
    elif resume and bench_dir.exists():
        skill, completed_groups, train_summaries = _load_resumed_skill(
            bench_dir, adapter.initial_skill_path
        )
        print(
            f"[resume] {adapter.name}: {len(completed_groups)} tasks done, "
            f"skill v{skill.version}"
        )
    else:
        skill = SkillState.from_path(adapter.initial_skill_path)
        completed_groups = set()
        train_summaries = []
        (bench_dir / "skills").mkdir(parents=True, exist_ok=True)
        skill.save_snapshot(bench_dir / "skills" / "0000_initial.md")

    if mode in ("evolve", "evolve+test"):
        train_tasks = adapter.load_tasks("train", args.train_limit)
        total_train = len(train_tasks)
        for idx, task in enumerate(train_tasks):
            group_id = f"{idx:04d}_{safe_id(task)}"
            if group_id in completed_groups:
                print(f"[resume] skip task {idx} ({group_id})")
                if not any(s.get("task_index") == idx for s in train_summaries):
                    try:
                        s = read_json(bench_dir / "evolution" / "groups" / group_id / "summary.json")
                        train_summaries.append(s)
                    except Exception:
                        pass
                continue

            train_summaries.append(
                evolve_on_task(
                    adapter,
                    task,
                    idx,
                    total_train,
                    skill,
                    student,
                    teacher,
                    critic,
                    judge,
                    bench_dir,
                    args.max_iters,
                    args.skip_self_evolution_if_teacher_fails,
                )
            )

        write_json(bench_dir / "evolution" / "summary.json", train_summaries)
        skill.save_snapshot(bench_dir / "skills" / "final.md")
        skill.save_edit_log(bench_dir / "evolution" / "edit_log.json")

    metrics = []
    if mode in ("test", "evolve+test"):
        # Which splits to evaluate is declared per-benchmark in yaml; defaults
        # to a single "test" split. ALFWorld sets [test_seen, test_unseen].
        eval_splits = adapter.config.get("evaluation_splits", ["test"])
        for split in eval_splits:
            if split not in adapter.config.get("splits", {}):
                continue
            split_metrics_path = bench_dir / "evaluation" / split / "metrics.json"
            if resume and split_metrics_path.exists():
                # Check if any individual task records are marked as failed —
                # if so, re-run the split (per-task resume will re-run only
                # the failed ones).
                split_dir = bench_dir / "evaluation" / split
                has_failed = False
                if split_dir.exists():
                    for f in split_dir.glob("*.json"):
                        if f.name == "metrics.json":
                            continue
                        try:
                            if read_json(f).get("failed"):
                                has_failed = True
                                break
                        except Exception:
                            pass
                if not has_failed:
                    print(f"[resume] skip {split} eval - already done")
                    try:
                        metrics.append(read_json(split_metrics_path))
                    except Exception:
                        pass
                    continue
                print(f"[resume] {split} eval has failed tasks - re-running")
            tasks = adapter.load_tasks(split, args.eval_limit)
            metrics.append(evaluate_split(adapter, split, tasks, skill, student, judge, bench_dir, resume=resume))

    # Aggregate per-benchmark usage from evolution groups and eval splits.
    bench_usage: dict[str, Any] = {}
    for gs in train_summaries:
        gu = gs.get("usage") or {}
        for role, role_usage in gu.items():
            if isinstance(role_usage, dict):
                bench_usage[role] = merge_usage(bench_usage.get(role), role_usage)
    for m in metrics:
        mu = m.get("usage")
        if mu:
            # Eval usage goes under "student" (eval always uses student model).
            bench_usage["student"] = merge_usage(bench_usage.get("student"), mu)
    # Compute totals.
    total_usage: dict[str, Any] | None = None
    for role_usage in bench_usage.values():
        if isinstance(role_usage, dict):
            total_usage = merge_usage(total_usage, role_usage)
    if total_usage:
        bench_usage["total"] = total_usage
    if bench_usage:
        write_json(bench_dir / "usage.json", bench_usage)

    write_json(
        bench_dir / "benchmark_summary.json",
        {
            "benchmark": adapter.name,
            "train_groups": len(train_summaries),
            "train_success_count": sum(1 for item in train_summaries if item["success"]),
            "final_skill_version": skill.version,
            "metrics": metrics,
        },
    )
    return metrics, bench_usage


def write_summary_csv(run_dir: Path, metrics: list[dict[str, Any]]) -> None:
    path = run_dir / "summary.csv"
    # SkillOpt-aligned benchmarks emit extra _mean aggregates from
    # evaluate_split (f1_mean / anls_mean / sub_em_mean). Surface them as
    # CSV columns so the frontend can pick them up without reading per-task
    # metrics.json. extrasaction="ignore" keeps the writer forgiving.
    fieldnames = [
        "benchmark",
        "split",
        "target_benchmark",
        "target_split",
        "num_tasks",
        "success_count",
        "success_rate",
        "soft_restriction_mean",
        "hard_restriction_mean",
        "f1_mean",
        "anls_mean",
        "sub_em_mean",
        "evaluation_modes",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in metrics:
            writer.writerow(row)


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime())


def start_run_timer(run_dir: Path) -> dict[str, Any]:
    """Open a timing segment for this run. Appends to ``timing.json`` if it
    already exists (resume), so total elapsed accumulates across restarts.
    Any previously-unclosed segment is closed with ``elapsed_seconds = 0``
    so the file stays well-formed after a crash.
    """
    timing_path = run_dir / "timing.json"
    existing: dict[str, Any] = {}
    if timing_path.exists():
        try:
            loaded = read_json(timing_path)
            if isinstance(loaded, dict):
                existing = loaded
        except Exception:
            pass
    segments = list(existing.get("segments") or [])
    for seg in segments:
        if seg.get("ended_at") is None:
            seg["ended_at"] = seg.get("started_at")
            seg["elapsed_seconds"] = float(seg.get("elapsed_seconds") or 0.0)
            seg["crashed"] = True
    started_iso = _now_iso()
    started_ts = time.time()
    segments.append({"started_at": started_iso, "ended_at": None, "elapsed_seconds": 0.0})
    timing = {
        "started_at": existing.get("started_at") or started_iso,
        "ended_at": None,
        "elapsed_seconds": round(sum(float(s.get("elapsed_seconds") or 0.0) for s in segments), 3),
        "segments": segments,
    }
    write_json(timing_path, timing)
    return {"path": timing_path, "started_ts": started_ts}


def end_run_timer(timer: dict[str, Any]) -> None:
    """Close the open segment opened by ``start_run_timer`` and refresh
    the top-level ``ended_at`` / ``elapsed_seconds`` aggregates."""
    timing_path: Path = timer["path"]
    try:
        timing = read_json(timing_path)
    except Exception:
        return
    if not isinstance(timing, dict):
        return
    ended_iso = _now_iso()
    elapsed = round(time.time() - timer["started_ts"], 3)
    segments = timing.get("segments") or []
    if segments:
        last = segments[-1]
        last["ended_at"] = ended_iso
        last["elapsed_seconds"] = elapsed
    timing["ended_at"] = ended_iso
    timing["elapsed_seconds"] = round(
        sum(float(s.get("elapsed_seconds") or 0.0) for s in segments), 3
    )
    write_json(timing_path, timing)


def safe_id(task: dict[str, Any]) -> str:
    raw = str(task.get("id") or task.get("_path") or "task")
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in raw)[-80:]


def make_llm_clients(args: argparse.Namespace) -> tuple[Any, Any | None, Any | None]:
    base_url = (
        args.base_url
        or os.environ.get("PACT_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("DASHSCOPE_BASE_URL")
    )
    student_url = args.student_base_url or base_url
    teacher_url = args.teacher_base_url or base_url
    api_env = args.api_key_env
    student = LLMClient(LLMConfig(args.student_model, student_url, api_env))
    if not args.teacher_model:
        return student, None, None
    teacher = LLMClient(LLMConfig(args.teacher_model, teacher_url, api_env))
    return student, teacher, teacher


_SAFE_RUN_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def sanitize_run_name(name: str) -> str:
    """Make `name` safe for use as a directory segment: keep [A-Za-z0-9._-],
    collapse runs of other chars to '_', trim leading/trailing '_'."""
    cleaned = _SAFE_RUN_NAME.sub("_", name).strip("_")
    return cleaned or "run"


def main() -> None:
    load_env_file()
    parser = argparse.ArgumentParser(description="Run SKILL-KD skill evolution experiments.")
    parser.add_argument("--mode", choices=["evolve", "test", "evolve+test"], default="evolve+test",
                        help="Run mode: evolve (train only), test (eval only), or evolve+test (default).")
    parser.add_argument("--skill", default=None,
                        help="Path to skill.md for test mode, or initial skill for evolve mode.")
    parser.add_argument("--exp-name", default=None,
                        help="Human-readable experiment name (e.g. 'main_ablation_1').")
    parser.add_argument("--benchmarks", nargs="+", default=None)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--eval-limit", type=int, default=None)
    parser.add_argument("--max-iters", type=int, default=int(os.environ.get("PACT_MAX_ITERS", "3")))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--student-model", default="qwen3-8b")
    parser.add_argument("--teacher-model", default=None,
                        help="Teacher model. Required for evolve/evolve+test; ignored in test mode.")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--student-base-url", default=None,
                        help="Base URL for student model. Falls back to --base-url if not set.")
    parser.add_argument("--teacher-base-url", default=None,
                        help="Base URL for teacher/critic model. Falls back to --base-url if not set.")
    parser.add_argument("--api-key-env", default=os.environ.get("PACT_API_KEY_ENV", "DASHSCOPE_API_KEY"))
    parser.add_argument(
        "--skip-self-evolution-if-teacher-fails",
        action="store_true",
        default=None,
        help="If set, stop a training task's evolution loop when the teacher also fails.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previous run results if they exist.",
    )
    args = parser.parse_args()

    if args.skip_self_evolution_if_teacher_fails is None:
        args.skip_self_evolution_if_teacher_fails = False

    if args.benchmarks is None:
        args.benchmarks = ["spreadsheetbench"]

    if args.mode != "test" and not args.teacher_model:
        parser.error(f"--teacher-model is required for mode={args.mode}")

    ts = time.strftime("%Y%m%d_%H%M%S")
    if args.exp_name:
        prefix = f"{sanitize_run_name(args.exp_name)}_"
    elif args.student_model:
        prefix = f"{sanitize_run_name(args.student_model)}_"
    else:
        prefix = "pact_"
    default_run_id = f"{prefix}{ts}"
    run_id = args.run_id or default_run_id

    result_dir = ROOT / "results"
    if args.resume and args.run_id is None:
        if result_dir.exists():
            candidates = sorted(
                [d.name for d in result_dir.iterdir() if d.is_dir() and d.name.startswith(prefix)],
                reverse=True,
            )
            if candidates:
                run_id = candidates[0]
                print(f"[resume] auto-matched previous run: {run_id}")
            else:
                print(f"[resume] no previous run found for prefix '{prefix}', starting fresh.")

    run_dir = result_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    config_to_save = vars(args).copy()
    config_to_save["exp_name"] = args.exp_name
    write_json(run_dir / "run_config.json", config_to_save)

    timer = start_run_timer(run_dir)
    try:
        student, teacher, critic = make_llm_clients(args)
        all_metrics: list[dict[str, Any]] = []
        run_usage: dict[str, Any] = {}
        for name in args.benchmarks:
            config_path = ROOT / "configs" / "benchmarks" / f"{name}.yaml"
            adapter = load_adapter(config_path)
            bench_metrics, bench_usage = run_benchmark(
                adapter, args, student, teacher, critic, None, run_dir, resume=args.resume
            )
            all_metrics.extend(bench_metrics)
            # Aggregate bench usage into run-level usage.
            for role, role_usage in bench_usage.items():
                if isinstance(role_usage, dict):
                    run_usage[role] = merge_usage(run_usage.get(role), role_usage)
            # Rewrite the run-level summary after every benchmark so partial
            # multi-benchmark runs are inspectable before the full run finishes.
            write_summary_csv(run_dir, all_metrics)
            write_json(run_dir / "summary.json", all_metrics)
        # Write run-level usage.json.
        if run_usage:
            # Recompute total from individual roles (excluding "total" key from bench).
            total: dict[str, Any] | None = None
            for role, role_usage in run_usage.items():
                if role != "total" and isinstance(role_usage, dict):
                    total = merge_usage(total, role_usage)
            if total:
                run_usage["total"] = total
            write_json(run_dir / "usage.json", run_usage)
    finally:
        end_run_timer(timer)
    elapsed = read_json(timer["path"]).get("elapsed_seconds", 0.0)
    print(f"SKILL-KD run finished: {run_dir} (elapsed {elapsed:.1f}s)")


if __name__ == "__main__":
    main()
