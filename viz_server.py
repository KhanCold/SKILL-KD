#!/usr/bin/env python3
"""PACT Results Board — single-file visualization server.

Usage:
    python viz_server.py          # serves on http://localhost:8080
    python viz_server.py 9000     # serves on custom port
"""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs

from src.pact.pricing import compute_run_cost

# Matches the `_YYYYMMDD_HHMMSS` suffix that run_experiment.py auto-appends
# to every run_id, e.g. `3-4b-e50_20260531_195441`. We use this to recover
# a start timestamp for legacy runs that pre-date timing.json.
_RUN_ID_TS_RE = re.compile(r"_(\d{8})_(\d{6})$")


def started_from_run_id(run_id: str) -> str | None:
    m = _RUN_ID_TS_RE.search(run_id)
    if not m:
        return None
    try:
        dt = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
    except ValueError:
        return None
    return dt.isoformat(timespec="seconds")

RESULTS_DIR = Path("results")
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8080

# Repo root for resolving relative image paths from saved trajectories.
_REPO_ROOT = Path(__file__).resolve().parent
# Whitelist for the /image route. Add additional dataset image roots here.
# Each entry must be an already-resolved absolute path; the handler refuses
# requests whose resolved target is not contained in one of these roots.
IMAGE_ALLOWED_ROOTS = [
    (_REPO_ROOT / "data" / "docvqa_images").resolve(),
]
_IMAGE_CONTENT_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
}


def _path_is_within(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False

# Git LFS pointer files start with this magic line. Detect them so the board can
# surface a clear warning instead of silently rendering an empty row.
_LFS_MAGIC = b"version https://git-lfs.github.com/spec/v1"


def is_lfs_pointer(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(len(_LFS_MAGIC)) == _LFS_MAGIC
    except Exception:
        return False


def _load_json_or_lfs(path: Path):
    """Return (value, is_lfs). value is None on missing/corrupt; is_lfs flags an LFS pointer."""
    if not path.exists():
        return None, False
    if is_lfs_pointer(path):
        return None, True
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f), False
    except Exception:
        return None, False


# ═══════════════════════════════════════════════════════════════════════════════
# Data scanning
# ═══════════════════════════════════════════════════════════════════════════════

# Lazy lookup: questionId -> image_path for docvqa, so legacy eval records
# that didn't persist the original task can still render the document image.
# Cached for the lifetime of the process; if the underlying jsonl is updated
# the user just restarts the viz server.
_DOCVQA_ID_TO_IMAGE: dict | None = None


def _load_docvqa_id_to_image() -> dict:
    global _DOCVQA_ID_TO_IMAGE
    if _DOCVQA_ID_TO_IMAGE is not None:
        return _DOCVQA_ID_TO_IMAGE
    mapping: dict = {}
    docvqa_dir = _REPO_ROOT / "data" / "docvqa"
    for name in ("test.jsonl", "val.jsonl", "train.jsonl"):
        path = docvqa_dir / name
        if not path.exists():
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    img = obj.get("image_path")
                    if not img:
                        continue
                    for key in (obj.get("id"), obj.get("questionId")):
                        if key is None:
                            continue
                        # Index under both the native type and its string form
                        # because task_id can land on disk as either.
                        mapping[key] = img
                        mapping[str(key)] = img
        except OSError:
            continue
    _DOCVQA_ID_TO_IMAGE = mapping
    return mapping


def _slim_evaluation(evaluation):
    if not isinstance(evaluation, dict):
        return {}
    keep = (
        "success",
        "num_actions",
        "answer_match",
        "exact_match",
        "f1",
        "anls",
        "hard_restriction",
        "soft_restriction",
        "error",
    )
    return {k: evaluation[k] for k in keep if k in evaluation}


def _slim_trajectory(record):
    if not isinstance(record, dict):
        return {}
    slim = {}
    if "task_id" in record:
        slim["task_id"] = record["task_id"]
    if "index" in record:
        slim["index"] = record["index"]
    if "evaluation" in record:
        slim["evaluation"] = _slim_evaluation(record.get("evaluation"))
    return slim


def _slim_critic(record):
    if not isinstance(record, dict):
        return {}
    slim = {}
    for key in ("accepted", "ops_valid"):
        if key in record:
            slim[key] = record[key]
    if "loop_stats" in record and isinstance(record["loop_stats"], dict):
        stats = record["loop_stats"]
        slim["loop_stats"] = {
            k: stats[k]
            for k in ("investigation_calls", "forced_terminate")
            if k in stats
        }
    return slim


def scan_results():
    runs = []
    if not RESULTS_DIR.exists():
        return runs
    for run_dir in sorted(RESULTS_DIR.iterdir()):
        if not run_dir.is_dir():
            continue
        run = {"id": run_dir.name, "benchmarks": [], "lfs_pending": False}

        cfg, cfg_lfs = _load_json_or_lfs(run_dir / "run_config.json")
        if cfg_lfs:
            run["lfs_pending"] = True
        if cfg:
            run["student_model"] = cfg.get("student_model", "")
            run["teacher_model"] = cfg.get("teacher_model", "") or ""
            run["benchmarks_list"] = cfg.get("benchmarks", [])
            run["exp_name"] = cfg.get("exp_name", "")
            run["mode"] = cfg.get("mode", "evolve+test")

        summary, summary_lfs = _load_json_or_lfs(run_dir / "summary.json")
        if summary_lfs:
            run["lfs_pending"] = True
        run["summary"] = summary or []

        timing, timing_lfs = _load_json_or_lfs(run_dir / "timing.json")
        if timing_lfs:
            run["lfs_pending"] = True
        timing = timing or {}
        # Backfill started_at from the run_id timestamp suffix for legacy
        # runs that ran before timing.json existed.
        if not timing.get("started_at"):
            fallback = started_from_run_id(run_dir.name)
            if fallback:
                timing["started_at"] = fallback
                timing["started_at_source"] = "run_id"
        run["timing"] = timing

        usage, usage_lfs = _load_json_or_lfs(run_dir / "usage.json")
        if usage_lfs:
            run["lfs_pending"] = True
        run["usage"] = usage
        if usage:
            run["cost"] = compute_run_cost(usage)
        else:
            run["cost"] = None

        for sub in run_dir.iterdir():
            if not sub.is_dir():
                continue
            if (sub / "evolution").exists() or (sub / "evaluation").exists():
                bench = scan_benchmark(sub, detail=False)
                if bench:
                    if bench.get("lfs_pending"):
                        run["lfs_pending"] = True
                    run["benchmarks"].append(bench)
        # Keep the run visible even if every benchmark sub-tree is still an LFS
        # pointer — otherwise the user gets no clue why their run vanished.
        # Also keep runs that only have a top-level summary.json (e.g. skillopt
        # paper imports) — they have no per-benchmark sub-trees but carry valid
        # aggregate metrics that the board can render.
        if run["benchmarks"] or run["lfs_pending"] or run.get("summary"):
            runs.append(run)
    return runs


def scan_benchmark(bench_dir: Path, detail: bool = True):
    bench = {"name": bench_dir.name, "dir": str(bench_dir), "lfs_pending": False}

    evo_summary_data, evo_lfs = _load_json_or_lfs(bench_dir / "evolution" / "summary.json")
    if evo_lfs:
        bench["lfs_pending"] = True
    bench["evolution_summary"] = evo_summary_data or []

    groups_dir = bench_dir / "evolution" / "groups"
    groups = []
    if groups_dir.exists():
        for group_dir in sorted(groups_dir.iterdir()):
            if not group_dir.is_dir():
                continue
            group = scan_group_detail(group_dir) if detail else scan_group_summary(group_dir)
            if group:
                groups.append(group)
    bench["groups"] = groups

    bench["evaluation"] = {}
    eval_root = bench_dir / "evaluation"
    if eval_root.exists():
        for split_dir in sorted(eval_root.iterdir()):
            if not split_dir.is_dir():
                continue
            split_info = {"metrics": None, "tasks": []}
            metrics_data, metrics_lfs = _load_json_or_lfs(split_dir / "metrics.json")
            if metrics_lfs:
                bench["lfs_pending"] = True
            if metrics_data is not None:
                split_info["metrics"] = metrics_data
            for task_file in sorted(split_dir.glob("*.json")):
                if task_file.name == "metrics.json":
                    continue
                try:
                    with open(task_file, "r", encoding="utf-8") as f:
                        full_record = json.load(f)
                    record = full_record if detail else _slim_trajectory(full_record)
                    # Derive a slim task_info from the persisted original task
                    # so the viz renderer can show multimodal context (e.g.
                    # docvqa image_path).
                    if isinstance(full_record.get("task"), dict):
                        record["task_info"] = extract_task_info(full_record["task"])
                    elif bench_dir.name == "docvqa":
                        # Legacy docvqa eval records (pre-fix) didn't persist
                        # `task`; recover image_path via questionId lookup.
                        img = _load_docvqa_id_to_image().get(full_record.get("task_id"))
                        if img:
                            record["task_info"] = {"type": "docvqa", "image_path": img}
                    split_info["tasks"].append(record)
                except Exception:
                    pass
            if split_info["metrics"] or split_info["tasks"]:
                bench["evaluation"][split_dir.name] = split_info

    skills_dir = bench_dir / "skills"
    bench["skills"] = []
    if skills_dir.exists():
        for skill_file in sorted(skills_dir.glob("*.md")):
            try:
                with open(skill_file, "r", encoding="utf-8") as f:
                    bench["skills"].append({"name": skill_file.name, "content": f.read()})
            except Exception:
                pass

    edit_log_file = bench_dir / "evolution" / "edit_log.json"
    bench["edit_log"] = []
    if edit_log_file.exists():
        try:
            with open(edit_log_file, "r", encoding="utf-8") as f:
                bench["edit_log"] = json.load(f)
        except Exception:
            pass

    return bench


def _scan_group_base(group_dir: Path):
    group = {"id": group_dir.name}

    task_file = group_dir / "task.json"
    if task_file.exists():
        try:
            with open(task_file, "r", encoding="utf-8") as f:
                group["task"] = extract_task_info(json.load(f))
        except Exception:
            group["task"] = {}
    else:
        return None

    summary_file = group_dir / "summary.json"
    if summary_file.exists():
        try:
            with open(summary_file, "r", encoding="utf-8") as f:
                group["summary"] = json.load(f)
        except Exception:
            group["summary"] = {}
    return group


def scan_group_summary(group_dir: Path):
    group = _scan_group_base(group_dir)
    if group is None:
        return None

    student_file = group_dir / "student_initial.json"
    if student_file.exists():
        try:
            with open(student_file, "r", encoding="utf-8") as f:
                group["student_initial"] = _slim_trajectory(json.load(f))
        except Exception:
            pass

    rounds = []
    round_dirs = sorted([d for d in group_dir.iterdir() if d.is_dir() and d.name.startswith("round_")])
    for round_dir in round_dirs:
        rd = scan_round_summary(round_dir)
        if rd:
            rounds.append(rd)
    group["rounds"] = rounds
    return group


def scan_group_detail(group_dir: Path):
    group = _scan_group_base(group_dir)
    if group is None:
        return None

    student_file = group_dir / "student_initial.json"
    if student_file.exists():
        try:
            with open(student_file, "r", encoding="utf-8") as f:
                group["student_initial"] = json.load(f)
        except Exception:
            pass

    teacher_file = group_dir / "teacher.json"
    if teacher_file.exists():
        try:
            with open(teacher_file, "r", encoding="utf-8") as f:
                group["teacher"] = json.load(f)
        except Exception:
            pass

    rounds = []
    round_dirs = sorted([d for d in group_dir.iterdir() if d.is_dir() and d.name.startswith("round_")])
    for round_dir in round_dirs:
        rd = scan_round_detail(round_dir)
        if rd:
            rounds.append(rd)
    group["rounds"] = rounds
    group["_detailLoaded"] = True
    return group


def extract_task_info(task_data):
    info = {}
    # Discriminator order matters: structurally-specific shapes first, then the
    # bare-question fallbacks (wikitq, searchqa) last. Each branch keeps the
    # row dict's foreign fields out of the info payload so the frontend renderer
    # decides what to show per type.
    if "task_type" in task_data and "turk_annotations" in task_data:
        info["type"] = "alfworld"
        info["task_type"] = task_data.get("task_type", "")
        anns = task_data.get("turk_annotations", {}).get("anns", [{}])
        info["goal"] = anns[0].get("task_desc", "") if anns else ""
        info["scene"] = task_data.get("scene", {}).get("floor_plan", "")
        info["pddl_target"] = task_data.get("pddl_params", {}).get("object_target", "")
        info["pddl_parent"] = task_data.get("pddl_params", {}).get("parent_target", "")
    elif "instruction" in task_data and "spreadsheet_path" in task_data:
        info["type"] = "spreadsheetbench"
        info["instruction"] = task_data.get("instruction", "")[:400]
        info["instruction_type"] = task_data.get("instruction_type", "")
        info["answer_position"] = task_data.get("answer_position", "")
        info["task_id"] = task_data.get("id", "")
    elif "image_path" in task_data or "questionId" in task_data:
        info["type"] = "docvqa"
        info["question"] = task_data.get("question", "")
        info["image_path"] = task_data.get("image_path", "")
        info["doc_id"] = task_data.get("docId", "")
        info["question_id"] = task_data.get("questionId", "")
        info["topic"] = task_data.get("task_type", "") or task_data.get("subtask", "")
    elif "choices" in task_data and "correct_choice" in task_data:
        info["type"] = "livemathc"
        info["question"] = task_data.get("question", "")
        info["month"] = task_data.get("month", "")
        info["theorem_type"] = task_data.get("theorem_type", []) or []
        info["correct_label"] = (task_data.get("correct_choice", {}) or {}).get("label", "")
    elif "context" in task_data and "question" in task_data:
        info["type"] = "searchqa"
        info["question"] = task_data.get("question", "")
        # context can be very long; keep a preview for the task summary view.
        ctx = task_data.get("context", "") or ""
        info["context_preview"] = ctx[:400]
    elif "question" in task_data and "table" in task_data:
        info["type"] = "wikitq"
        info["question"] = task_data.get("question", "")
        info["table_id"] = task_data.get("table_id", "")
    else:
        info["type"] = "unknown"
    return info


def scan_round_summary(round_dir: Path):
    rd = {"id": round_dir.name}
    critic_file = round_dir / "critic.json"
    if critic_file.exists():
        try:
            with open(critic_file, "r", encoding="utf-8") as f:
                rd["critic"] = _slim_critic(json.load(f))
        except Exception:
            pass
    return rd


def scan_round_detail(round_dir: Path):
    rd = {"id": round_dir.name}
    retry_file = round_dir / "student_retry.json"
    if retry_file.exists():
        try:
            with open(retry_file, "r", encoding="utf-8") as f:
                rd["student_retry"] = json.load(f)
        except Exception:
            pass
    critic_file = round_dir / "critic.json"
    if critic_file.exists():
        try:
            with open(critic_file, "r", encoding="utf-8") as f:
                rd["critic"] = json.load(f)
        except Exception:
            pass
    return rd


def _child_dir(parent: Path, name: str) -> Path | None:
    if not name or "/" in name or "\\" in name:
        return None
    path = parent / name
    try:
        resolved = path.resolve()
        parent_resolved = parent.resolve()
    except (OSError, RuntimeError):
        return None
    if not _path_is_within(resolved, parent_resolved):
        return None
    return path if path.is_dir() else None


def _run_bench_dir(run_id: str, bench_name: str) -> Path | None:
    run_dir = _child_dir(RESULTS_DIR, run_id)
    if run_dir is None:
        return None
    return _child_dir(run_dir, bench_name)


def scan_eval_task_detail(split_dir: Path, task_index: int, bench_name: str):
    task_files = [p for p in sorted(split_dir.glob("*.json")) if p.name != "metrics.json"]
    if task_index < 0 or task_index >= len(task_files):
        return None
    task_file = task_files[task_index]
    try:
        with open(task_file, "r", encoding="utf-8") as f:
            record = json.load(f)
    except Exception:
        return None
    if isinstance(record.get("task"), dict):
        record["task_info"] = extract_task_info(record["task"])
    elif bench_name == "docvqa":
        img = _load_docvqa_id_to_image().get(record.get("task_id"))
        if img:
            record["task_info"] = {"type": "docvqa", "image_path": img}
    record["_detailLoaded"] = True
    return record


# ═══════════════════════════════════════════════════════════════════════════════
# HTML Page (embedded)
# ═══════════════════════════════════════════════════════════════════════════════

HTML_PAGE = '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PACT Results Board</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500&family=Noto+Serif+SC:wght@400;600&display=swap');

:root {
  --bg: #ffffff;
  --fg: #111111;
  --fg-secondary: #888888;
  --line: #e0e0e0;
  --line-dark: #c0c0c0;
  --accent: #111111;
  --success: #1a7f37;
  --fail: #cf222e;
  --warn: #9a6700;
  --info: #0969da;
  --mono: 'JetBrains Mono', monospace;
  --serif: 'Noto Serif SC', Georgia, serif;
}

* { margin: 0; padding: 0; box-sizing: border-box; }

body {
  font-family: var(--serif);
  background: var(--bg);
  color: var(--fg);
  line-height: 1.6;
  -webkit-font-smoothing: antialiased;
}

.app {
  display: flex;
  height: 100vh;
  overflow: hidden;
}

.sidebar {
  width: 280px;
  border-right: 1px solid var(--line);
  display: flex;
  flex-direction: column;
  overflow: hidden;
  background: var(--bg);
  flex-shrink: 0;
}

.main {
  flex: 1;
  overflow-y: auto;
  padding: 40px 48px;
}

.sidebar-header {
  padding: 28px 24px 20px;
  border-bottom: 1px solid var(--line);
}

.sidebar-header h1 {
  font-family: var(--mono);
  font-size: 11px;
  font-weight: 500;
  letter-spacing: 0.15em;
  text-transform: uppercase;
  color: var(--fg-secondary);
  margin-bottom: 4px;
}

.sidebar-header .title {
  font-family: var(--serif);
  font-size: 22px;
  font-weight: 600;
  letter-spacing: -0.01em;
}

.run-list {
  flex: 1;
  overflow-y: auto;
  padding: 8px 0;
}

.run-item {
  padding: 10px 24px;
  cursor: pointer;
  border-left: 2px solid transparent;
  transition: all 0.15s ease;
  font-family: var(--mono);
  font-size: 11px;
  letter-spacing: 0.02em;
  color: var(--fg-secondary);
}

.run-item:hover {
  background: rgba(0,0,0,0.02);
  color: var(--fg);
}

.run-item.active {
  border-left-color: var(--accent);
  color: var(--fg);
  background: rgba(0,0,0,0.03);
}

.run-item .name {
  display: block;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.run-item .meta {
  font-size: 10px;
  color: var(--fg-secondary);
  margin-top: 2px;
}

.section-title {
  font-family: var(--mono);
  font-size: 10px;
  font-weight: 500;
  letter-spacing: 0.2em;
  text-transform: uppercase;
  color: var(--fg-secondary);
  margin-bottom: 20px;
  padding-bottom: 8px;
  border-bottom: 1px solid var(--line);
}

/* ── Metrics ── */
.metrics-row {
  display: flex;
  gap: 0;
  border: 1px solid var(--line);
  margin-bottom: -1px;
}
.metrics-row:last-child {
  margin-bottom: 40px;
}

.metric-cell {
  flex: 1;
  padding: 18px 20px;
  border-right: 1px solid var(--line);
  min-width: 140px;
}
.metric-cell:last-child { border-right: none; }

.metric-label {
  font-family: var(--mono);
  font-size: 9px;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--fg-secondary);
  margin-bottom: 6px;
}

.metric-value {
  font-family: var(--mono);
  font-size: 24px;
  font-weight: 300;
  letter-spacing: -0.02em;
}
.metric-value.success { color: var(--success); }
.metric-value.fail { color: var(--fail); }

.metric-sub {
  font-family: var(--mono);
  font-size: 11px;
  color: var(--fg-secondary);
  margin-top: 4px;
}

/* ── Evaluation Split Card ── */
.eval-card {
  flex: 1;
  padding: 18px 20px;
  border-right: 1px solid var(--line);
  min-width: 180px;
}
.eval-card:last-child { border-right: none; }

.eval-card-header {
  font-family: var(--mono);
  font-size: 9px;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--fg-secondary);
  margin-bottom: 10px;
}

.eval-card-main {
  display: flex;
  align-items: baseline;
  gap: 8px;
}

.eval-card-rate {
  font-family: var(--mono);
  font-size: 28px;
  font-weight: 300;
  letter-spacing: -0.02em;
}
.eval-card-rate.success { color: var(--success); }
.eval-card-rate.fail { color: var(--fail); }

.eval-card-count {
  font-family: var(--mono);
  font-size: 12px;
  color: var(--fg-secondary);
}

.eval-card-extra {
  font-family: var(--mono);
  font-size: 10px;
  color: var(--fg-secondary);
  margin-top: 6px;
}

/* ── Tabs ── */
.bench-tabs {
  display: flex;
  gap: 0;
  border-bottom: 1px solid var(--line);
  margin-bottom: 32px;
}

.bench-tab {
  padding: 10px 20px;
  font-family: var(--mono);
  font-size: 11px;
  letter-spacing: 0.05em;
  cursor: pointer;
  border-bottom: 2px solid transparent;
  margin-bottom: -1px;
  color: var(--fg-secondary);
  transition: all 0.15s;
}

.bench-tab:hover { color: var(--fg); }
.bench-tab.active {
  color: var(--fg);
  border-bottom-color: var(--accent);
}

/* ── Timeline ── */
.timeline {
  position: relative;
  padding-left: 24px;
}

.timeline::before {
  content: '';
  position: absolute;
  left: 6px;
  top: 8px;
  bottom: 8px;
  width: 1px;
  background: var(--line);
}

.group-item {
  position: relative;
  margin-bottom: 2px;
  cursor: pointer;
  transition: background 0.1s;
  padding: 2px 0;
}

.group-item:hover {
  background: rgba(0,0,0,0.015);
}

.group-item::before {
  content: '';
  position: absolute;
  left: -20px;
  top: 14px;
  width: 7px;
  height: 7px;
  border-radius: 50%;
  border: 1.5px solid var(--line-dark);
  background: var(--bg);
}

.group-item.success::before {
  border-color: var(--success);
  background: var(--success);
}

.group-item.fail::before {
  border-color: var(--fail);
}

.group-header {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 8px 0;
  font-family: var(--mono);
  font-size: 11px;
}

.group-idx {
  color: var(--fg-secondary);
  min-width: 28px;
}

.group-task {
  flex: 1;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  color: var(--fg);
}

.group-status {
  font-size: 9px;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  padding: 2px 6px;
  border: 1px solid var(--line);
}

.group-status.success {
  color: var(--success);
  border-color: var(--success);
}

.group-status.fail {
  color: var(--fail);
  border-color: var(--fail);
}

.group-rounds {
  color: var(--fg-secondary);
  font-size: 10px;
}

.group-detail {
  border: 1px solid var(--line);
  margin-top: 32px;
  padding: 32px;
}

.group-detail-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: 24px;
  padding-bottom: 16px;
  border-bottom: 1px solid var(--line);
}

.group-detail-title {
  font-family: var(--mono);
  font-size: 13px;
  font-weight: 500;
}

.round-flow {
  display: flex;
  flex-direction: column;
  gap: 0;
  margin-top: 24px;
}

.round-block {
  border: 1px solid var(--line);
  margin-bottom: -1px;
}

.round-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px 16px;
  background: rgba(0,0,0,0.015);
  border-bottom: 1px solid var(--line);
  font-family: var(--mono);
  font-size: 10px;
  letter-spacing: 0.1em;
  text-transform: uppercase;
}

.round-body {
  padding: 16px;
}

/* ── Collapsible ── */
.collapsible {
  border: 1px solid var(--line);
  margin-bottom: -1px;
  background: var(--bg);
}

.collapsible-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 10px 14px;
  background: rgba(0,0,0,0.015);
  cursor: pointer;
  user-select: none;
  font-family: var(--mono);
  font-size: 11px;
}

.collapsible-header:hover {
  background: rgba(0,0,0,0.03);
}

.collapsible-left {
  display: flex;
  align-items: center;
  gap: 8px;
}

.caret {
  color: var(--fg-secondary);
  font-size: 9px;
  width: 10px;
  display: inline-block;
}

.collapsible-title {
  font-weight: 500;
  letter-spacing: 0.05em;
}

.collapsible-right {
  display: flex;
  align-items: center;
  gap: 10px;
}

.collapsible-summary {
  font-family: var(--mono);
  font-size: 10px;
  color: var(--fg-secondary);
  letter-spacing: 0.04em;
}
.collapsible-summary.ok { color: var(--success); }
.collapsible-summary.bad { color: var(--fail); }

.collapsible-body {
  padding: 14px;
  border-top: 1px solid var(--line);
}

.collapsible .collapsible {
  margin: 0 0 -1px 0;
}

/* ── ALFWorld action table ── */
.env-steps-table {
  width: 100%;
  border-collapse: collapse;
  border: 1px solid var(--line);
  font-family: var(--mono);
  font-size: 11px;
}
.env-steps-table th {
  text-align: left;
  padding: 6px 10px;
  background: rgba(0,0,0,0.02);
  border-bottom: 1px solid var(--line);
  font-weight: 500;
  font-size: 9px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--fg-secondary);
}
.env-steps-table td {
  padding: 5px 10px;
  border-bottom: 1px solid var(--line);
  vertical-align: top;
}
.env-steps-table tr:last-child td { border-bottom: none; }
.env-steps-table td.idx { color: var(--fg-secondary); width: 32px; }
.env-steps-table td.action { font-weight: 500; }

/* ── Trajectory (expanded) ── */
.traj-container {
  font-family: var(--mono);
  font-size: 11px;
  line-height: 1.7;
}

.traj-summary-bar {
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 10px 14px;
  background: rgba(0,0,0,0.02);
  border: 1px solid var(--line);
  border-bottom: none;
  font-family: var(--mono);
  font-size: 10px;
}

.traj-summary-bar .result {
  font-weight: 500;
  font-size: 11px;
}
.traj-summary-bar .result.success { color: var(--success); }
.traj-summary-bar .result.fail { color: var(--fail); }
.traj-summary-bar .detail {
  color: var(--fg-secondary);
}

.traj-info-bar {
  border: 1px solid var(--line);
  border-top: none;
  padding: 12px 16px;
  background: rgba(0,0,0,0.01);
}

.traj-info-row {
  display: flex;
  gap: 12px;
  margin-bottom: 8px;
}
.traj-info-row:last-child { margin-bottom: 0; }

.traj-info-label {
  font-family: var(--mono);
  font-size: 9px;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--fg-secondary);
  min-width: 100px;
  flex-shrink: 0;
  padding-top: 2px;
}

.traj-info-value {
  font-family: var(--serif);
  font-size: 13px;
  line-height: 1.5;
  flex: 1;
  word-break: break-word;
}

.traj-table {
  width: 100%;
  border-collapse: collapse;
  border: 1px solid var(--line);
  font-family: var(--mono);
  font-size: 11px;
}

.traj-table th {
  text-align: left;
  padding: 8px 12px;
  background: rgba(0,0,0,0.02);
  border-bottom: 1px solid var(--line);
  font-weight: 500;
  font-size: 9px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--fg-secondary);
}

.traj-table td {
  padding: 6px 12px;
  border-bottom: 1px solid var(--line);
  vertical-align: top;
}

.traj-table tr:last-child td { border-bottom: none; }

.traj-table .idx { color: var(--fg-secondary); font-size: 10px; width: 40px; }
.traj-table .action { word-break: break-all; }
.traj-table .tag {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  font-size: 9px;
  font-weight: 500;
  padding: 2px 8px;
  border-radius: 3px;
  margin-right: 4px;
  min-width: 28px;
  letter-spacing: 0.04em;
}
.traj-table .tag.ok { background: #dafbe1; color: var(--success); border: 1px solid var(--success); }
.traj-table .tag.bad { background: #ffebe9; color: var(--fail); border: 1px solid var(--fail); }
.traj-table .tag.info { background: #ddf4ff; color: var(--info); border: 1px solid var(--info); }
.traj-table .tag.warn { background: #fff8c5; color: var(--warn); border: 1px solid var(--warn); }

/* ── SpreadsheetBench specific ── */
.case-grid {
  display: flex;
  gap: 6px;
  margin: 8px 0;
}
.case-dot {
  width: 22px;
  height: 22px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 10px;
  font-family: var(--mono);
  border: 1px solid var(--line);
}
.case-dot.ok { background: #dafbe1; color: var(--success); border-color: var(--success); }
.case-dot.fail { background: #ffebe9; color: var(--fail); border-color: var(--fail); }

.case-detail {
  margin-top: 12px;
  padding: 12px;
  border: 1px solid var(--line);
  background: rgba(0,0,0,0.01);
}
.case-detail-header {
  font-family: var(--mono);
  font-size: 9px;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--fg-secondary);
  margin-bottom: 8px;
}

/* ── WikiTQ specific ── */
.pred-gold-comp {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0;
  border: 1px solid var(--line);
  margin-top: 8px;
}
.pred-gold-comp > div {
  padding: 10px 14px;
}
.pred-gold-comp > div:first-child {
  border-right: 1px solid var(--line);
}
.pred-gold-comp .label {
  font-family: var(--mono);
  font-size: 9px;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--fg-secondary);
  margin-bottom: 4px;
}

/* ── Code blocks ── */
.patch-block {
  background: #fafafa;
  border: 1px solid var(--line);
  padding: 16px;
  font-family: var(--mono);
  font-size: 10px;
  line-height: 1.7;
  overflow-x: auto;
  white-space: pre;
  max-height: 300px;
  overflow-y: auto;
}

.patch-add { color: var(--success); }
.patch-del { color: var(--fail); }

.code-block {
  background: #fafafa;
  border: 1px solid var(--line);
  padding: 16px;
  font-family: var(--mono);
  font-size: 10px;
  line-height: 1.7;
  overflow-x: auto;
  max-height: 400px;
  overflow-y: auto;
  white-space: pre-wrap;
  word-break: break-all;
}

.task-info {
  font-family: var(--serif);
  font-size: 14px;
  line-height: 1.7;
  padding: 16px;
  background: rgba(0,0,0,0.015);
  border: 1px solid var(--line);
  margin-bottom: 16px;
}

.task-info .label {
  font-family: var(--mono);
  font-size: 9px;
  letter-spacing: 0.15em;
  text-transform: uppercase;
  color: var(--fg-secondary);
  margin-bottom: 4px;
}

.breadcrumb {
  font-family: var(--mono);
  font-size: 10px;
  color: var(--fg-secondary);
  margin-bottom: 24px;
  letter-spacing: 0.05em;
}

.breadcrumb span { cursor: pointer; }
.breadcrumb span:hover { color: var(--fg); text-decoration: underline; }
.breadcrumb .sep { margin: 0 8px; color: var(--line-dark); }

.comparison-header {
  font-family: var(--mono);
  font-size: 9px;
  letter-spacing: 0.15em;
  text-transform: uppercase;
  color: var(--fg-secondary);
  margin-bottom: 12px;
  padding-bottom: 8px;
  border-bottom: 1px solid var(--line);
}

.traj-timeline {
  display: flex;
  flex-wrap: wrap;
  gap: 3px;
  margin-top: 8px;
  padding-left: 40px;
}

.traj-dot {
  width: 6px;
  height: 6px;
  border: 1px solid var(--line-dark);
}

.traj-dot.ok { background: var(--success); border-color: var(--success); }
.traj-dot.bad { background: var(--fail); border-color: var(--fail); }
.traj-dot.na { background: var(--line); border-color: var(--line-dark); }

.empty {
  text-align: center;
  padding: 80px 40px;
  color: var(--fg-secondary);
  font-family: var(--mono);
  font-size: 12px;
}

.loading {
  text-align: center;
  padding: 120px 40px;
  color: var(--fg-secondary);
  font-family: var(--mono);
  font-size: 12px;
}

::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--line-dark); }
::-webkit-scrollbar-thumb:hover { background: var(--fg-secondary); }

.board-table {
  width: 100%;
  border-collapse: collapse;
  font-family: var(--mono);
  font-size: 12px;
  margin-bottom: 32px;
}

.board-table {
  border-collapse: collapse;
  font-family: var(--mono);
  font-size: 12px;
  white-space: nowrap;
}

.board-table th, .board-table td {
  box-sizing: border-box;
}

.board-table th:nth-child(1),  .board-table td:nth-child(1)  { min-width: 24px; width: 24px; } /* Drag handle */
.board-table th:nth-child(2),  .board-table td:nth-child(2)  { min-width: 170px; } /* Experiment */
.board-table th:nth-child(3),  .board-table td:nth-child(3)  { min-width: 100px; } /* Student */
.board-table th:nth-child(4),  .board-table td:nth-child(4)  { min-width: 100px; } /* Teacher */
.board-table th:nth-child(5),  .board-table td:nth-child(5)  { min-width: 115px; } /* Started */
.board-table th:nth-child(6),  .board-table td:nth-child(6)  { min-width: 100px; } /* Duration */
.board-table th:nth-child(7),  .board-table td:nth-child(7)  { min-width: 95px;  } /* Cost */
.board-table th:nth-child(8),  .board-table td:nth-child(8)  { min-width: 120px; } /* SearchQA Evol */
.board-table th:nth-child(9),  .board-table td:nth-child(9)  { min-width: 95px;  } /* SearchQA success */
.board-table th:nth-child(10), .board-table td:nth-child(10) { min-width: 95px;  } /* SearchQA EM */
.board-table th:nth-child(11), .board-table td:nth-child(11) { min-width: 95px;  } /* SearchQA F1 */
.board-table th:nth-child(12), .board-table td:nth-child(12) { min-width: 95px;  } /* SearchQA SubEM */
.board-table th:nth-child(13), .board-table td:nth-child(13) { min-width: 125px; } /* Spreadsheet Evol */
.board-table th:nth-child(14), .board-table td:nth-child(14) { min-width: 95px;  } /* Sheet success */
.board-table th:nth-child(15), .board-table td:nth-child(15) { min-width: 95px;  } /* Sheet Soft */
.board-table th:nth-child(16), .board-table td:nth-child(16) { min-width: 95px;  } /* Sheet Hard */
.board-table th:nth-child(17), .board-table td:nth-child(17) { min-width: 115px; } /* DocVQA Evol */
.board-table th:nth-child(18), .board-table td:nth-child(18) { min-width: 100px; } /* DocVQA success */
.board-table th:nth-child(19), .board-table td:nth-child(19) { min-width: 100px; } /* DocVQA ANLS */
.board-table th:nth-child(20), .board-table td:nth-child(20) { min-width: 100px; } /* DocVQA Hard */
.board-table th:nth-child(21), .board-table td:nth-child(21) { min-width: 125px; } /* LiveMathC Evol */
.board-table th:nth-child(22), .board-table td:nth-child(22) { min-width: 110px; } /* LiveMathC success */
.board-table th:nth-child(23), .board-table td:nth-child(23) { min-width: 95px;  } /* LiveMathC Acc */
.board-table th:nth-child(24), .board-table td:nth-child(24) { min-width: 115px; } /* ALFWorld Evol */
.board-table th:nth-child(25), .board-table td:nth-child(25) { min-width: 100px; } /* ALFWorld success */
.board-table th:nth-child(26), .board-table td:nth-child(26) { min-width: 95px;  } /* ALFWorld Avg */
.board-table th:nth-child(27), .board-table td:nth-child(27) { min-width: 95px;  } /* ALFWorld Seen */
.board-table th:nth-child(28), .board-table td:nth-child(28) { min-width: 95px;  } /* ALFWorld Unseen */
.board-table th:nth-child(29), .board-table td:nth-child(29) { min-width: 105px; } /* WikiTQ Evol */
.board-table th:nth-child(30), .board-table td:nth-child(30) { min-width: 110px; } /* WikiTQ success */
.board-table th:nth-child(31), .board-table td:nth-child(31) { min-width: 120px; } /* WikiTQ SR */

.drag-handle {
  cursor: grab;
  color: var(--line-dark);
  user-select: none;
  text-align: center;
  font-family: var(--mono);
  font-size: 13px;
  line-height: 1;
  padding: 8px 4px !important;
}
.drag-handle:hover { color: var(--fg-secondary); }
.drag-handle:active { cursor: grabbing; }
.board-tr.dragging { opacity: 0.35; }
.board-tr.drop-above > td { box-shadow: inset 0 2px 0 var(--accent); }
.board-tr.drop-below > td { box-shadow: inset 0 -2px 0 var(--accent); }

.board-controls {
  display: flex;
  align-items: center;
  gap: 12px;
  margin: -8px 0 12px 0;
}
.board-reset {
  font-family: var(--mono);
  font-size: 10px;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--fg-secondary);
  cursor: pointer;
  padding: 4px 10px;
  border: 1px solid var(--line);
  background: var(--bg);
}
.board-reset:hover { color: var(--fg); border-color: var(--fg); }

.board-th {
  padding: 8px 10px;
  text-align: left;
  border-bottom: 2px solid var(--line-dark);
  color: var(--fg-secondary);
  font-weight: normal;
  cursor: pointer;
  user-select: none;
}

.board-th:hover {
  color: var(--accent);
}

.board-border-right {
  border-right: 2px solid var(--line-dark);
}

.board-td {
  padding: 8px 10px;
  border-bottom: 1px solid var(--line);
  vertical-align: middle;
}

.board-td-text {
  text-align: left;
  overflow: hidden;
  text-overflow: ellipsis;
}

.board-td-num {
  text-align: right;
}

.board-tr:hover {
  background: rgba(9, 105, 218, 0.03);
  cursor: pointer;
}

@media (max-width: 900px) {
  .sidebar { width: 220px; }
  .main { padding: 24px; }
}

/* ─── Skill Timeline ───────────────────────────────────────────── */
.skill-panel {
  border: 1px solid var(--line);
  border-radius: 4px;
  margin-bottom: 32px;
  background: var(--bg);
}
.skill-panel > summary {
  list-style: none;
  cursor: pointer;
  padding: 12px 16px;
  display: flex;
  align-items: center;
  gap: 10px;
  font-family: var(--mono);
  font-size: 12px;
  font-weight: 500;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  color: var(--fg);
  border-bottom: 1px solid transparent;
}
.skill-panel[open] > summary {
  border-bottom-color: var(--line);
}
.skill-panel > summary::-webkit-details-marker { display: none; }
.skill-panel > summary::before {
  content: '▶';
  font-size: 9px;
  color: var(--fg-secondary);
  transition: transform 0.15s;
}
.skill-panel[open] > summary::before { transform: rotate(90deg); }
.skill-panel > summary .panel-count {
  color: var(--fg-secondary);
  font-weight: 400;
  text-transform: none;
  letter-spacing: 0;
}

.skill-versions {
  position: relative;
  padding: 16px 16px 16px 40px;
}
.skill-versions::before {
  content: '';
  position: absolute;
  left: 22px;
  top: 24px;
  bottom: 24px;
  width: 1px;
  background: var(--line);
}

.skill-version {
  position: relative;
  margin-bottom: 4px;
  border-radius: 3px;
}
.skill-version > summary {
  list-style: none;
  cursor: pointer;
  padding: 8px 8px;
  display: flex;
  align-items: baseline;
  gap: 12px;
  font-family: var(--mono);
  font-size: 11px;
}
.skill-version > summary::-webkit-details-marker { display: none; }
.skill-version > summary:hover { background: rgba(0,0,0,0.02); }
.skill-version > summary::before {
  content: '';
  position: absolute;
  left: -22px;
  top: 14px;
  width: 9px;
  height: 9px;
  border-radius: 50%;
  border: 1.5px solid var(--line-dark);
  background: var(--bg);
}
.skill-version.kind-initial > summary::before { background: var(--fg-secondary); border-color: var(--fg-secondary); }
.skill-version.kind-final > summary::before { background: var(--accent); border-color: var(--accent); }
.skill-version.kind-after > summary::before { background: var(--info); border-color: var(--info); }

.skill-version .v-label {
  color: var(--fg);
  font-weight: 500;
  min-width: 56px;
}
.skill-version .v-meta {
  color: var(--fg-secondary);
}
.skill-version .v-ops-summary {
  color: var(--fg-secondary);
  margin-left: auto;
  font-size: 10px;
}
.skill-version .v-ops-summary .op-add    { color: var(--success); }
.skill-version .v-ops-summary .op-update { color: var(--info); }
.skill-version .v-ops-summary .op-delete { color: var(--fail); }

.skill-version-body {
  padding: 8px 12px 16px 12px;
  border-left: 2px solid var(--line);
  margin: 0 0 0 8px;
}
.skill-view-toggle {
  display: inline-flex;
  gap: 0;
  margin-bottom: 12px;
  border: 1px solid var(--line);
  border-radius: 3px;
  overflow: hidden;
  font-family: var(--mono);
  font-size: 10px;
}
.skill-view-toggle button {
  background: var(--bg);
  border: 0;
  padding: 4px 10px;
  cursor: pointer;
  color: var(--fg-secondary);
  font-family: inherit;
  font-size: inherit;
}
.skill-view-toggle button + button { border-left: 1px solid var(--line); }
.skill-view-toggle button.active { background: var(--fg); color: var(--bg); }

.skill-rule-card {
  border: 1px solid var(--line);
  border-radius: 3px;
  padding: 10px 12px;
  margin-bottom: 8px;
  background: var(--bg);
}
.skill-rule-card .r-head {
  display: flex;
  align-items: baseline;
  gap: 8px;
  margin-bottom: 6px;
}
.skill-rule-card .r-id {
  font-family: var(--mono);
  font-size: 10px;
  color: var(--fg-secondary);
  letter-spacing: 0.05em;
}
.skill-rule-card .r-title {
  font-family: var(--serif);
  font-size: 13px;
  font-weight: 600;
}
.skill-rule-card .r-content {
  font-size: 13px;
  margin: 4px 0;
  white-space: pre-wrap;
}
.skill-rule-card .r-why {
  font-size: 11px;
  color: var(--fg-secondary);
  border-left: 2px solid var(--line);
  padding-left: 8px;
  margin-top: 6px;
  font-style: italic;
}

.skill-rule-card.op-add    { border-left: 3px solid var(--success); }
.skill-rule-card.op-update { border-left: 3px solid var(--info); }
.skill-rule-card.op-delete { border-left: 3px solid var(--fail); opacity: 0.7; }
.skill-rule-card .op-badge {
  font-family: var(--mono);
  font-size: 9px;
  padding: 1px 6px;
  border-radius: 2px;
  letter-spacing: 0.05em;
}
.skill-rule-card.op-add    .op-badge { background: var(--success); color: var(--bg); }
.skill-rule-card.op-update .op-badge { background: var(--info);    color: var(--bg); }
.skill-rule-card.op-delete .op-badge { background: var(--fail);    color: var(--bg); }

.skill-empty {
  font-family: var(--mono);
  font-size: 11px;
  color: var(--fg-secondary);
  padding: 8px 0;
}
</style>
</head>
<body>
<div class="app" id="app">
  <div class="loading">Loading results...</div>
</div>

<script>
let DATA = [];
let groupDetailLoading = {};
let evalTaskDetailLoading = {};
let state = {
  view: 'board',
  selectedRun: null,
  selectedBench: null,
  selectedGroup: null,
  benchTab: 'evolve',      // 'evolve' | 'test'
  selectedSplit: null,
  selectedEvalTask: null,
  boardSort: { col: 'startedAt', dir: 'desc' }, // dir: 'desc' | 'asc' | null
  boardOrder: [],          // session-only manual order of run ids
};

// Module-level transient state for HTML5 drag-and-drop on the Board table.
let draggedId = null;

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}

function loadingMain(title, text) {
  const main = el('div', 'main');
  if (title) main.appendChild(el('div', 'section-title', title));
  main.appendChild(el('div', 'loading', text || 'Loading...'));
  return main;
}

function detailParams(run, bench, extra) {
  const params = new URLSearchParams();
  params.set('run', run.id);
  params.set('bench', bench.name);
  Object.entries(extra || {}).forEach(([k, v]) => params.set(k, String(v)));
  return params.toString();
}

async function fetchJson(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(await res.text() || (res.status + ' ' + res.statusText));
  return await res.json();
}

function loadGroupDetail(runIdx, benchIdx, groupIdx) {
  const run = DATA[runIdx];
  const bench = run && run.benchmarks[benchIdx];
  const group = bench && bench.groups[groupIdx];
  if (!run || !bench || !group || group._detailLoaded) return;
  const key = run.id + '/' + bench.name + '/' + group.id;
  if (groupDetailLoading[key]) return;
  groupDetailLoading[key] = true;
  fetchJson('/group?' + detailParams(run, bench, {group: group.id}))
    .then(detail => {
      DATA[runIdx].benchmarks[benchIdx].groups[groupIdx] = Object.assign({}, group, detail, {_detailLoaded: true});
      if (state.view === 'group' && state.selectedRun === runIdx && state.selectedBench === benchIdx && state.selectedGroup === groupIdx) render();
    })
    .catch(err => {
      group._detailError = err.message;
      if (state.view === 'group') render();
    })
    .finally(() => { delete groupDetailLoading[key]; });
}

function loadEvalTaskDetail(runIdx, benchIdx, splitName, taskIdx) {
  const run = DATA[runIdx];
  const bench = run && run.benchmarks[benchIdx];
  const splitInfo = bench && (bench.evaluation || {})[splitName];
  const task = splitInfo && (splitInfo.tasks || [])[taskIdx];
  if (!run || !bench || !splitInfo || !task || task._detailLoaded) return;
  const key = run.id + '/' + bench.name + '/' + splitName + '/' + taskIdx;
  if (evalTaskDetailLoading[key]) return;
  evalTaskDetailLoading[key] = true;
  fetchJson('/eval-task?' + detailParams(run, bench, {split: splitName, index: taskIdx}))
    .then(detail => {
      DATA[runIdx].benchmarks[benchIdx].evaluation[splitName].tasks[taskIdx] = Object.assign({}, task, detail, {_detailLoaded: true});
      if (state.view === 'eval_task' && state.selectedRun === runIdx && state.selectedBench === benchIdx && state.selectedSplit === splitName && state.selectedEvalTask === taskIdx) render();
    })
    .catch(err => {
      task._detailError = err.message;
      if (state.view === 'eval_task') render();
    })
    .finally(() => { delete evalTaskDetailLoading[key]; });
}

function fmt(n) {
  if (n === undefined || n === null) return '—';
  if (typeof n === 'number') return Number.isInteger(n) ? n.toString() : n.toFixed(3);
  return String(n);
}

function fmtPct(n) {
  if (n === undefined || n === null) return '—';
  return (n * 100).toFixed(1) + '%';
}

function fmtK(n) {
  if (n === undefined || n === null) return '—';
  if (n >= 1e6) return (n / 1e6).toFixed(2) + 'M';
  if (n >= 1e3) return (n / 1e3).toFixed(1) + 'K';
  return String(n);
}

function fmtStarted(iso) {
  if (!iso) return '—';
  // Accept either `2026-05-31T19:54:41` or `2026-05-31T19:54:41+0800`.
  const m = iso.match(/^(\\d{4})-(\\d{2})-(\\d{2})/);
  if (!m) return iso;
  return `${m[1]}-${m[2]}-${m[3]}`;
}

function fmtDuration(seconds) {
  if (seconds === undefined || seconds === null || isNaN(seconds)) return '—';
  const s = Math.max(0, Math.floor(seconds));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  if (h > 0) return `${h}h ${m}m`;
  if (m > 0) return `${m}m ${sec}s`;
  return `${sec}s`;
}

function extractTaskName(group) {
  const t = group.task || {};
  if (t.type === 'alfworld') return t.goal || t.task_type || 'ALFWorld task';
  if (t.type === 'spreadsheetbench') return (t.instruction || '').slice(0, 80) + ((t.instruction || '').length > 80 ? '...' : '');
  if (t.type === 'wikitq') return t.question || 'WikiTQ task';
  if (t.type === 'searchqa') return t.question || 'SearchQA task';
  if (t.type === 'livemathc') return (t.question || 'LiveMathC task').slice(0, 80);
  if (t.type === 'docvqa') return t.question || 'DocVQA task';
  return 'Task';
}

function renderSidebar() {
  const sidebar = el('div', 'sidebar');
  const header = el('div', 'sidebar-header');
  header.appendChild(el('h1', '', 'Experiment Runs'));
  const titleEl = el('div', 'title', 'PACT Board');
  titleEl.style.cursor = 'pointer';
  titleEl.title = 'Back to Board';
  titleEl.onclick = () => goBoard();
  header.appendChild(titleEl);
  sidebar.appendChild(header);

  const list = el('div', 'run-list');
  // Match the board's default ordering: most recent started_at first, with
  // missing timestamps sinking to the bottom. Click handlers still need the
  // original DATA index, so we attach it before sorting.
  const sidebarRuns = DATA.map((run, i) => ({ run, i }));
  sidebarRuns.sort((a, b) => {
    const ta = a.run.timing && a.run.timing.started_at ? Date.parse(a.run.timing.started_at) : NaN;
    const tb = b.run.timing && b.run.timing.started_at ? Date.parse(b.run.timing.started_at) : NaN;
    const aNil = isNaN(ta), bNil = isNaN(tb);
    if (aNil && bNil) return 0;
    if (aNil) return 1;
    if (bNil) return -1;
    return tb - ta;
  });
  sidebarRuns.forEach(({ run, i }) => {
    const item = el('div', 'run-item' + (state.selectedRun === i ? ' active' : ''));
    const displayName = run.exp_name || run.id;
    item.appendChild(el('span', 'name', displayName));
    let metaText = '';
    if (run.student_model) {
      metaText = `${run.student_model}`;
      if (run.teacher_model) metaText += ` ← ${run.teacher_model}`;
    } else {
      metaText = `${run.benchmarks.length} benchmark${run.benchmarks.length > 1 ? 's' : ''}`;
    }
    item.appendChild(el('span', 'meta', metaText));
    item.onclick = () => { state.selectedRun = i; state.selectedBench = 0; state.view = 'benchmark'; render(); };
    list.appendChild(item);
  });
  sidebar.appendChild(list);
  return sidebar;
}

function renderBoard() {
  const main = el('div', 'main');
  main.appendChild(el('div', 'section-title', 'Experiment Board'));

  if (DATA.length === 0) {
    main.appendChild(el('div', 'empty', 'No results found in the results/ directory.'));
    return main;
  }

  const rows = DATA.map(run => {
    const timing = run.timing || {};
    const r = {
      id: run.id,
      expName: run.exp_name || run.id,
      student: run.student_model || '',
      teacher: run.teacher_model || '',
      startedAt: timing.started_at || null,
      startedFromRunId: timing.started_at_source === 'run_id',
      duration: (typeof timing.elapsed_seconds === 'number') ? timing.elapsed_seconds : null,
      durationRunning: timing.started_at && !timing.ended_at,
      cost: run.cost ? run.cost.total : null,
      costDetail: run.cost || null,
      lfsPending: !!run.lfs_pending,
      alfworldSeen: null,        alfworldSeenN: null,    alfworldSeenSuccess: null,
      alfworldUnseen: null,      alfworldUnseenN: null,  alfworldUnseenSuccess: null,
      alfworldAvg: null,         alfworldTotalN: null,
      alfworldSuccess: null,
      sheetSoft: null,
      sheetHard: null,
      sheetN: null,
      sheetSuccess: null,
      wikitqSR: null,
      wikitqN: null,
      wikitqSuccess: null,
      searchqaEM: null,          searchqaF1: null,       searchqaSubEm: null,    searchqaN: null,
      searchqaSuccess: null,
      livemathcAcc: null,        livemathcN: null,
      livemathcSuccess: null,
      docvqaAnls: null,          docvqaHard: null,       docvqaN: null,
      docvqaSuccess: null,
      evoAlfw:     { tasks: 0, success: 0, rounds: 0, ops: 0 },
      evoSheet:    { tasks: 0, success: 0, rounds: 0, ops: 0 },
      evoWikitq:   { tasks: 0, success: 0, rounds: 0, ops: 0 },
      evoSearchqa: { tasks: 0, success: 0, rounds: 0, ops: 0 },
      evoLivemathc:{ tasks: 0, success: 0, rounds: 0, ops: 0 },
      evoDocvqa:   { tasks: 0, success: 0, rounds: 0, ops: 0 },
    };
    // Split-name aliases for back-compat: old runs in results/* use
    // {id_eval, ood_eval}; new runs (post Phase 2) use {test, test_seen, test_unseen}.
    // Match both so the board renders historical and current runs uniformly.
    const ALFW_SEEN_SPLITS = new Set(['id_eval', 'test_seen']);
    const ALFW_UNSEEN_SPLITS = new Set(['ood_eval', 'test_unseen']);
    const SHEET_TEST_SPLITS = new Set(['id_eval', 'test']);
    const WIKITQ_TEST_SPLITS = new Set(['ood_eval', 'test']);
    const GENERIC_TEST_SPLITS = new Set(['id_eval', 'ood_eval', 'test']);

    (run.summary || []).forEach(m => {
      const bench = m.benchmark;
      const target = m.target_benchmark;
      const split = m.split;

      if (bench === 'alfworld' && ALFW_SEEN_SPLITS.has(split)) {
        r.alfworldSeen = m.success_rate;
        r.alfworldSeenN = m.num_tasks;
        r.alfworldSeenSuccess = m.success_count;
      }
      if (bench === 'alfworld' && ALFW_UNSEEN_SPLITS.has(split)) {
        r.alfworldUnseen = m.success_rate;
        r.alfworldUnseenN = m.num_tasks;
        r.alfworldUnseenSuccess = m.success_count;
      }
      if (bench === 'spreadsheetbench' && SHEET_TEST_SPLITS.has(split)) {
        r.sheetSoft = m.soft_restriction_mean;
        r.sheetHard = m.hard_restriction_mean;
        r.sheetSuccess = m.soft_restriction_mean;
        r.sheetN = m.num_tasks;
      }
      if (bench === 'wikitq' && WIKITQ_TEST_SPLITS.has(split)) {
        r.wikitqSR = m.success_rate;
        r.wikitqSuccess = m.success_rate;
        r.wikitqN = m.num_tasks;
      }
      if (bench === 'searchqa' && GENERIC_TEST_SPLITS.has(split)) {
        r.searchqaEM = m.success_rate;            // SearchQA success == EM
        r.searchqaSuccess = m.success_rate;
        r.searchqaF1 = (typeof m.f1_mean === 'number') ? m.f1_mean : null;
        r.searchqaSubEm = (typeof m.sub_em_mean === 'number') ? m.sub_em_mean : null;
        r.searchqaN = m.num_tasks;
      }
      if (bench === 'livemathc' && GENERIC_TEST_SPLITS.has(split)) {
        r.livemathcAcc = m.success_rate;          // MCQ accuracy
        r.livemathcSuccess = m.success_rate;
        r.livemathcN = m.num_tasks;
      }
      if (bench === 'docvqa' && GENERIC_TEST_SPLITS.has(split)) {
        r.docvqaAnls = (typeof m.anls_mean === 'number') ? m.anls_mean : null;
        r.docvqaHard = m.success_rate;            // hard = ANLS >= 0.999
        r.docvqaSuccess = m.success_rate;
        r.docvqaN = m.num_tasks;
      }
      // Legacy cross-eval (spreadsheet student evaluated on wikitq) — kept
      // for back-compat with old runs; new runs no longer emit these.
      if (target === 'wikitq' && typeof split === 'string' && split.includes('wikitq')) {
        r.wikitqSR = m.success_rate;
        r.wikitqN = m.num_tasks;
      }
    });
    // ALFWorld Avg: sample-level macro across seen + unseen.
    // (rate_seen * N_seen + rate_unseen * N_unseen) / (N_seen + N_unseen)
    //   = (success_count_seen + success_count_unseen) / (N_seen + N_unseen)
    const ns = r.alfworldSeenN || 0, nu = r.alfworldUnseenN || 0;
    if (ns + nu > 0) {
      r.alfworldAvg = ((r.alfworldSeenSuccess || 0) + (r.alfworldUnseenSuccess || 0)) / (ns + nu);
      r.alfworldTotalN = ns + nu;
      r.alfworldSuccess = r.alfworldUnseen;
    }
    (run.benchmarks || []).forEach(b => {
      const evo = b.evolution_summary || [];
      const stat = {
        tasks: evo.length,
        success: evo.filter(g => g.success).length,
        rounds: evo.reduce((s, g) => s + (g.rounds || []).length, 0),
        ops: evo.reduce((s, g) => s + (g.rounds || []).filter(rnd => rnd.ops_committed).length, 0),
      };
      if (b.name === 'alfworld') r.evoAlfw = stat;
      else if (b.name === 'spreadsheetbench') r.evoSheet = stat;
      else if (b.name === 'wikitq') r.evoWikitq = stat;
      else if (b.name === 'searchqa') r.evoSearchqa = stat;
      else if (b.name === 'livemathc') r.evoLivemathc = stat;
      else if (b.name === 'docvqa') r.evoDocvqa = stat;
    });
    return r;
  });

  function sortRows(colKey, rowsToSort) {
    // sortHeader.onclick is the single source of truth for direction —
    // we just read the current state and apply it.
    const dir = state.boardSort.dir;
    if (!dir) return rowsToSort.slice().reverse();
    function getVal(obj, key) {
      const parts = key.split('.');
      let v = obj;
      for (const p of parts) { v = v ? v[p] : undefined; }
      return v;
    }
    // ISO-string time columns must sort as actual instants, not
    // lexicographically — otherwise mixed/absent timezone suffixes can
    // flip order across runs from different zones.
    const TIME_COLS = new Set(['startedAt']);
    const isTimeCol = TIME_COLS.has(colKey);
    return rowsToSort.slice().sort((a, b) => {
      let va = getVal(a, colKey), vb = getVal(b, colKey);
      // Nest objects (e.g. evoAlfw) are sorted by their success count so
      // the column is at least usable; non-numeric/non-string scalars fall
      // through to the comparison below.
      if (va && typeof va === 'object') va = va.success ?? null;
      if (vb && typeof vb === 'object') vb = vb.success ?? null;
      if (isTimeCol) {
        const ta = va ? Date.parse(va) : NaN;
        const tb = vb ? Date.parse(vb) : NaN;
        va = isNaN(ta) ? null : ta;
        vb = isNaN(tb) ? null : tb;
      }
      const aNil = va === null || va === undefined;
      const bNil = vb === null || vb === undefined;
      // Nulls always sink to the bottom regardless of asc/desc so missing
      // data never spuriously beats real data.
      if (aNil && bNil) return 0;
      if (aNil) return 1;
      if (bNil) return -1;
      if (typeof va === 'string') va = va.toLowerCase();
      if (typeof vb === 'string') vb = vb.toLowerCase();
      if (va < vb) return dir === 'asc' ? -1 : 1;
      if (va > vb) return dir === 'asc' ? 1 : -1;
      return 0;
    });
  }

  function sortHeader(label, colKey, currentSort, extraCls) {
    const th = el('th', 'board-th' + (extraCls ? ' ' + extraCls : ''));
    th.textContent = label;
    if (currentSort.col === colKey && currentSort.dir) {
      th.textContent += currentSort.dir === 'desc' ? ' ↓' : ' ↑';
      th.style.color = 'var(--accent)';
    }
    th.onclick = () => {
      state.boardSort = { col: colKey, dir: 'desc' };
      if (currentSort.col === colKey) {
        if (currentSort.dir === 'desc') state.boardSort.dir = 'asc';
        else if (currentSort.dir === 'asc') state.boardSort.dir = null;
      }
      render();
    };
    return th;
  }

  function pctCell(v, n) {
    const td = el('td', 'board-td board-td-num');
    if (v !== null && v !== undefined) {
      const pct = fmtPct(v);
      const color = v >= 0.5 ? 'var(--success)' : 'var(--fail)';
      const suffix = (n !== null && n !== undefined)
        ? ` <span style="color:var(--fg-secondary);font-size:10px">(${n})</span>`
        : '';
      td.innerHTML = `<span style="color:${color}">${pct}</span>${suffix}`;
    } else {
      td.textContent = '—';
    }
    return td;
  }

  function evolCell(stat) {
    const td = el('td', 'board-td board-td-num');
    if (stat.tasks === 0) {
      td.textContent = '—';
    } else {
      const rate = stat.tasks > 0 ? (stat.success / stat.tasks) : 0;
      td.innerHTML = `<span style="color:${rate >= 0.5 ? 'var(--success)' : 'var(--fail)'}">${stat.success}/${stat.tasks}</span>&nbsp;<span style="color:var(--fg-secondary)">${stat.rounds}r&nbsp;${stat.ops}o</span>`;
    }
    return td;
  }

  function applyManualOrder(rowsToOrder, order) {
    if (!order || order.length === 0) return rowsToOrder;
    const indexMap = new Map(order.map((id, i) => [id, i]));
    const known = [], unknown = [];
    rowsToOrder.forEach(r => {
      if (indexMap.has(r.id)) known.push(r); else unknown.push(r);
    });
    known.sort((a, b) => indexMap.get(a.id) - indexMap.get(b.id));
    return known.concat(unknown);
  }

  const currentSort = state.boardSort;
  let displayRows = rows;
  if (currentSort.col) {
    displayRows = sortRows(currentSort.col, rows);
  } else if (state.boardOrder && state.boardOrder.length > 0) {
    displayRows = applyManualOrder(rows, state.boardOrder);
  }

  // Controls: reset button shows up only when manual order is active.
  if (state.boardOrder && state.boardOrder.length > 0) {
    const controls = el('div', 'board-controls');
    const resetBtn = el('button', 'board-reset', 'Reset Order');
    resetBtn.onclick = () => { state.boardOrder = []; render(); };
    controls.appendChild(resetBtn);
    main.appendChild(controls);
  }

  const wrap = el('div', '');
  wrap.style.overflowX = 'auto';
  wrap.style.border = '1px solid var(--line)';

  const table = el('table', 'board-table');

  const thead = el('thead', '');
  const h2 = el('tr', '');
  // Drag-handle column header (non-sortable placeholder for alignment).
  const handleTh = el('th', 'board-th');
  handleTh.style.cursor = 'default';
  handleTh.textContent = '';
  h2.appendChild(handleTh);
  const colMeta = [
    { key: 'expName',    label: 'Experiment',       cls: '' },
    { key: 'student',    label: 'Student',          cls: '' },
    { key: 'teacher',    label: 'Teacher',          cls: '' },
    { key: 'startedAt',  label: 'Started',          cls: '' },
    { key: 'duration',   label: 'Duration',         cls: '' },
    { key: 'cost',       label: 'Cost',             cls: 'board-border-right' },
    { key: 'evoSearchqa',    label: 'SearchQA Evol',    cls: '' },
    { key: 'searchqaSuccess', label: 'success(em)',     cls: '' },
    { key: 'searchqaEM',     label: 'EM',               cls: '' },
    { key: 'searchqaF1',     label: 'F1',               cls: '' },
    { key: 'searchqaSubEm',  label: 'SubEM',            cls: 'board-border-right' },
    { key: 'evoSheet',       label: 'Spreadsheet Evol', cls: '' },
    { key: 'sheetSuccess',   label: 'success(soft)',    cls: '' },
    { key: 'sheetSoft',      label: 'Soft',             cls: '' },
    { key: 'sheetHard',      label: 'Hard',             cls: 'board-border-right' },
    { key: 'evoDocvqa',      label: 'DocVQA Evol',      cls: '' },
    { key: 'docvqaSuccess',  label: 'success(hard)',    cls: '' },
    { key: 'docvqaAnls',     label: 'ANLS',             cls: '' },
    { key: 'docvqaHard',     label: 'Hard',             cls: 'board-border-right' },
    { key: 'evoLivemathc',   label: 'LiveMathC Evol',   cls: '' },
    { key: 'livemathcSuccess', label: 'success(em)',    cls: '' },
    { key: 'livemathcAcc',   label: 'Acc',              cls: 'board-border-right' },
    { key: 'evoAlfw',        label: 'ALFWorld Evol',    cls: '' },
    { key: 'alfworldSuccess', label: 'success(unseen)', cls: '' },
    { key: 'alfworldAvg',    label: 'Avg',              cls: '' },
    { key: 'alfworldSeen',   label: 'Seen',             cls: '' },
    { key: 'alfworldUnseen', label: 'Unseen',           cls: 'board-border-right' },
    { key: 'evoWikitq',      label: 'WikiTQ Evol',      cls: '' },
    { key: 'wikitqSuccess',  label: 'success(em)',      cls: '' },
    { key: 'wikitqSR',       label: 'WikiTQ SR',        cls: '' },
  ];
  colMeta.forEach(c => {
    h2.appendChild(sortHeader(c.label, c.key, currentSort, c.cls));
  });
  thead.appendChild(h2);
  table.appendChild(thead);

  const tbody = el('tbody', '');
  displayRows.forEach(r => {
    const tr = el('tr', 'board-tr');
    tr.dataset.runId = r.id;
    tr.onclick = () => {
      const idx = DATA.findIndex(d => d.id === r.id);
      if (idx >= 0) {
        state.selectedRun = idx;
        state.selectedBench = 0;
        state.view = 'benchmark';
        render();
      }
    };

    // ── Drag handle (only this cell starts a drag; rest of row stays clickable) ──
    const handleTd = el('td', 'board-td drag-handle');
    handleTd.textContent = '⋮⋮';
    handleTd.title = 'Drag to reorder';
    handleTd.draggable = true;
    handleTd.onclick = (e) => e.stopPropagation();
    handleTd.ondragstart = (e) => {
      draggedId = r.id;
      e.dataTransfer.effectAllowed = 'move';
      try {
        e.dataTransfer.setData('text/plain', r.id);
        if (e.dataTransfer.setDragImage) e.dataTransfer.setDragImage(tr, 0, 0);
      } catch (err) {}
      setTimeout(() => tr.classList.add('dragging'), 0);
    };
    handleTd.ondragend = () => {
      draggedId = null;
      document.querySelectorAll('.board-tr').forEach(t => {
        t.classList.remove('dragging', 'drop-above', 'drop-below');
      });
    };
    tr.appendChild(handleTd);

    // Row-level drop target.
    tr.ondragover = (e) => {
      if (!draggedId || draggedId === r.id) return;
      e.preventDefault();
      e.dataTransfer.dropEffect = 'move';
      const rect = tr.getBoundingClientRect();
      const above = e.clientY < rect.top + rect.height / 2;
      document.querySelectorAll('.board-tr').forEach(t => {
        if (t !== tr) t.classList.remove('drop-above', 'drop-below');
      });
      tr.classList.toggle('drop-above', above);
      tr.classList.toggle('drop-below', !above);
    };
    tr.ondrop = (e) => {
      e.preventDefault();
      if (!draggedId || draggedId === r.id) return;
      const rect = tr.getBoundingClientRect();
      const above = e.clientY < rect.top + rect.height / 2;
      const currentIds = displayRows.map(d => d.id);
      const fromIdx = currentIds.indexOf(draggedId);
      if (fromIdx < 0) return;
      currentIds.splice(fromIdx, 1);
      let toIdx = currentIds.indexOf(r.id);
      if (toIdx < 0) return;
      if (!above) toIdx += 1;
      currentIds.splice(toIdx, 0, draggedId);
      state.boardOrder = currentIds;
      state.boardSort = { col: null, dir: null }; // manual order takes over
      render();
    };

    // Info columns
    const nameTd = el('td', 'board-td board-td-text');
    if (r.lfsPending) {
      nameTd.innerHTML = `<span style="display:inline-block;padding:1px 5px;margin-right:6px;font-size:9px;letter-spacing:0.08em;color:var(--warn);border:1px solid var(--warn);border-radius:2px">LFS</span>${escapeHtml(r.expName)}`;
      nameTd.title = r.expName + '  —  files are Git LFS pointers; run `git lfs pull` to download.';
    } else {
      nameTd.textContent = r.expName;
      nameTd.title = r.expName;
    }
    tr.appendChild(nameTd);

    [r.student, r.teacher].forEach(v => {
      const td = el('td', 'board-td board-td-text');
      td.textContent = v || '—';
      tr.appendChild(td);
    });

    const startTd = el('td', 'board-td board-td-text');
    if (r.startedAt) {
      const txt = fmtStarted(r.startedAt);
      const color = r.startedFromRunId ? 'var(--fg-secondary)' : 'var(--fg)';
      const title = r.startedFromRunId
        ? 'Inferred from run_id (no timing.json on disk)'
        : r.startedAt;
      startTd.innerHTML = `<span style="color:${color}" title="${title}">${txt}</span>`;
    } else {
      startTd.textContent = '—';
    }
    tr.appendChild(startTd);

    const durTd = el('td', 'board-td board-td-num');
    if (r.duration !== null) {
      const txt = fmtDuration(r.duration);
      durTd.innerHTML = r.durationRunning
        ? `<span style="color:var(--info)" title="run still in progress">${txt}…</span>`
        : `<span style="color:var(--fg-secondary)">${txt}</span>`;
    } else {
      durTd.textContent = '—';
    }
    tr.appendChild(durTd);

    // Cost column
    const costTd = el('td', 'board-td board-td-num');
    if (r.cost !== null && r.cost !== undefined) {
      const costStr = '$' + r.cost.toFixed(2);
      let title = '';
      if (r.costDetail) {
        const parts = [];
        for (const role of ['student', 'teacher', 'critic']) {
          if (r.costDetail[role]) {
            const d = r.costDetail[role];
            parts.push(role + ': ' + (d.cost !== null ? '$' + d.cost.toFixed(2) : 'N/A')
              + ' (' + d.model + ')');
          }
        }
        if (!r.costDetail.all_priced) parts.push('(some models unpriced)');
        title = parts.join('\\n');
      }
      costTd.innerHTML = `<span style="color:var(--fg)"${title ? ` title="${title}"` : ''}>${costStr}</span>`;
    } else {
      costTd.textContent = '—';
    }
    tr.appendChild(costTd);

    // SearchQA: Evol → success(em) → EM → F1 → SubEM
    tr.appendChild(evolCell(r.evoSearchqa));
    tr.appendChild(pctCell(r.searchqaSuccess, r.searchqaN));
    tr.appendChild(pctCell(r.searchqaEM,    r.searchqaN));
    tr.appendChild(pctCell(r.searchqaF1,    r.searchqaN));
    tr.appendChild(pctCell(r.searchqaSubEm, r.searchqaN));

    // Spreadsheet: Evol → success(soft) → Soft → Hard
    tr.appendChild(evolCell(r.evoSheet));
    tr.appendChild(pctCell(r.sheetSuccess, r.sheetN));
    tr.appendChild(pctCell(r.sheetSoft, r.sheetN));
    tr.appendChild(pctCell(r.sheetHard, r.sheetN));

    // DocVQA: Evol → success(hard) → ANLS → Hard
    tr.appendChild(evolCell(r.evoDocvqa));
    tr.appendChild(pctCell(r.docvqaSuccess, r.docvqaN));
    tr.appendChild(pctCell(r.docvqaAnls,   r.docvqaN));
    tr.appendChild(pctCell(r.docvqaHard,   r.docvqaN));

    // LiveMathC: Evol → success(em) → Acc
    tr.appendChild(evolCell(r.evoLivemathc));
    tr.appendChild(pctCell(r.livemathcSuccess, r.livemathcN));
    tr.appendChild(pctCell(r.livemathcAcc, r.livemathcN));

    // ALFWorld: Evol → success(unseen) → Avg → Seen → Unseen
    tr.appendChild(evolCell(r.evoAlfw));
    tr.appendChild(pctCell(r.alfworldSuccess, r.alfworldUnseenN));
    tr.appendChild(pctCell(r.alfworldAvg,    r.alfworldTotalN));
    tr.appendChild(pctCell(r.alfworldSeen,   r.alfworldSeenN));
    tr.appendChild(pctCell(r.alfworldUnseen, r.alfworldUnseenN));

    // WikiTQ: Evol → success(em) → SR
    tr.appendChild(evolCell(r.evoWikitq));
    tr.appendChild(pctCell(r.wikitqSuccess, r.wikitqN));
    tr.appendChild(pctCell(r.wikitqSR, r.wikitqN));

    tbody.appendChild(tr);
  });
  table.appendChild(tbody);
  wrap.appendChild(table);
  main.appendChild(wrap);

  return main;
}

function isSpreadsheetSplit(splitInfo) {
  if (!splitInfo) return false;
  const m = splitInfo.metrics || {};
  const modes = m.evaluation_modes || [];
  if (modes.some(mode => String(mode).includes('spreadsheet'))) return true;
  if (m.target_benchmark === 'spreadsheetbench') return true;
  return false;
}

function renderBenchmark() {
  const run = DATA[state.selectedRun];
  if (!run) return renderBoard();
  const bench = run.benchmarks[state.selectedBench] || run.benchmarks[0];
  if (!bench) return renderBoard();

  const main = el('div', 'main');
  const bc = el('div', 'breadcrumb');
  bc.innerHTML = '<span onclick="goBoard()">Board</span><span class="sep">/</span><span>' + run.id + '</span><span class="sep">/</span><span>' + bench.name + '</span>';
  main.appendChild(bc);
  main.appendChild(el('div', 'section-title', bench.name));

  // ── Cost summary (run-level) ──
  if (run.cost && run.cost.total !== null && run.cost.total !== undefined) {
    const costPanel = el('div', 'metrics-row');
    const costCells = [];
    costCells.push(['Total Cost', '$' + run.cost.total.toFixed(2)]);
    for (const role of ['student', 'teacher', 'critic']) {
      if (run.cost[role]) {
        const d = run.cost[role];
        const label = role.charAt(0).toUpperCase() + role.slice(1);
        const costStr = d.cost !== null ? '$' + d.cost.toFixed(2) : 'N/A';
        costCells.push([label + ' (' + d.model + ')', costStr]);
      }
    }
    costCells.forEach(([label, value]) => {
      const cell = el('div', 'metric-cell');
      cell.appendChild(el('div', 'metric-label', label));
      cell.appendChild(el('div', 'metric-value', value));
      costPanel.appendChild(cell);
    });
    main.appendChild(costPanel);
  }

  // ── Benchmark selector ──
  if (run.benchmarks.length > 1) {
    const benchSwitch = el('div', 'bench-tabs');
    run.benchmarks.forEach((b, i) => {
      const tab = el('div', 'bench-tab' + (i === state.selectedBench ? ' active' : ''), b.name);
      tab.onclick = () => {
        state.selectedBench = i;
        state.selectedSplit = null;
        state.selectedEvalTask = null;
        render();
      };
      benchSwitch.appendChild(tab);
    });
    main.appendChild(benchSwitch);
  }

  // ── Evolve / Test tab switcher ──
  const phaseTabs = el('div', 'bench-tabs');
  ['evolve', 'test'].forEach(name => {
    const label = name === 'evolve' ? 'Evolve' : 'Test';
    const tab = el('div', 'bench-tab' + (state.benchTab === name ? ' active' : ''), label);
    tab.onclick = () => {
      state.benchTab = name;
      state.selectedSplit = null;
      state.selectedEvalTask = null;
      render();
    };
    phaseTabs.appendChild(tab);
  });
  main.appendChild(phaseTabs);

  if (state.benchTab === 'evolve') {
    main.appendChild(renderEvolveSection(bench));
  } else {
    main.appendChild(renderTestSection(bench));
  }

  return main;
}

// ─── Skill Timeline ──────────────────────────────────────────────
// File-name shapes produced by run_experiment.py:
//   0000_initial.md
//   000N_after_group_GGGG_round_RR.md
//   final.md
function parseSkillFilename(name) {
  if (name === 'final.md') return { kind: 'final' };
  const initMatch = name.match(/^(\\d+)_initial\\.md$/);
  if (initMatch) return { kind: 'initial', idx: parseInt(initMatch[1], 10) };
  const afterMatch = name.match(/^(\\d+)_after_group_(\\d+)_round_(\\d+)\\.md$/);
  if (afterMatch) {
    return {
      kind: 'after',
      idx: parseInt(afterMatch[1], 10),
      groupIdx: parseInt(afterMatch[2], 10),
      round: parseInt(afterMatch[3], 10),
    };
  }
  return { kind: 'unknown' };
}

// Parse the canonical rule block format (3-line):
//   [RULE 001] Title
//   content: ...
//   why: ...
function parseSkillRules(text) {
  const rules = [];
  if (!text) return rules;
  const blocks = text.split(/\\n(?=\\[RULE\\s+\\d)/);
  for (const block of blocks) {
    const m = block.match(/^\\[RULE\\s+(\\d+)\\]\\s*(.*)/);
    if (!m) continue;
    const id = m[1];
    const title = m[2].trim();
    let content = '', why = '';
    const lines = block.split('\\n').slice(1);
    let mode = null, buf = [];
    const flush = () => {
      if (!mode) return;
      const joined = buf.join('\\n').trim();
      if (mode === 'content') content = joined;
      else if (mode === 'why') why = joined;
      buf = [];
    };
    for (const line of lines) {
      if (line.startsWith('content:')) { flush(); mode = 'content'; buf.push(line.slice(8).trim()); }
      else if (line.startsWith('why:'))  { flush(); mode = 'why';     buf.push(line.slice(4).trim()); }
      else if (mode)                     { buf.push(line); }
    }
    flush();
    rules.push({ id, title, content, why });
  }
  return rules;
}

// Build a synthetic ops list for version `v` from bench.edit_log. The edit
// log is written live by SkillState.record_edits() — single source of truth.
// Each `after_group_NN_round_MM` snapshot maps to all edits whose
// (group_index, round_index) match. (Most rounds commit at most a few ops.)
function opsForVersion(v, editLog) {
  if (v.kind !== 'after') return [];
  return (editLog || []).filter(e => e.group_index === v.groupIdx && e.round_index === v.round)
    .map(e => ({
      op: e.op,
      id: e.rule_id,
      rule: e.after || e.before || {},
      before: e.before,
      after: e.after,
      edit_index: e.edit_index,
    }));
}

function renderOpsSummary(ops) {
  const counts = { add: 0, update: 0, delete: 0 };
  ops.forEach(o => { if (counts[o.op] !== undefined) counts[o.op]++; });
  const span = el('span', 'v-ops-summary');
  const parts = [];
  if (counts.add)    parts.push(['op-add',    '+' + counts.add]);
  if (counts.update) parts.push(['op-update', '~' + counts.update]);
  if (counts.delete) parts.push(['op-delete', '-' + counts.delete]);
  parts.forEach(([cls, txt], i) => {
    if (i > 0) span.appendChild(document.createTextNode(' '));
    span.appendChild(el('span', cls, txt));
  });
  return span;
}

function renderRuleCard(rule, opTag, opMeta) {
  const card = el('div', 'skill-rule-card' + (opTag ? ' op-' + opTag : ''));
  const head = el('div', 'r-head');
  head.appendChild(el('span', 'r-id', '[' + (rule.id || '—') + ']'));
  head.appendChild(el('span', 'r-title', rule.title || '(untitled)'));
  if (opTag) head.appendChild(el('span', 'op-badge', opTag.toUpperCase()));
  if (opMeta && opMeta.edit_index !== undefined) {
    const tag = el('span', '');
    tag.style.cssText = 'margin-left:6px;font-family:var(--mono);font-size:9px;color:var(--fg-secondary);';
    tag.textContent = 'edit ' + opMeta.edit_index;
    head.appendChild(tag);
  }
  card.appendChild(head);
  if (rule.content) card.appendChild(el('div', 'r-content', rule.content));
  if (rule.why)     card.appendChild(el('div', 'r-why', 'why: ' + rule.why));
  // For update ops show the prior rule so the change is visible.
  if (opMeta && opMeta.before && opTag === 'update') {
    const beforeBlock = el('div', '');
    beforeBlock.style.cssText = 'margin-top:6px;padding:6px 8px;background:rgba(0,0,0,0.03);border-left:3px solid var(--fg-secondary);font-size:11px;';
    const lbl = el('div', '');
    lbl.style.cssText = 'font-family:var(--mono);font-size:9px;color:var(--fg-secondary);margin-bottom:3px;';
    lbl.textContent = 'before:';
    beforeBlock.appendChild(lbl);
    if (opMeta.before.content) beforeBlock.appendChild(el('div', 'r-content', opMeta.before.content));
    if (opMeta.before.why)     beforeBlock.appendChild(el('div', 'r-why', 'why: ' + opMeta.before.why));
    card.appendChild(beforeBlock);
  }
  return card;
}

function renderSkillVersionBody(v, ops) {
  const body = el('div', 'skill-version-body');

  const toggle = el('div', 'skill-view-toggle');
  const btnFull = el('button', 'active', 'Full skill (' + v.rules.length + ')');
  const btnDiff = el('button', '', 'Diff vs prev (' + ops.length + ')');
  toggle.appendChild(btnFull);
  toggle.appendChild(btnDiff);
  body.appendChild(toggle);

  const fullView = el('div', '');
  if (v.rules.length === 0) {
    fullView.appendChild(el('div', 'skill-empty', '(no rules at this version)'));
  } else {
    v.rules.forEach(r => fullView.appendChild(renderRuleCard(r)));
  }

  const diffView = el('div', '');
  diffView.style.display = 'none';
  if (ops.length === 0) {
    diffView.appendChild(el('div', 'skill-empty',
      v.kind === 'initial' ? '(initial version — nothing before it)' :
      v.kind === 'final'   ? '(final = last committed version)' :
                             '(no ops recorded for this version)'));
  } else {
    ops.forEach(op => {
      const rule = op.rule || { id: op.id, title: '(rule ' + op.id + ')', content: '', why: '' };
      diffView.appendChild(renderRuleCard(rule, op.op, op));
    });
  }
  body.appendChild(fullView);
  body.appendChild(diffView);

  btnFull.onclick = () => {
    btnFull.classList.add('active'); btnDiff.classList.remove('active');
    fullView.style.display = ''; diffView.style.display = 'none';
  };
  btnDiff.onclick = () => {
    btnDiff.classList.add('active'); btnFull.classList.remove('active');
    diffView.style.display = ''; fullView.style.display = 'none';
  };

  return body;
}

function renderSkillTimeline(bench) {
  const skills = bench.skills || [];
  if (skills.length === 0) return null;

  const versions = skills.map(s => {
    const meta = parseSkillFilename(s.name);
    return Object.assign({ name: s.name, rules: parseSkillRules(s.content) }, meta);
  });
  const editLog = bench.edit_log || [];

  const panel = el('details', 'skill-panel');
  panel.open = true;
  const sum = el('summary', '');
  sum.appendChild(document.createTextNode('Skill Timeline'));
  sum.appendChild(el('span', 'panel-count', '(' + versions.length + ' versions)'));
  panel.appendChild(sum);

  const list = el('div', 'skill-versions');
  versions.forEach(v => {
    const ops = opsForVersion(v, editLog);
    const node = el('details', 'skill-version kind-' + v.kind);
    const head = el('summary', '');

    let label, meta;
    if (v.kind === 'initial')      { label = 'v' + String(v.idx).padStart(4, '0'); meta = 'initial'; }
    else if (v.kind === 'final')   { label = 'final'; meta = 'last committed skill'; }
    else if (v.kind === 'after')   {
      label = 'v' + String(v.idx).padStart(4, '0');
      meta = 'group ' + String(v.groupIdx).padStart(4, '0') + ' · round ' + v.round;
    } else                         { label = v.name; meta = ''; }

    head.appendChild(el('span', 'v-label', label));
    head.appendChild(el('span', 'v-meta', meta));
    if (ops.length) head.appendChild(renderOpsSummary(ops));
    node.appendChild(head);

    // Lazy-render the body on first open to keep initial render cheap.
    let rendered = false;
    node.addEventListener('toggle', () => {
      if (node.open && !rendered) {
        node.appendChild(renderSkillVersionBody(v, ops));
        rendered = true;
      }
    });
    list.appendChild(node);
  });
  panel.appendChild(list);
  return panel;
}

function renderEvolveSection(bench) {
  const wrap = el('div', '');

  const skillPanel = renderSkillTimeline(bench);
  if (skillPanel) wrap.appendChild(skillPanel);

  // ── Training statistics ──
  // Prefer evolution_summary when available; otherwise compute from groups on disk.
  const evo = bench.evolution_summary || [];
  const groups = bench.groups || [];
  const totalTasks = evo.length || groups.length;
  const successTasks = evo.length
    ? evo.filter(g => g.success).length
    : groups.filter(g => (g.summary || {}).success).length;
  const totalRounds = evo.length
    ? evo.reduce((s, g) => s + (g.rounds || []).length, 0)
    : groups.reduce((s, g) => s + (g.rounds || []).length, 0);
  const committedRounds = evo.length
    ? evo.reduce((s, g) => s + (g.rounds || []).filter(r => r.ops_committed).length, 0)
    : groups.reduce((s, g) => s + (g.rounds || []).filter(r => (r.critic || {}).accepted).length, 0);

  const trainRow = el('div', 'metrics-row');
  const trainCells = [
    ['Train Tasks', fmt(totalTasks)],
    ['Train Successes', fmt(successTasks)],
    ['Total Rounds', fmt(totalRounds)],
    ['Ops Committed', fmt(committedRounds)],
  ];
  trainCells.forEach(([label, value]) => {
    const cell = el('div', 'metric-cell');
    cell.appendChild(el('div', 'metric-label', label));
    const valEl = el('div', 'metric-value', value);
    if (label === 'Train Successes' && totalTasks > 0) {
      valEl.classList.add(successTasks / totalTasks >= 0.5 ? 'success' : 'fail');
    }
    cell.appendChild(valEl);
    trainRow.appendChild(cell);
  });
  wrap.appendChild(trainRow);

  wrap.appendChild(el('div', 'section-title', 'Evolution Timeline (' + (bench.groups || []).length + ' groups)'));

  const timeline = el('div', 'timeline');
  (bench.groups || []).forEach((group, gi) => {
    const summary = group.summary || {};
    const success = summary.success || false;
    const rounds = group.rounds || [];

    const item = el('div', 'group-item ' + (success ? 'success' : 'fail'));
    const header = el('div', 'group-header');
    header.appendChild(el('span', 'group-idx', String(gi).padStart(4, '0')));
    header.appendChild(el('span', 'group-task', extractTaskName(group)));
    header.appendChild(el('span', 'group-status ' + (success ? 'success' : 'fail'), success ? 'Success' : 'Fail'));
    header.appendChild(el('span', 'group-rounds', rounds.length + ' round' + (rounds.length > 1 ? 's' : '')));
    item.appendChild(header);

    if (group.student_initial && group.student_initial.evaluation && group.student_initial.evaluation.steps) {
      const dots = el('div', 'traj-timeline');
      group.student_initial.evaluation.steps.forEach(step => {
        dots.appendChild(el('div', 'traj-dot ' + (step.admissible ? 'ok' : 'bad')));
      });
      item.appendChild(dots);
    }

    item.onclick = () => { state.selectedGroup = gi; state.view = 'group'; render(); };
    timeline.appendChild(item);
  });
  wrap.appendChild(timeline);
  return wrap;
}

function renderTestSection(bench) {
  const wrap = el('div', '');
  const ev = bench.evaluation || {};
  const splits = Object.entries(ev);
  if (splits.length === 0) {
    wrap.appendChild(el('div', 'empty', 'No evaluation results for this benchmark.'));
    return wrap;
  }

  // ── Metric cards: one per split. Spreadsheet splits → soft/hard primary; others → success rate only ──
  const evalRow = el('div', 'metrics-row');
  splits.forEach(([split, info]) => {
    const m = info.metrics || {};
    const isSheet = isSpreadsheetSplit(info);
    const card = el('div', 'eval-card');
    card.appendChild(el('div', 'eval-card-header', split));

    if (isSheet) {
      // Primary metric: soft accuracy (lenient); secondary: hard accuracy
      const hard = m.hard_restriction_mean;
      const soft = m.soft_restriction_mean;
      const mainLine = el('div', 'eval-card-main');
      const rateEl = el('div', 'eval-card-rate');
      rateEl.textContent = (soft !== undefined) ? fmt(soft * 100) + '%' : '—';
      if (soft !== undefined) rateEl.classList.add(soft >= 0.5 ? 'success' : 'fail');
      mainLine.appendChild(rateEl);
      mainLine.appendChild(el('span', 'eval-card-count', 'soft'));
      card.appendChild(mainLine);

      if (hard !== undefined) {
        const subLine = el('div', 'eval-card-main');
        subLine.style.marginTop = '6px';
        const hardEl = el('div', 'eval-card-rate');
        hardEl.style.fontSize = '20px';
        hardEl.textContent = fmt(hard * 100) + '%';
        if (hard !== undefined) hardEl.classList.add(hard >= 0.5 ? 'success' : 'fail');
        subLine.appendChild(hardEl);
        subLine.appendChild(el('span', 'eval-card-count', 'hard'));
        card.appendChild(subLine);
      }
    } else {
      const sr = m.success_rate;
      const mainLine = el('div', 'eval-card-main');
      const rateEl = el('div', 'eval-card-rate');
      rateEl.textContent = (sr !== undefined) ? fmt(sr * 100) + '%' : '—';
      if (sr !== undefined) rateEl.classList.add(sr >= 0.5 ? 'success' : 'fail');
      mainLine.appendChild(rateEl);
      const countEl = el('span', 'eval-card-count');
      countEl.textContent = (m.success_count !== undefined && m.num_tasks !== undefined)
        ? `${m.success_count}/${m.num_tasks}` : '';
      mainLine.appendChild(countEl);
      card.appendChild(mainLine);
    }

    evalRow.appendChild(card);
  });
  wrap.appendChild(evalRow);

  // ── Split selector (only if more than 1 split) ──
  if (state.selectedSplit === null || !ev[state.selectedSplit]) {
    state.selectedSplit = splits[0][0];
  }

  if (splits.length > 1) {
    const splitTabs = el('div', 'bench-tabs');
    splits.forEach(([name]) => {
      const tab = el('div', 'bench-tab' + (state.selectedSplit === name ? ' active' : ''), name);
      tab.onclick = () => { state.selectedSplit = name; state.selectedEvalTask = null; render(); };
      splitTabs.appendChild(tab);
    });
    wrap.appendChild(splitTabs);
  }

  // ── Task list for the active split ──
  const active = ev[state.selectedSplit];
  const tasks = (active && active.tasks) || [];
  wrap.appendChild(el('div', 'section-title', state.selectedSplit + ' — ' + tasks.length + ' tasks'));

  const timeline = el('div', 'timeline');
  tasks.forEach((task, ti) => {
    const evd = task.evaluation || {};
    const success = !!evd.success;
    const item = el('div', 'group-item ' + (success ? 'success' : 'fail'));
    const header = el('div', 'group-header');
    header.appendChild(el('span', 'group-idx', String(ti).padStart(4, '0')));
    const label = evalTaskLabel(task);
    header.appendChild(el('span', 'group-task', label));
    header.appendChild(el('span', 'group-status ' + (success ? 'success' : 'fail'), success ? 'Success' : 'Fail'));
    if (evd.num_actions !== undefined) {
      header.appendChild(el('span', 'group-rounds', evd.num_actions + ' actions'));
    }
    item.appendChild(header);

    // Show per-step admissibility dots for test tasks too
    const steps = evd.steps || [];
    if (steps.length > 0) {
      const dots = el('div', 'traj-timeline');
      steps.forEach(step => {
        dots.appendChild(el('div', 'traj-dot ' + (step.admissible ? 'ok' : 'bad')));
      });
      item.appendChild(dots);
    }

    item.onclick = () => {
      state.selectedEvalTask = ti;
      state.view = 'eval_task';
      render();
    };
    timeline.appendChild(item);
  });
  wrap.appendChild(timeline);
  return wrap;
}

function evalTaskLabel(task) {
  const tid = String(task.task_id || '');
  const slash = tid.lastIndexOf('/');
  const short = slash >= 0 ? tid.slice(slash + 1) : tid;
  return short || ('Task ' + (task.index ?? ''));
}

function renderEvalTask() {
  const run = DATA[state.selectedRun];
  const bench = run.benchmarks[state.selectedBench];
  const splitInfo = (bench.evaluation || {})[state.selectedSplit];
  if (!splitInfo) return renderBenchmark();
  const task = (splitInfo.tasks || [])[state.selectedEvalTask];
  if (!task) return renderBenchmark();

  if (!task._detailLoaded) {
    loadEvalTaskDetail(state.selectedRun, state.selectedBench, state.selectedSplit, state.selectedEvalTask);
    const main = loadingMain('Loading Task', evalTaskLabel(task));
    if (task._detailError) main.appendChild(el('div', 'empty', 'Failed to load task: ' + task._detailError));
    return main;
  }

  const main = el('div', 'main');
  const bc = el('div', 'breadcrumb');
  bc.innerHTML = '<span onclick="goBoard()">Board</span><span class="sep">/</span>' +
    '<span onclick="goBenchmark()">' + run.id + '</span><span class="sep">/</span>' +
    '<span onclick="goBenchmark()">' + bench.name + '</span><span class="sep">/</span>' +
    '<span onclick="goBenchmark()">Test · ' + state.selectedSplit + '</span><span class="sep">/</span>' +
    '<span>Task ' + state.selectedEvalTask + '</span>';
  main.appendChild(bc);
  main.appendChild(el('div', 'section-title', evalTaskLabel(task)));

  const ev = task.evaluation || {};
  const summary = (ev.success ? 'SUCCESS' : 'FAIL') +
    (ev.num_actions !== undefined ? ' · ' + ev.num_actions + ' actions' : '');
  main.appendChild(collapsible('Trajectory', summary, ev.success,
      () => renderTrajectory(task, {taskInfo: task.task_info || {}}), {defaultOpen: true}));
  return main;
}

function renderGroup() {
  const run = DATA[state.selectedRun];
  const bench = run.benchmarks[state.selectedBench];
  const group = bench.groups[state.selectedGroup];
  if (!group) return renderBenchmark();

  if (!group._detailLoaded) {
    loadGroupDetail(state.selectedRun, state.selectedBench, state.selectedGroup);
    const main = loadingMain('Loading Group', extractTaskName(group));
    if (group._detailError) main.appendChild(el('div', 'empty', 'Failed to load group: ' + group._detailError));
    return main;
  }

  const main = el('div', 'main');
  const bc = el('div', 'breadcrumb');
  bc.innerHTML = '<span onclick="goBoard()">Board</span><span class="sep">/</span><span onclick="goBenchmark()">' + run.id + '</span><span class="sep">/</span><span onclick="goBenchmark()">' + bench.name + '</span><span class="sep">/</span><span>Group ' + state.selectedGroup + '</span>';
  main.appendChild(bc);
  main.appendChild(el('div', 'section-title', 'Group ' + state.selectedGroup + ' — ' + extractTaskName(group)));

  const taskInfo = el('div', 'task-info');
  const t = group.task || {};
  if (t.type === 'alfworld') {
    taskInfo.innerHTML = '<div class="label">Goal</div><div>' + (t.goal || '—') + '</div><div style="margin-top:12px;" class="label">Task Type</div><div>' + (t.task_type || '—') + ' | ' + (t.scene || '—') + '</div><div style="margin-top:12px;" class="label">Target</div><div>' + (t.pddl_target || '—') + ' → ' + (t.pddl_parent || '—') + '</div>';
  } else if (t.type === 'spreadsheetbench') {
    taskInfo.innerHTML = '<div class="label">Instruction</div><div>' + (t.instruction || '—') + '</div><div style="margin-top:12px;" class="label">Type</div><div>' + (t.instruction_type || '—') + ' | Answer: ' + (t.answer_position || '—') + '</div>';
  } else if (t.type === 'wikitq') {
    taskInfo.innerHTML = '<div class="label">Question</div><div>' + (t.question || '—') + '</div><div style="margin-top:12px;" class="label">Table</div><div>' + (t.table_id || '—') + '</div>';
  } else if (t.type === 'searchqa') {
    taskInfo.innerHTML = '<div class="label">Question</div><div>' + (t.question || '—') + '</div><div style="margin-top:12px;" class="label">Context (preview)</div><div style="white-space:pre-wrap;font-size:11px;color:var(--fg-secondary)">' + (t.context_preview || '—') + '</div>';
  } else if (t.type === 'livemathc') {
    const tt = Array.isArray(t.theorem_type) ? t.theorem_type.join(' / ') : (t.theorem_type || '');
    taskInfo.innerHTML = '<div class="label">Question</div><div>' + (t.question || '—') + '</div><div style="margin-top:12px;" class="label">Month / Theorem Type</div><div>' + (t.month || '—') + ' · ' + (tt || '—') + ' · correct=' + (t.correct_label || '—') + '</div>';
  } else if (t.type === 'docvqa') {
    const imgBlock = t.image_path
      ? '<img src="/image?path=' + encodeURIComponent(t.image_path) + '" alt="document" loading="lazy" style="max-width:100%;max-height:520px;border:1px solid var(--line);display:block;margin-top:6px;background:#fff;">'
      : '<div style="color:var(--fg-secondary)">—</div>';
    taskInfo.innerHTML =
      '<div class="label">Question</div><div>' + (t.question || '—') + '</div>' +
      '<div style="margin-top:12px;" class="label">Document</div><div>q=' + (t.question_id || '—') + ' · doc=' + (t.doc_id || '—') + ' · topic=' + (t.topic || '—') + '</div>' +
      '<div style="margin-top:12px;" class="label">Image</div>' + imgBlock +
      '<div style="font-size:10px;color:var(--fg-secondary);margin-top:4px;">' + escapeHtml(t.image_path || '') + '</div>';
  }
  main.appendChild(taskInfo);

  // ── Per-group token usage ──
  const groupUsage = (group.summary || {}).usage;
  if (groupUsage) {
    const usagePanel = el('div', 'metrics-row');
    const totalInput = (groupUsage.student ? groupUsage.student.input_tokens || 0 : 0)
      + (groupUsage.teacher ? groupUsage.teacher.input_tokens || 0 : 0)
      + (groupUsage.critic ? groupUsage.critic.input_tokens || 0 : 0);
    const totalOutput = (groupUsage.student ? groupUsage.student.output_tokens || 0 : 0)
      + (groupUsage.teacher ? groupUsage.teacher.output_tokens || 0 : 0)
      + (groupUsage.critic ? groupUsage.critic.output_tokens || 0 : 0);
    const totalCached = (groupUsage.student ? groupUsage.student.cached_tokens || 0 : 0)
      + (groupUsage.teacher ? groupUsage.teacher.cached_tokens || 0 : 0)
      + (groupUsage.critic ? groupUsage.critic.cached_tokens || 0 : 0);
    const totalReqs = (groupUsage.student ? groupUsage.student.requests || 0 : 0)
      + (groupUsage.teacher ? groupUsage.teacher.requests || 0 : 0)
      + (groupUsage.critic ? groupUsage.critic.requests || 0 : 0);
    const usageCells = [
      ['Input Tokens', fmtK(totalInput)],
      ['Output Tokens', fmtK(totalOutput)],
      ['Cached Tokens', fmtK(totalCached)],
      ['Requests', fmt(totalReqs)],
    ];
    usageCells.forEach(([label, value]) => {
      const cell = el('div', 'metric-cell');
      cell.appendChild(el('div', 'metric-label', label));
      cell.appendChild(el('div', 'metric-value', value));
      usagePanel.appendChild(cell);
    });
    main.appendChild(usagePanel);
  }

  const groupTaskInfo = group.task || {};

  // 1. Student initial — collapsible
  if (group.student_initial) {
    const ev = group.student_initial.evaluation || {};
    const summary = (ev.success ? 'SUCCESS' : 'FAIL') +
      (ev.num_actions !== undefined ? ' · ' + ev.num_actions + ' actions' : '');
    main.appendChild(collapsible('Student Initial', summary, ev.success, () => renderTrajectory(group.student_initial, {taskInfo: groupTaskInfo})));
  }

  // 2. Teacher (if present) — collapsible
  const summaryObj = group.summary || {};
  if (group.teacher) {
    const ev = group.teacher.evaluation || {};
    const summary = (ev.success ? 'SUCCESS' : 'FAIL') +
      (ev.num_actions !== undefined ? ' · ' + ev.num_actions + ' actions' : '');
    main.appendChild(collapsible('Teacher', summary, ev.success, () => renderTrajectory(group.teacher, {taskInfo: groupTaskInfo})));
  } else if (summaryObj.teacher_failed) {
    const notice = el('div', 'task-info');
    notice.style.marginTop = '8px';
    notice.style.color = 'var(--warn)';
    notice.innerHTML = '<div class="label">Teacher</div><div>Teacher failed — self-evolution skipped.</div>';
    main.appendChild(notice);
  }

  // 3. Rounds — each round is collapsible. Inside: critic (collapsible) + student_retry (collapsible)
  if (group.rounds && group.rounds.length > 0) {
    main.appendChild(el('div', 'section-title', 'Rounds (' + group.rounds.length + ')'));
    group.rounds.forEach((round, ri) => main.appendChild(renderRound(round, ri, {taskInfo: groupTaskInfo})));
  }

  return main;
}

function collapsible(title, summaryText, statusOk, contentFactory, opts) {
  opts = opts || {};
  const wrap = el('div', 'collapsible');
  const header = el('div', 'collapsible-header');
  const left = el('div', 'collapsible-left');
  const caret = el('span', 'caret', '▶');
  left.appendChild(caret);
  left.appendChild(el('span', 'collapsible-title', title));
  header.appendChild(left);
  const right = el('div', 'collapsible-right');
  if (summaryText) {
    const tag = el('span', 'collapsible-summary');
    if (statusOk === true) tag.classList.add('ok');
    else if (statusOk === false) tag.classList.add('bad');
    tag.textContent = summaryText;
    right.appendChild(tag);
  }
  header.appendChild(right);
  wrap.appendChild(header);

  const body = el('div', 'collapsible-body');
  body.style.display = 'none';
  wrap.appendChild(body);

  let loaded = false;
  header.onclick = function() {
    const open = body.style.display === 'none';
    body.style.display = open ? 'block' : 'none';
    caret.textContent = open ? '▼' : '▶';
    if (open && !loaded) {
      body.appendChild(contentFactory());
      loaded = true;
    }
  };
  if (opts.defaultOpen) header.onclick();
  return wrap;
}

function renderCritic(critic, opts) {
  opts = opts || {};
  const taskImagePath = opts.imagePath || '';
  const container = el('div', 'traj-container');

  // ── Top status bar ──
  const bar = el('div', 'traj-summary-bar');
  const opsValid = critic.ops_valid;
  const accepted = critic.accepted;
  const validTag = el('span', 'result ' + (opsValid ? 'success' : 'fail'));
  validTag.textContent = opsValid ? 'OPS VALID' : 'OPS INVALID';
  bar.appendChild(validTag);
  const accTag = el('span', 'result ' + (accepted ? 'success' : 'fail'));
  accTag.textContent = accepted ? 'ACCEPTED' : 'NOT ACCEPTED';
  bar.appendChild(accTag);
  const ops = critic.ops || [];
  bar.appendChild(el('span', 'detail', ops.length + ' op' + (ops.length === 1 ? '' : 's')));

  const loopStats = critic.loop_stats || {};
  if (loopStats.investigation_calls !== undefined) {
    const invPill = el('span', '');
    invPill.style.cssText =
      'padding:2px 8px;border-radius:10px;font-size:10px;font-family:var(--mono);'
      + 'border:1px solid var(--info);color:var(--info);margin-left:6px;';
    invPill.textContent = loopStats.investigation_calls + ' inv / '
      + (loopStats.total_turns || '?') + ' turns';
    bar.appendChild(invPill);
  }
  if (loopStats.forced_terminate) {
    const warnPill = el('span', '');
    warnPill.style.cssText =
      'padding:2px 8px;border-radius:10px;font-size:10px;font-family:var(--mono);'
      + 'border:1px solid var(--warn);color:var(--warn);margin-left:6px;';
    warnPill.textContent = 'FORCED TERMINATE';
    bar.appendChild(warnPill);
  }
  if (loopStats.no_tool_nudges && loopStats.no_tool_nudges > 0) {
    const nudgePill = el('span', '');
    nudgePill.style.cssText =
      'padding:2px 8px;border-radius:10px;font-size:10px;font-family:var(--mono);'
      + 'border:1px solid var(--warn);color:var(--warn);margin-left:6px;';
    nudgePill.textContent = loopStats.no_tool_nudges + ' nudge'
      + (loopStats.no_tool_nudges === 1 ? '' : 's');
    bar.appendChild(nudgePill);
  }
  if (critic.ops_error) {
    bar.appendChild(el('span', 'detail', 'error: ' + critic.ops_error));
  }
  if (critic.loop_error) {
    bar.appendChild(el('span', 'detail', 'loop_error: ' + critic.loop_error));
  }
  container.appendChild(bar);

  // ── Investigations summary ──
  const investigations = critic.investigations || [];
  if (investigations.length > 0) {
    const invHeader = el('div', 'comparison-header');
    invHeader.style.marginTop = '12px';
    invHeader.textContent = 'Investigations';
    container.appendChild(invHeader);
    const invList = el('div', '');
    investigations.forEach(inv => {
      const row = el('div', '');
      row.style.cssText = 'font-family:var(--mono);font-size:11px;padding:4px 8px;'
        + 'background:rgba(9,105,218,0.04);border-left:3px solid var(--info);'
        + 'margin-bottom:2px;';
      const argText = inv.args ? JSON.stringify(inv.args) : '';
      row.textContent = '[turn ' + inv.turn + '] ' + inv.name + '(' + argText + ')'
        + '   → ' + (inv.result_bytes || 0) + 'B';
      invList.appendChild(row);
    });
    container.appendChild(invList);
  }

  // ── Parsed ops ──
  if (ops.length > 0) {
    const opsHeader = el('div', 'comparison-header');
    opsHeader.style.marginTop = '16px';
    opsHeader.textContent = 'Parsed Ops';
    container.appendChild(opsHeader);
    const opsBlock = el('div', 'code-block');
    opsBlock.textContent = JSON.stringify(ops, null, 2);
    container.appendChild(opsBlock);
  }

  // ── Candidate skill (after ops applied) ──
  if (critic.candidate_skill) {
    const skHeader = el('div', 'comparison-header');
    skHeader.style.marginTop = '16px';
    skHeader.textContent = 'Candidate Skill';
    container.appendChild(skHeader);
    const skBlock = el('div', 'code-block');
    skBlock.textContent = critic.candidate_skill;
    container.appendChild(skBlock);
  }

  // ── Full messages trajectory (system / user / assistant) ──
  const messages = critic.messages || [];
  if (messages.length > 0) {
    const msgHeader = el('div', 'comparison-header');
    msgHeader.style.marginTop = '16px';
    msgHeader.textContent = 'Full Messages (API trajectory)';
    container.appendChild(msgHeader);
    container.appendChild(renderMessages(messages, [], {imagePath: taskImagePath}));
  } else if (critic.raw) {
    const rawHeader = el('div', 'comparison-header');
    rawHeader.style.marginTop = '16px';
    rawHeader.textContent = 'Raw Output';
    container.appendChild(rawHeader);
    const rawBlock = el('div', 'code-block');
    rawBlock.textContent = critic.raw;
    container.appendChild(rawBlock);
  }

  return container;
}

function renderEnvStepsTable(envSteps) {
  const wrap = el('div', '');
  wrap.style.marginTop = '8px';

  const table = el('table', 'env-steps-table');
  const thead = el('thead', '');
  const trh = el('tr', '');
  ['#', 'Action', 'Adm.', 'Observation (after)', ''].forEach(h => {
    const th = el('th', '', h);
    trh.appendChild(th);
  });
  thead.appendChild(trh);
  table.appendChild(thead);

  const tbody = el('tbody', '');
  envSteps.forEach((step, i) => {
    const tr = el('tr', '');
    tr.appendChild(el('td', 'idx', String(i + 1).padStart(2, '0')));

    const actionTd = el('td', 'action');
    actionTd.textContent = step.action || '';
    tr.appendChild(actionTd);

    const adm = step.admissible || [];
    const admOk = adm.length > 0 ? adm.indexOf(step.action) >= 0 : null;
    const admTd = el('td', '');
    if (admOk === true) {
      admTd.innerHTML = '<span class="tag ok">ok</span>';
    } else if (admOk === false) {
      admTd.innerHTML = '<span class="tag bad">✗</span>';
    } else {
      admTd.textContent = '—';
    }
    tr.appendChild(admTd);

    const obsTd = el('td', '');
    const obs = (step.observation || '').replace(/\\s+/g, ' ').trim();
    obsTd.textContent = obs.length > 120 ? obs.slice(0, 120) + '…' : obs;
    obsTd.title = step.observation || '';
    tr.appendChild(obsTd);

    const flagsTd = el('td', '');
    let flags = '';
    if (step.won) flags += '<span class="tag ok">WON</span>';
    if (step.done && !step.won) flags += '<span class="tag info">DONE</span>';
    flagsTd.innerHTML = flags;
    tr.appendChild(flagsTd);

    tbody.appendChild(tr);
  });
  table.appendChild(tbody);
  wrap.appendChild(table);
  return wrap;
}

function renderToolCallPill(call) {
  const wrap = el('div', '');
  wrap.style.marginTop = '4px';
  const name = (call.function && call.function.name) || '';
  const args = (call.function && call.function.arguments) || '';
  const color = name === 'submit_ops'
    ? 'var(--success)'
    : (name === 'get_original_trajectories' ? 'var(--info)' : 'var(--warn)');
  const pill = el('span', '');
  pill.style.cssText =
    'display:inline-block;padding:2px 8px;border-radius:10px;border:1px solid '
    + color + ';color:' + color
    + ';font-family:var(--mono);font-size:10px;margin-right:6px;';
  pill.textContent = '→ ' + name;
  const idTag = el('span', '');
  idTag.style.cssText = 'font-family:var(--mono);font-size:9px;color:var(--fg-secondary);';
  idTag.textContent = call.id || '';
  const head = el('div', '');
  head.appendChild(pill);
  head.appendChild(idTag);
  wrap.appendChild(head);

  let pretty = args;
  try {
    pretty = JSON.stringify(JSON.parse(args), null, 2);
  } catch (e) {}
  const argsBlock = el('div', 'code-block');
  argsBlock.style.maxHeight = '260px';
  argsBlock.style.margin = '4px 0 0 0';
  argsBlock.textContent = pretty;
  wrap.appendChild(argsBlock);
  return wrap;
}

function renderMessages(messages, envSteps, opts) {
  envSteps = envSteps || [];
  opts = opts || {};
  const imagePath = opts.imagePath || '';
  const container = el('div', '');

  // For ALFWorld-style transcripts we want a stable per-turn number. The
  // old convention (alternating user/assistant after the system message)
  // breaks once tool-role messages enter the mix, so for non-alfworld
  // shapes we just drop the turn label and let the role badge stand.
  const looksAlfworld = messages.length > 0
    && messages.every(m => m.role === 'system' || m.role === 'user' || m.role === 'assistant')
    && messages.filter(m => !m.tool_calls).length === messages.length;

  messages.forEach((msg, i) => {
    const block = el('div', '');
    block.style.marginBottom = '4px';

    const header = el('div', '');
    header.style.padding = '6px 12px';
    header.style.fontFamily = 'var(--mono)';
    header.style.fontSize = '10px';

    const turnIdx = looksAlfworld
      ? (msg.role === 'system' ? -1 : Math.floor((i - 1) / 2))
      : -1;
    const env = (turnIdx >= 0 && envSteps[turnIdx]) || null;
    const turnPrefix = turnIdx >= 0
      ? '<span style="color:var(--fg-secondary)">turn ' + String(turnIdx + 1).padStart(2, '0') + '</span> '
      : '';

    if (msg.role === 'system') {
      header.style.background = 'rgba(9,105,218,0.06)';
      header.style.color = 'var(--info)';
      header.innerHTML = '<b>system</b>';
    } else if (msg.role === 'user') {
      header.style.background = 'rgba(0,0,0,0.02)';
      header.innerHTML = turnPrefix + '<b>user</b>';
    } else if (msg.role === 'assistant') {
      header.style.background = 'rgba(0,0,0,0.02)';
      let suffix = '';
      if (env) {
        const adm = env.admissible || [];
        if (adm.length > 0) {
          suffix += adm.indexOf(env.action) >= 0
            ? ' <span style="color:var(--success)">✓</span>'
            : ' <span style="color:var(--fail)">✗</span>';
        }
        if (env.won) suffix += ' <span style="color:var(--success)">WON</span>';
        else if (env.done) suffix += ' <span style="color:var(--info)">DONE</span>';
      }
      const toolCalls = msg.tool_calls || [];
      if (toolCalls.length > 0) {
        const names = toolCalls.map(c => (c.function && c.function.name) || '?').join(', ');
        suffix += ' <span style="color:var(--fg-secondary)">[tool_call: ' + escapeHtml(names) + ']</span>';
      }
      header.innerHTML = turnPrefix + '<b>assistant</b>' + suffix;
    } else if (msg.role === 'tool') {
      header.style.background = 'rgba(46,160,67,0.06)';
      header.style.color = 'var(--success)';
      const toolName = msg.name || '?';
      const callId = msg.tool_call_id || '';
      header.innerHTML = '<b>tool</b> ← ' + escapeHtml(toolName)
        + ' <span style="color:var(--fg-secondary)">(' + escapeHtml(callId) + ')</span>';
    } else {
      header.innerHTML = '<b>' + escapeHtml(msg.role || '?') + '</b>';
    }
    block.appendChild(header);

    const body = el('div', '');
    body.style.borderLeft = msg.role === 'assistant'
      ? '3px solid var(--success)'
      : (msg.role === 'tool' ? '3px solid var(--info)' : '3px solid var(--line)');
    body.style.marginLeft = '12px';
    body.style.padding = '8px 12px';

    // Tool-result body: pretty-print JSON when possible.
    if (msg.role === 'tool') {
      const out = el('div', 'code-block');
      out.style.maxHeight = '320px';
      out.style.margin = '0';
      let pretty = msg.content || '';
      try { pretty = JSON.stringify(JSON.parse(pretty), null, 2); } catch (e) {}
      out.textContent = pretty;
      body.appendChild(out);
    } else {
      // msg.content can be a string OR a multimodal parts list
      // (e.g. docvqa: [{type:'text', ...}, {type:'image_url', image_url:'[omitted for log]'}]).
      const parts = Array.isArray(msg.content)
        ? msg.content
        : (msg.content !== undefined && msg.content !== null && msg.content !== ''
            ? [{type: 'text', text: msg.content}]
            : []);
      parts.forEach(part => {
        if (part && part.type === 'image_url') {
          // The on-disk format slims image_url to "[omitted for log]" to
          // avoid base64 bloat. Render the task's source image instead,
          // unless the saved value is itself a usable URL/data URI.
          let src = '';
          const iu = part.image_url;
          if (typeof iu === 'string' && (iu.startsWith('data:') || iu.startsWith('http://') || iu.startsWith('https://'))) {
            src = iu;
          } else if (iu && typeof iu === 'object' && typeof iu.url === 'string') {
            src = iu.url;
          } else if (imagePath) {
            src = '/image?path=' + encodeURIComponent(imagePath);
          }
          if (src) {
            const img = el('img', '');
            img.src = src;
            img.alt = 'image';
            img.loading = 'lazy';
            img.style.maxWidth = '100%';
            img.style.maxHeight = '480px';
            img.style.border = '1px solid var(--line)';
            img.style.display = 'block';
            img.style.background = '#fff';
            body.appendChild(img);
          } else {
            const note = el('div', '');
            note.style.fontSize = '11px';
            note.style.color = 'var(--fg-secondary)';
            note.textContent = '[image_url omitted]';
            body.appendChild(note);
          }
        } else {
          const text = (part && part.text !== undefined)
            ? String(part.text)
            : (typeof part === 'string' ? part : JSON.stringify(part));
          if (text) {
            const content = el('div', 'code-block');
            content.style.maxHeight = '320px';
            content.style.margin = '0';
            content.textContent = text;
            body.appendChild(content);
          }
        }
      });
      const toolCalls = msg.tool_calls || [];
      toolCalls.forEach(call => body.appendChild(renderToolCallPill(call)));
    }
    block.appendChild(body);
    container.appendChild(block);
  });
  return container;
}

function extractQuestionFromPrompt(prompt) {
  if (!prompt) return '';
  const goalMatch = prompt.match(/Goal:\\s*(.+?)(?:\\n|$)/);
  if (goalMatch) return goalMatch[1].trim();
  const qMatch = prompt.match(/Question:\\s*(.+?)(?:\\n\\n|$)/);
  if (qMatch) return qMatch[1].trim();
  const instMatch = prompt.match(/### instruction\\s*\\n(.+?)(?:\\n\\n|$)/);
  if (instMatch) return instMatch[1].trim();
  return '';
}

function renderTrajectoryInfo(traj, taskInfo, mode) {
  const evalData = traj.evaluation || {};
  const bar = el('div', 'traj-info-bar');

  // Question
  let questionText = '';
  if (taskInfo) {
    if (taskInfo.type === 'alfworld') questionText = taskInfo.goal || taskInfo.task_type || '';
    else if (taskInfo.type === 'spreadsheetbench') questionText = taskInfo.instruction || '';
    else if (taskInfo.type === 'wikitq') questionText = taskInfo.question || '';
    else if (taskInfo.type === 'searchqa') questionText = taskInfo.question || '';
    else if (taskInfo.type === 'livemathc') questionText = taskInfo.question || '';
    else if (taskInfo.type === 'docvqa') questionText = taskInfo.question || '';
  }
  if (!questionText && traj.prompt) {
    questionText = extractQuestionFromPrompt(traj.prompt);
  }
  if (questionText) {
    const qRow = el('div', 'traj-info-row');
    qRow.appendChild(el('span', 'traj-info-label', 'Question'));
    qRow.appendChild(el('span', 'traj-info-value', questionText));
    bar.appendChild(qRow);
  }

  // Ground Truth
  let gtText = '';
  if (mode.includes('wikitq') || mode.includes('denotation')) {
    const gold = evalData.gold || [];
    gtText = gold.length > 0 ? gold.map(g => String(g)).join(', ') : '';
  } else if (mode.includes('spreadsheet')) {
    const cases = evalData.cases || [];
    if (cases.length > 0 && cases[0].answer_preview) {
      gtText = cases[0].answer_preview;
    }
  }
  if (gtText) {
    const gtRow = el('div', 'traj-info-row');
    gtRow.appendChild(el('span', 'traj-info-label', 'Ground Truth'));
    const gtVal = el('span', 'traj-info-value');
    gtVal.textContent = gtText;
    gtRow.appendChild(gtVal);
    bar.appendChild(gtRow);
  }

  // Answer
  let answerText = '';
  if (mode.includes('wikitq') || mode.includes('denotation')) {
    answerText = String(evalData.prediction || traj.final_answer || '');
  } else if (mode.includes('spreadsheet')) {
    answerText = evalData.solution_code || traj.final_answer || '';
  } else if (traj.final_answer) {
    answerText = traj.final_answer;
  } else if (traj.raw) {
    answerText = traj.raw;
  }
  if (answerText) {
    const aRow = el('div', 'traj-info-row');
    aRow.appendChild(el('span', 'traj-info-label', 'Answer'));
    const aVal = el('span', 'traj-info-value');
    aVal.textContent = answerText;
    aRow.appendChild(aVal);
    bar.appendChild(aRow);
  }

  return bar;
}

function renderTrajectory(traj, opts) {
  opts = opts || {};
  const container = el('div', 'traj-container');
  const evalData = traj.evaluation || {};
  const mode = evalData.mode || '';
  const taskInfo = opts.taskInfo || null;

  // ── Summary bar ──
  const bar = el('div', 'traj-summary-bar');
  const success = evalData.success !== undefined ? evalData.success : false;
  const result = el('span', 'result ' + (success ? 'success' : 'fail'));
  result.textContent = success ? 'SUCCESS' : 'FAILURE';
  bar.appendChild(result);
  if (evalData.num_actions !== undefined) {
    bar.appendChild(el('span', 'detail', fmt(evalData.num_actions) + ' actions'));
  }
  if (evalData.mode) {
    bar.appendChild(el('span', 'detail', evalData.mode));
  }
  container.appendChild(bar);

  // ── Question / Ground Truth / Answer info bar ──
  if (taskInfo || traj.prompt) {
    container.appendChild(renderTrajectoryInfo(traj, taskInfo, mode));
  }

  let messages = traj.messages || [];
  const envSteps = traj.env_steps || [];
  if (messages.length === 0 && (traj.prompt || traj.raw)) {
    messages = [];
    if (traj.prompt) messages.push({role: 'user', content: traj.prompt});
    if (traj.raw) messages.push({role: 'assistant', content: traj.raw});
  } else if (traj.raw && messages.length > 0) {
    // Backwards-compat: rollouts saved before the assistant turn was
    // appended into `messages` only stored the input. If the trajectory
    // has a `raw` reply but the last message isn't already that reply,
    // synthesize it so the viewer shows the complete conversation.
    const last = messages[messages.length - 1];
    if (!last || last.role !== 'assistant') {
      messages = messages.concat([{role: 'assistant', content: traj.raw}]);
    }
  }
  const taskImagePath = (taskInfo && taskInfo.image_path) ? taskInfo.image_path : '';

  // ── ALFWorld: structured action table first, raw messages collapsed underneath ──
  if (envSteps.length > 0) {
    container.appendChild(renderEnvStepsTable(envSteps));
    if (messages.length > 0) {
      container.appendChild(collapsible(
        'Raw messages',
        messages.length + ' messages',
        null,
        () => renderMessages(messages, envSteps, {imagePath: taskImagePath})
      ));
    }
  }
  // ── SpreadsheetBench: case results ──
  else if (mode.includes('spreadsheet')) {
    if (evalData.test_case_results && evalData.test_case_results.length > 0) {
      const caseGrid = el('div', 'case-grid');
      evalData.test_case_results.forEach((pass, i) => {
        const dot = el('div', 'case-dot ' + (pass ? 'ok' : 'fail'));
        dot.textContent = String(i + 1);
        dot.title = 'Case ' + (i + 1) + ': ' + (pass ? 'PASS' : 'FAIL');
        caseGrid.appendChild(dot);
      });
      container.appendChild(caseGrid);

      if (evalData.cases && evalData.cases.length > 0) {
        container.appendChild(collapsible(
          'Case Details',
          evalData.cases.length + ' cases',
          null,
          () => {
            const body = el('div', '');
            evalData.cases.forEach((c, i) => {
              const cd = el('div', 'case-detail');
              const header = el('div', 'case-detail-header');
              header.textContent = 'Case ' + (i + 1) + (c.success ? ' ✓ PASS' : ' ✗ FAIL');
              if (!c.success && c.error) header.textContent += ' | ' + c.error;
              cd.appendChild(header);
              if (c.execution) {
                const execInfo = el('div', '');
                execInfo.style.fontSize = '10px';
                execInfo.style.color = 'var(--fg-secondary)';
                execInfo.style.marginBottom = '6px';
                execInfo.textContent = 'exit: ' + c.execution.returncode;
                if (c.execution.stderr) {
                  const errBlock = el('div', 'code-block');
                  errBlock.style.maxHeight = '120px';
                  errBlock.style.marginTop = '4px';
                  errBlock.textContent = c.execution.stderr;
                  cd.appendChild(errBlock);
                }
                cd.appendChild(execInfo);
              }
              body.appendChild(cd);
            });
            return body;
          }
        ));
      }
    }

    if (evalData.solution_code) {
      container.appendChild(collapsible('Solution Code', '', null, () => {
        const block = el('div', 'code-block');
        block.textContent = evalData.solution_code;
        return block;
      }));
    }
    if (messages.length > 0) {
      container.appendChild(collapsible(
        'Raw messages',
        messages.length + ' messages',
        null,
        () => renderMessages(messages, [], {imagePath: taskImagePath})
      ));
    }
  }
  // ── WikiTQ: prediction vs gold ──
  else if (mode.includes('denotation') || mode.includes('wikitq')) {
    const comp = el('div', 'pred-gold-comp');
    const predDiv = el('div', '');
    predDiv.innerHTML = '<div class="label">Prediction</div><div>' + escapeHtml(String(evalData.prediction || '—')) + '</div>';
    comp.appendChild(predDiv);
    const goldDiv = el('div', '');
    const gold = evalData.gold || [];
    goldDiv.innerHTML = '<div class="label">Gold (' + gold.length + ')</div><div>' + gold.map(g => escapeHtml(String(g))).join('<br>') + '</div>';
    comp.appendChild(goldDiv);
    container.appendChild(comp);

    if (messages.length > 0) {
      container.appendChild(collapsible(
        'Raw messages',
        messages.length + ' messages',
        null,
        () => renderMessages(messages, [], {imagePath: taskImagePath})
      ));
    }
  }
  // ── Fallback: render messages directly, or raw text ──
  else if (messages.length > 0) {
    container.appendChild(renderMessages(messages, [], {imagePath: taskImagePath}));
  } else {
    const raw = traj.final_answer || traj.raw || '';
    if (raw) {
      const block = el('div', 'code-block');
      block.textContent = raw;
      container.appendChild(block);
    }
  }

  return container;
}

function renderTag(cond, yesText, noText, yesCls, noCls) {
  if (cond) return '<span class="tag ' + yesCls + '">' + yesText + '</span>';
  if (noCls) return '<span class="tag ' + noCls + '">' + noText + '</span>';
  return '<span style="color:var(--fg-secondary)">' + noText + '</span>';
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

function renderRound(round, idx, opts) {
  opts = opts || {};
  const critic = round.critic || null;
  const retry = round.student_retry || null;
  const retryEv = (retry && retry.evaluation) || {};
  const ops = (critic && critic.ops) || [];
  const taskInfo = opts.taskInfo || null;
  const taskImagePath = (taskInfo && taskInfo.image_path) ? taskInfo.image_path : '';

  // Summary on the round header — include first op title so multi-round groups
  // are skimmable without expanding each round.
  const parts = [];
  if (critic) {
    parts.push(ops.length + ' op' + (ops.length === 1 ? '' : 's'));
    parts.push(critic.accepted ? 'accepted' : (critic.ops_valid ? 'rejected' : 'invalid'));
    const ls = critic.loop_stats || {};
    if (ls.investigation_calls && ls.investigation_calls > 0) {
      parts.push(ls.investigation_calls + ' inv');
    }
    if (ls.forced_terminate) {
      parts.push('forced');
    }
    if (ops.length > 0) {
      const titles = ops.map(o => {
        const sigil = o.op === 'add' ? '+' : o.op === 'update' ? '~' : o.op === 'delete' ? '−' : '?';
        const title = (o.rule && o.rule.title) || ('rule ' + (o.id || '?'));
        return sigil + ' ' + title;
      });
      let joined = titles.join(' · ');
      if (joined.length > 110) joined = joined.slice(0, 107) + '…';
      parts.push(joined);
    }
  }
  if (retry) parts.push('retry: ' + (retryEv.success ? 'SUCCESS' : 'FAIL'));
  const status = retry ? !!retryEv.success : null;
  const summary = parts.join(' · ');

  return collapsible('Round ' + String(idx).padStart(2, '0'), summary, status, () => {
    const body = el('div', 'round-body');

    // ── Proposed Changes (rule cards) — shown directly so each round's diff
    //    is visible without diving into the nested Critic panel.
    if (ops.length > 0) {
      const header = el('div', 'comparison-header');
      header.textContent = 'Proposed Changes (' + ops.length + ')';
      body.appendChild(header);
      ops.forEach(op => {
        const rule = op.rule || { id: op.id, title: '(rule ' + op.id + ')', content: '', why: '' };
        body.appendChild(renderRuleCard(rule, op.op, op));
      });
    } else if (critic) {
      const empty = el('div', 'skill-empty');
      empty.textContent = critic.ops_error
        ? '(no ops parsed — ' + critic.ops_error + ')'
        : '(no ops proposed this round)';
      body.appendChild(empty);
    }

    if (critic) {
      const cv = critic.ops_valid;
      const cAccepted = critic.accepted;
      const ls = critic.loop_stats || {};
      const invPart = ls.investigation_calls && ls.investigation_calls > 0
        ? ' · ' + ls.investigation_calls + ' inv'
        : '';
      const forcedPart = ls.forced_terminate ? ' · forced' : '';
      const criticSummary = (cv ? 'OPS VALID' : 'OPS INVALID') + ' · ' + (cAccepted ? 'ACCEPTED' : 'NOT ACCEPTED') +
        ' · ' + ops.length + ' op' + (ops.length === 1 ? '' : 's') + invPart + forcedPart;
      body.appendChild(collapsible('Critic', criticSummary, cAccepted, () => renderCritic(critic, {imagePath: taskImagePath})));
    }
    if (retry) {
      const rs = (retryEv.success ? 'SUCCESS' : 'FAIL') +
        (retryEv.num_actions !== undefined ? ' · ' + retryEv.num_actions + ' actions' : '');
      body.appendChild(collapsible('Student Retry', rs, !!retryEv.success, () => renderTrajectory(retry, opts)));
    }
    return body;
  });
}

function goBoard() { state.view = 'board'; state.selectedRun = null; state.selectedEvalTask = null; render(); }
function goBenchmark() { state.view = 'benchmark'; state.selectedEvalTask = null; render(); }

function render() {
  const app = document.getElementById('app');
  app.innerHTML = '';
  app.appendChild(renderSidebar());
  if (state.view === 'benchmark') app.appendChild(renderBenchmark());
  else if (state.view === 'group') app.appendChild(renderGroup());
  else if (state.view === 'eval_task') app.appendChild(renderEvalTask());
  else app.appendChild(renderBoard());  // 'board' or any unknown view → board
}

async function loadData() {
  try {
    const res = await fetch('/data');
    DATA = await res.json();
    render();
  } catch (e) {
    document.getElementById('app').innerHTML = '<div class="empty">Failed to load data: ' + e.message + '</div>';
  }
}

loadData();
</script>
</body>
</html>
'''


# ═══════════════════════════════════════════════════════════════════════════════
# HTTP Handler
# ═══════════════════════════════════════════════════════════════════════════════

class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # silence logs

    def _write_bytes(self, payload: bytes):
        try:
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _write_json(self, data, status: int = 200):
        payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self._write_bytes(payload)

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self._write_bytes(HTML_PAGE.encode("utf-8"))
        elif self.path == "/data":
            data = scan_results()
            self._write_json(data)
        elif self.path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
        elif self.path.startswith("/group"):
            self._serve_group()
        elif self.path.startswith("/eval-task"):
            self._serve_eval_task()
        elif self.path.startswith("/image"):
            self._serve_image()
        else:
            self.send_response(404)
            self.end_headers()

    def _serve_group(self):
        qs = parse_qs(urlparse(self.path).query)
        bench_dir = _run_bench_dir((qs.get("run") or [""])[0], (qs.get("bench") or [""])[0])
        group_id = (qs.get("group") or [""])[0]
        if bench_dir is None:
            self._write_json({"error": "benchmark not found"}, status=404)
            return
        groups_dir = bench_dir / "evolution" / "groups"
        group_dir = _child_dir(groups_dir, group_id)
        if group_dir is None:
            self._write_json({"error": "group not found"}, status=404)
            return
        detail = scan_group_detail(group_dir)
        if detail is None:
            self._write_json({"error": "group not found"}, status=404)
            return
        self._write_json(detail)

    def _serve_eval_task(self):
        qs = parse_qs(urlparse(self.path).query)
        bench_name = (qs.get("bench") or [""])[0]
        bench_dir = _run_bench_dir((qs.get("run") or [""])[0], bench_name)
        split_name = (qs.get("split") or [""])[0]
        index_raw = (qs.get("index") or [""])[0]
        try:
            task_index = int(index_raw)
        except ValueError:
            self._write_json({"error": "invalid task index"}, status=400)
            return
        if bench_dir is None:
            self._write_json({"error": "benchmark not found"}, status=404)
            return
        split_dir = _child_dir(bench_dir / "evaluation", split_name)
        if split_dir is None:
            self._write_json({"error": "split not found"}, status=404)
            return
        detail = scan_eval_task_detail(split_dir, task_index, bench_name)
        if detail is None:
            self._write_json({"error": "task not found"}, status=404)
            return
        self._write_json(detail)

    def _serve_image(self):
        qs = parse_qs(urlparse(self.path).query)
        raw = (qs.get("path") or [""])[0]
        if not raw:
            self.send_response(400)
            self.end_headers()
            return
        # Resolve relative paths against the repo root; absolute paths are
        # checked as-is. Either way we then verify the resolved target is
        # inside one of the whitelisted image roots.
        target = Path(raw)
        if not target.is_absolute():
            target = _REPO_ROOT / target
        try:
            resolved = target.resolve()
        except (OSError, RuntimeError):
            self.send_response(400)
            self.end_headers()
            return
        if not any(_path_is_within(resolved, root) for root in IMAGE_ALLOWED_ROOTS):
            self.send_response(403)
            self.end_headers()
            return
        if not resolved.is_file():
            self.send_response(404)
            self.end_headers()
            return
        ctype = _IMAGE_CONTENT_TYPES.get(resolved.suffix.lower(), "application/octet-stream")
        try:
            payload = resolved.read_bytes()
        except OSError:
            self.send_response(500)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "public, max-age=3600")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self._write_bytes(payload)


def main():
    server = HTTPServer(("0.0.0.0", PORT), Handler)
    print(f"PACT Board serving at http://localhost:{PORT}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
