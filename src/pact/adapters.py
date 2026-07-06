from __future__ import annotations

import json
import random
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Any

from .io import read_jsonl, read_yaml
from .llm import LLMClient, parse_final_answer
from .prompts import rollout_prompt
from .spreadsheet_official import spreadsheet_preview


ROOT = Path(__file__).resolve().parents[2]


class BenchmarkAdapter(ABC):
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.config = read_yaml(config_path)
        self.name = self.config["name"]

    @property
    def initial_skill_path(self) -> Path:
        return ROOT / self.config["setup"]["initial_skill_path"]

    @abstractmethod
    def load_tasks(self, split: str, limit: int | None = None) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def render_task(self, task: dict[str, Any], output_dir: Path | None = None) -> str:
        raise NotImplementedError

    @abstractmethod
    def evaluate(
        self,
        task: dict[str, Any],
        final_answer: str,
        judge: LLMClient | None = None,
        trajectory: dict[str, Any] | None = None,
        output_dir: Path | None = None,
    ) -> dict[str, Any]:
        raise NotImplementedError

    def build_messages(
        self,
        task: dict[str, Any],
        skill_text_for_agent: str,
        output_dir: Path | None = None,
    ) -> list[dict[str, Any]]:
        """Return the message list sent to the LLM for this task.

        Default implementation preserves the legacy single-user-message path:
        render_task() text + rollout_prompt() wrapping with the skill block.
        Tasks whose prompt assembly produces a distinct (system, user) pair —
        or a multimodal content list — override this method directly.
        """
        task_text = self.render_task(task, output_dir=output_dir)
        user = rollout_prompt(self.name, "agent", task_text, skill_text_for_agent)
        return [{"role": "user", "content": user}]

    def extract_answer(self, raw_text: str) -> str:
        """Pull the final answer out of the LLM's raw response.

        Default implementation uses the framework's `FINAL_ANSWER:` convention
        (parse_final_answer). Tasks that ship a different output protocol
        (e.g. `<answer>...</answer>` for the SkillOpt-aligned tasks) override
        this method.
        """
        return parse_final_answer(raw_text)

    def critic_attachments(self, task: dict[str, Any]) -> list[dict[str, Any]] | None:
        """Return multimodal content parts (OpenAI chat format) to splice
        into the critic's user message alongside the text. Default: None
        (text-only). Multimodal benchmarks (e.g. docvqa) override this so
        the critic sees the same image the agent saw — without it, the
        critic is blind-judging from the agent's reply alone.
        """
        return None


class JsonlAdapter(BenchmarkAdapter):
    def split_file(self, split: str) -> Path:
        split_cfg = self.config["splits"][split]
        if "split_file" not in split_cfg:
            raise ValueError(f"{self.name}.{split} has no split_file")
        return ROOT / split_cfg["split_file"]

    def _sampling_seed(self, split: str) -> int:
        split_cfg = self.config["splits"][split]
        if "sampling_seed" in split_cfg:
            return int(split_cfg["sampling_seed"])
        return int(self.config.get("sampling_seed", 42))

    def _stratify_key(self, split: str) -> str | None:
        split_cfg = self.config["splits"][split]
        return split_cfg.get("stratify_by") or self.config.get("stratify_by") or None

    def _subsample(self, rows: list[dict[str, Any]], split: str, limit: int | None) -> list[dict[str, Any]]:
        """Deterministic subsampling shared by all JSONL-backed benchmarks.

        - limit is None or limit >= len(rows): return rows unchanged.
        - stratify_by set: proportional stratified sample on rows[stratify_by].
        - else: seeded full-shuffle then take the first `limit`.

        Cross-call determinism: same (split, limit, seed) yields the same items.
        Two calls with the same arguments return the same id list — verified
        in the Phase 2 reproducibility check.
        """
        if limit is None or limit >= len(rows):
            return rows
        seed = self._sampling_seed(split)
        stratify_by = self._stratify_key(split)
        if stratify_by:
            keys = [str(r.get(stratify_by, "")) for r in rows]
            chosen = _stratified_proportional_sample(rows, keys, limit, seed)
            chosen.sort()  # stable ordering matches what alfworld does for resume keys
            return [rows[i] for i in chosen]
        shuffled = list(rows)
        random.Random(seed).shuffle(shuffled)
        return shuffled[:limit]

    def load_tasks(self, split: str, limit: int | None = None) -> list[dict[str, Any]]:
        rows = read_jsonl(self.split_file(split))
        return self._subsample(rows, split, limit)


class SpreadsheetBenchAdapter(JsonlAdapter):
    def render_task(self, task: dict[str, Any], output_dir: Path | None = None) -> str:
        root = Path(task.get("dataset_root", ""))
        sheet_dir = root / task.get("spreadsheet_path", "")
        task_id = str(task["id"])

        input_file = f"1_{task_id}_input.xlsx"
        input_path = sheet_dir / input_file

        preview = spreadsheet_preview(input_path) if input_path.exists() else ""
        instruction = task.get("instruction", "")
        instruction_type = task.get("instruction_type", "")
        answer_position = task.get("answer_position", "")

        parts = [f"# Instruction\n{instruction}"]
        if instruction_type:
            parts.append(f"Instruction type: {instruction_type}")
        if answer_position:
            parts.append(f"Expected answer position: {answer_position}")
        if preview:
            parts.append(f"# Input spreadsheet preview\n{preview}")
        parts.append(
            "Write a Python script that reads from INPUT_PATH, applies the instruction, "
            "and writes the modified workbook to OUTPUT_PATH."
        )
        return "\n\n".join(parts)

    def evaluate(
        self,
        task: dict[str, Any],
        final_answer: str,
        judge: LLMClient | None = None,
        trajectory: dict[str, Any] | None = None,
        output_dir: Path | None = None,
    ) -> dict[str, Any]:
        from .spreadsheet_official import run_solution_on_cases

        if trajectory is None:
            raise ValueError("SpreadsheetBench official evaluation requires the full trajectory.")
        if output_dir is None:
            raise ValueError("SpreadsheetBench official evaluation requires an output_dir.")
        result = run_solution_on_cases(task, trajectory["raw"], output_dir)
        result["prediction"] = final_answer
        result["gold"] = "official SpreadsheetBench answer workbooks"
        return result


_DEFAULT_ALFWORLD_TASK_TYPE_PREFIXES = (
    "look_at_obj_in_light",
    "pick_and_place_with_movable_recep",
    "pick_and_place_simple",
    "pick_clean_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "pick_two_obj_and_place",
)


def _extract_alfworld_task_type(path: Path, prefixes: tuple[str, ...]) -> str:
    """Return the longest matching ALFWorld task-type prefix from the trial path.

    ALFWorld trial paths look like
      <split_root>/<task_type>-<obj>-<...>-<scene>/<trial_id>/traj_data.json
    so the task-type folder is `path.parent.parent.name`.
    """
    folder = path.parent.parent.name
    for prefix in prefixes:  # prefixes must be sorted longest-first
        if folder.startswith(prefix):
            return prefix
    return "unknown"


def _stratified_proportional_sample(
    items: list[Any],
    keys: list[str],
    limit: int,
    seed: int,
) -> list[int]:
    """Pick `limit` indices into `items` so the type distribution matches `keys`.

    Returns the chosen indices (not items themselves). Deterministic given seed.
    Allocation:
      - floor(limit * count_t / total) base slots per type;
      - remaining slots go to the types with the largest fractional remainders,
        with seeded random tiebreak.
    Each type's items are independently shuffled with the same seed so the
    chosen subset is also random within type.
    """
    total = len(items)
    if limit >= total:
        return list(range(total))

    by_type: dict[str, list[int]] = defaultdict(list)
    for idx, key in enumerate(keys):
        by_type[key].append(idx)

    rng = random.Random(seed)
    type_order = sorted(by_type.keys())
    base_alloc: dict[str, int] = {}
    remainders: list[tuple[float, float, str]] = []
    for t in type_order:
        n = len(by_type[t])
        exact = limit * n / total
        base = int(exact)  # floor
        # Cap at the number of items actually available for this type.
        base = min(base, n)
        base_alloc[t] = base
        remainders.append((exact - base, rng.random(), t))

    leftover = limit - sum(base_alloc.values())
    # Largest fractional remainder first; seeded random tiebreak.
    remainders.sort(key=lambda x: (-x[0], x[1]))
    for _, _, t in remainders:
        if leftover <= 0:
            break
        if base_alloc[t] < len(by_type[t]):
            base_alloc[t] += 1
            leftover -= 1

    chosen: list[int] = []
    for t in type_order:
        bucket = list(by_type[t])
        rng.shuffle(bucket)
        chosen.extend(bucket[: base_alloc[t]])
    return chosen


class ALFWorldAdapter(BenchmarkAdapter):
    def load_tasks(self, split: str, limit: int | None = None) -> list[dict[str, Any]]:
        if limit is not None and limit <= 0:
            return []
        data_root = ROOT / self.config["setup"]["data_root"]
        raw_root = data_root / "raw"

        split_cfg = self.config["splits"][split]
        manifest_file = split_cfg.get("manifest")

        if manifest_file:
            manifest_path = ROOT / manifest_file
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            candidates: list[tuple[Path, Path, dict[str, Any], str]] = []
            for entry in manifest:
                game_file = raw_root / entry["gamefile"]
                traj_file = game_file.with_name("traj_data.json")
                if not game_file.exists():
                    print(f"[alfworld] warning: manifest game not found: {game_file}")
                    continue
                try:
                    data = json.loads(traj_file.read_text(encoding="utf-8"))
                except Exception:
                    continue
                data["_path"] = str(traj_file)
                data["_game_file"] = str(game_file)
                candidates.append((traj_file, game_file, data, entry.get("task_type", "")))
        else:
            source = split_cfg["source"]
            root = data_root / source
            files = sorted(root.rglob("traj_data.json")) if root.exists() else []
            sampling_cfg = self.config.get("task_sampling") or {}
            prefixes = tuple(
                sampling_cfg.get("task_type_prefixes")
                or _DEFAULT_ALFWORLD_TASK_TYPE_PREFIXES
            )
            candidates = []
            for path in files:
                game_file = path.with_name("game.tw-pddl")
                if not game_file.exists():
                    continue
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    continue
                data["_path"] = str(path)
                data["_game_file"] = str(game_file)
                candidates.append((path, game_file, data, _extract_alfworld_task_type(path, prefixes)))

        total = len(candidates)
        if total == 0:
            return []

        sampling_cfg = self.config.get("task_sampling") or {}
        strategy = sampling_cfg.get("strategy", "stratified_proportional")
        seed = int(sampling_cfg.get("seed", 42))

        task_types = [c[3] for c in candidates]
        if limit is None or limit >= total:
            chosen_indices = list(range(total))
        elif strategy == "first_n":
            chosen_indices = list(range(limit))
        else:
            chosen_indices = _stratified_proportional_sample(
                candidates, task_types, limit, seed
            )

        chosen_indices.sort(key=lambda i: str(candidates[i][0]))

        tasks: list[dict[str, Any]] = []
        chosen_type_counts: dict[str, int] = defaultdict(int)
        for i in chosen_indices:
            tasks.append(candidates[i][2])
            chosen_type_counts[task_types[i]] += 1

        total_type_counts: dict[str, int] = defaultdict(int)
        for t in task_types:
            total_type_counts[t] += 1
        used_strategy = (
            "all" if (limit is None or limit >= total) else strategy
        )
        breakdown = ", ".join(
            f"{t}={chosen_type_counts.get(t, 0)}/{total_type_counts[t]}"
            for t in sorted(total_type_counts)
        )
        print(
            f"[alfworld] split={split} limit={limit} total={total} "
            f"sampled={len(tasks)} strategy={used_strategy} seed={seed} "
            f"by_task_type: {breakdown}"
        )
        return tasks

    def render_task(self, task: dict[str, Any], output_dir: Path | None = None) -> str:
        """Return a task description for ALFWorld.

        Note: ALFWorld uses multi-turn chat. The system prompt contains
        skill + task, and each turn sends only the raw observation.
        This method provides the static task metadata only.
        """
        ann = task.get("turk_annotations", {}).get("anns", [{}])[0]
        goal = task.get("goal") or task.get("task_desc") or ann.get("task_desc", "")
        task_type = task.get("task_type", "")
        return (
            f"Task type: {task_type}\n"
            f"Goal: {goal}"
        )

    def evaluate(
        self,
        task: dict[str, Any],
        final_answer: str,
        judge: LLMClient | None = None,
        trajectory: dict[str, Any] | None = None,
        output_dir: Path | None = None,
    ) -> dict[str, Any]:
        env_steps = (trajectory or {}).get("env_steps", [])
        won = env_steps[-1]["won"] if env_steps else False

        def is_admissible_step(step: dict[str, Any]) -> bool:
            if "admissible_action" in step:
                return bool(step["admissible_action"])
            admissible = step.get("admissible", [])
            return step["action"] in admissible if isinstance(admissible, list) else bool(admissible)

        eval_steps = [
            {
                "action": step["action"],
                "admissible": is_admissible_step(step),
                "observation": step.get("observation", ""),
                "won": step["won"],
                "done": step["done"],
                **({"stop_reason": step["stop_reason"]} if step.get("stop_reason") else {}),
            }
            for step in env_steps
        ]
        return {
            "success": won,
            "mode": "official_alfworld_textworld",
            "prediction": final_answer,
            "num_actions": len(env_steps),
            "executed_steps": len(eval_steps),
            **({"stop_reason": (trajectory or {}).get("stop_reason")} if (trajectory or {}).get("stop_reason") else {}),
            "steps": eval_steps,
            "game_file": str(Path(str(task.get("_game_file") or ""))),
        }


class SearchQAAdapter(JsonlAdapter):
    """SearchQA: single-turn document-QA. Reuses SkillOpt's prompt assembly
    (system + user) and SQuAD-style EM/F1 scoring."""

    def render_task(self, task: dict[str, Any], output_dir: Path | None = None) -> str:
        # Used by the critic and on-disk logs only — the actual prompt the agent
        # sees is built in build_messages().
        from .searchqa_official import _build_user
        return _build_user(task["question"], task["context"])

    def build_messages(
        self,
        task: dict[str, Any],
        skill_text_for_agent: str,
        output_dir: Path | None = None,
    ) -> list[dict[str, Any]]:
        from .searchqa_official import _build_system, _build_user
        system = _build_system(skill_text_for_agent)
        user = _build_user(task["question"], task["context"])
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def extract_answer(self, raw_text: str) -> str:
        from .searchqa_official import extract_answer
        return extract_answer(raw_text)

    def evaluate(
        self,
        task: dict[str, Any],
        final_answer: str,
        judge: LLMClient | None = None,
        trajectory: dict[str, Any] | None = None,
        output_dir: Path | None = None,
    ) -> dict[str, Any]:
        from .searchqa_official import evaluate as _eval
        # final_answer has already been extracted by extract_answer(); feed it
        # directly to the metric so the evaluator's own <answer>-stripping is
        # idempotent.
        metrics = _eval(final_answer, task.get("answers", []))
        return {
            "success": bool(metrics["em"]),
            "mode": "searchqa_squad",
            "prediction": final_answer,
            "gold": task.get("answers", []),
            **metrics,
        }


class LiveMathCAdapter(JsonlAdapter):
    """LiveMathematicianBench: single-turn math MCQ. Reuses SkillOpt's
    `_shuffle_item_choices` so the choice order (and correct_choice.label)
    matches SkillOpt for the same (item_id, seed).

    The shuffle seed is read from `config["setup"]["choice_shuffle_seed"]`
    and defaults to 42 (matches SkillOpt's `split_seed`).
    """

    @property
    def _shuffle_seed(self) -> int:
        return int((self.config.get("setup") or {}).get("choice_shuffle_seed", 42))

    @property
    def _shuffle_choices(self) -> bool:
        return bool((self.config.get("setup") or {}).get("shuffle_choices", True))

    def load_tasks(self, split: str, limit: int | None = None) -> list[dict[str, Any]]:
        rows = read_jsonl(self.split_file(split))
        # Inject a flat `theorem_type_primary` so the yaml's `stratify_by: theorem_type_primary`
        # can hit it without the base sampler having to special-case list-valued fields.
        for r in rows:
            tts = r.get("theorem_type") or []
            if isinstance(tts, list):
                r["theorem_type_primary"] = tts[0] if tts else ""
            else:
                r["theorem_type_primary"] = str(tts)
        rows = self._subsample(rows, split, limit)
        if not self._shuffle_choices:
            return rows
        from .livemathc_official import _shuffle_item_choices
        seed = self._shuffle_seed
        return [_shuffle_item_choices(row, seed) for row in rows]

    def render_task(self, task: dict[str, Any], output_dir: Path | None = None) -> str:
        from .livemathc_official import _build_user
        return _build_user(task)

    def build_messages(
        self,
        task: dict[str, Any],
        skill_text_for_agent: str,
        output_dir: Path | None = None,
    ) -> list[dict[str, Any]]:
        from .livemathc_official import _build_system, _build_user
        system = _build_system(skill_text_for_agent)
        user = _build_user(task)
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def extract_answer(self, raw_text: str) -> str:
        from .livemathc_official import extract_answer
        return extract_answer(raw_text)

    def evaluate(
        self,
        task: dict[str, Any],
        final_answer: str,
        judge: LLMClient | None = None,
        trajectory: dict[str, Any] | None = None,
        output_dir: Path | None = None,
    ) -> dict[str, Any]:
        from .livemathc_official import evaluate as _eval
        metrics = _eval(final_answer, task["correct_choice"], task["choices"])
        return {
            "success": bool(metrics["em"]),
            "mode": "livemathc_mcq",
            "prediction": final_answer,
            "gold": task["correct_choice"].get("label", ""),
            **metrics,
        }


class DocVQAAdapter(JsonlAdapter):
    """DocVQA: single-turn multimodal VQA. Reuses SkillOpt's _build_messages
    (multimodal content with image_url part) and ANLS scoring.

    Resolves `task["image_path"]` against `config["setup"]["data_root"]`
    when relative (matches SkillOpt: image_path lives under data/docvqa_images/).
    """

    @property
    def _data_root(self) -> Path:
        return ROOT / (self.config.get("setup", {}).get("data_root", ""))

    @property
    def _image_detail(self) -> str:
        return str((self.config.get("setup") or {}).get("image_detail", "auto"))

    def _resolve_image_path(self, task: dict[str, Any]) -> str:
        raw_path = task.get("image_path") or ""
        p = Path(raw_path)
        if p.is_absolute() and p.exists():
            return str(p)
        # Try `<data_root>/<image_path>` first, then ROOT/<image_path>.
        candidate = self._data_root / raw_path
        if candidate.exists():
            return str(candidate)
        candidate2 = ROOT / raw_path
        if candidate2.exists():
            return str(candidate2)
        # Fall back to the raw string; the open() in _image_to_data_uri will
        # produce a clear FileNotFoundError so the operator can fix paths.
        return raw_path

    def render_task(self, task: dict[str, Any], output_dir: Path | None = None) -> str:
        # Text-only summary for critic / log purposes (no image bytes).
        image_path = self._resolve_image_path(task)
        return (
            f"## Question\n{task['question']}\n\n"
            f"## Document Image\n{image_path}\n\n"
            "Return the final answer inside <answer>...</answer>."
        )

    def build_messages(
        self,
        task: dict[str, Any],
        skill_text_for_agent: str,
        output_dir: Path | None = None,
    ) -> list[dict[str, Any]]:
        from .docvqa_official import _build_messages
        resolved = dict(task)
        resolved["image_path"] = self._resolve_image_path(task)
        return _build_messages(resolved, skill_text_for_agent, image_detail=self._image_detail)

    def critic_attachments(self, task: dict[str, Any]) -> list[dict[str, Any]] | None:
        """Hand the critic the same document image the agent saw, encoded
        as a data URI. Without this the critic only sees the file path
        string from render_task() and cannot tell whether the agent
        misread the document or applied the wrong rule."""
        from .docvqa_official import _image_to_data_uri
        path = self._resolve_image_path(task)
        try:
            uri = _image_to_data_uri(path)
        except Exception:
            return None
        image_url: dict[str, Any] = {"url": uri}
        if self._image_detail and self._image_detail != "auto":
            image_url["detail"] = self._image_detail
        return [{"type": "image_url", "image_url": image_url}]

    def extract_answer(self, raw_text: str) -> str:
        from .docvqa_official import extract_answer
        return extract_answer(raw_text)

    def evaluate(
        self,
        task: dict[str, Any],
        final_answer: str,
        judge: LLMClient | None = None,
        trajectory: dict[str, Any] | None = None,
        output_dir: Path | None = None,
    ) -> dict[str, Any]:
        from .docvqa_official import evaluate as _eval
        gold = task.get("answers") or task.get("answer") or []
        metrics = _eval(final_answer, gold)
        # ANLS ≥ 0.999 is effectively exact match (matches SkillOpt's `hard`).
        success = metrics["anls"] >= 0.999
        return {
            "success": bool(success),
            "mode": "docvqa_anls",
            "prediction": final_answer,
            "gold": gold,
            **metrics,
        }


def load_adapter(config_path: Path) -> BenchmarkAdapter:
    name = read_yaml(config_path)["name"]
    if name == "spreadsheetbench":
        return SpreadsheetBenchAdapter(config_path)
    if name == "alfworld":
        return ALFWorldAdapter(config_path)
    if name == "searchqa":
        return SearchQAAdapter(config_path)
    if name == "livemathc":
        return LiveMathCAdapter(config_path)
    if name == "docvqa":
        return DocVQAAdapter(config_path)
    raise ValueError(f"Unknown benchmark: {name}")
