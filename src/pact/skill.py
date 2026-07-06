from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .io import read_json, write_json, write_text
from .skill_ops import (
    SkillRule,
    apply_skill_ops,
    canonicalize_skill_text,
    parse_skill_rules,
)


@dataclass
class EditRecord:
    """One entry in the whole-skills edit log.

    `before`/`after` carry the full rule three-tuple (title/content/why)
    when applicable. `group_dir` is a path relative to the benchmark root
    (e.g. `evolution/groups/0007_<safe_id>`) used by the agentic critic's
    `get_original_trajectories` tool to locate the originating rollouts on
    disk; we deliberately do not embed the trajectory payloads themselves.
    """

    edit_index: int
    group_index: int
    round_index: int
    op: str
    rule_id: str
    before: dict[str, str] | None
    after: dict[str, str] | None
    group_dir: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EditRecord":
        return cls(
            edit_index=int(data["edit_index"]),
            group_index=int(data["group_index"]),
            round_index=int(data["round_index"]),
            op=str(data["op"]),
            rule_id=str(data["rule_id"]),
            before=data.get("before"),
            after=data.get("after"),
            group_dir=str(data.get("group_dir", "")),
        )


@dataclass
class SkillState:
    original: str
    current: str
    version: int = 0
    edit_log: list[EditRecord] = field(default_factory=list)

    @classmethod
    def from_path(cls, path: Path) -> "SkillState":
        text = canonicalize_skill_text(path.read_text(encoding="utf-8"))
        return cls(original=text, current=text, version=0, edit_log=[])

    def apply_patch(self, patch_text: str) -> tuple[bool, str | None]:
        result = apply_skill_ops(self.current, patch_text)
        if not result.ok:
            return False, result.error
        self.current = result.text
        self.version += 1
        return True, None

    def record_edits(
        self,
        ops: list[dict[str, Any]],
        *,
        rules_before: list[SkillRule],
        rules_after: list[SkillRule],
        group_index: int,
        round_index: int,
        group_dir: str,
    ) -> list[EditRecord]:
        """Append one EditRecord per op to the log. Reconstructs each op's
        `before` and `after` rule snapshots from rules_before / rules_after.
        Should be called only when the candidate skill is committed."""

        before_by_id = {r.id: r for r in rules_before}
        after_by_id = {r.id: r for r in rules_after}

        # `add` ops assign new ids in sequence after the prior max id.
        before_ids = {r.id for r in rules_before}
        added_ids_in_order = [r.id for r in rules_after if r.id not in before_ids]
        add_cursor = 0

        appended: list[EditRecord] = []
        next_index = len(self.edit_log)
        for op in ops:
            op_name = op["op"]
            if op_name == "add":
                new_id = added_ids_in_order[add_cursor]
                add_cursor += 1
                after_rule = after_by_id[new_id]
                record = EditRecord(
                    edit_index=next_index,
                    group_index=group_index,
                    round_index=round_index,
                    op="add",
                    rule_id=new_id,
                    before=None,
                    after=_rule_payload(after_rule),
                    group_dir=group_dir,
                )
            elif op_name == "update":
                rule_id = op["id"]
                record = EditRecord(
                    edit_index=next_index,
                    group_index=group_index,
                    round_index=round_index,
                    op="update",
                    rule_id=rule_id,
                    before=_rule_payload(before_by_id[rule_id]),
                    after=_rule_payload(after_by_id[rule_id]),
                    group_dir=group_dir,
                )
            elif op_name == "delete":
                rule_id = op["id"]
                record = EditRecord(
                    edit_index=next_index,
                    group_index=group_index,
                    round_index=round_index,
                    op="delete",
                    rule_id=rule_id,
                    before=_rule_payload(before_by_id[rule_id]),
                    after=None,
                    group_dir=group_dir,
                )
            else:
                raise ValueError(f"unknown op {op_name!r}")

            self.edit_log.append(record)
            appended.append(record)
            next_index += 1

        return appended

    def save_snapshot(self, path: Path) -> None:
        write_text(path, self.current)

    def save_edit_log(self, path: Path) -> None:
        write_json(path, [record.to_dict() for record in self.edit_log])

    def load_edit_log(self, path: Path) -> None:
        if not path.exists():
            self.edit_log = []
            return
        data = read_json(path)
        if not isinstance(data, list):
            raise ValueError(f"{path} must contain a JSON array")
        self.edit_log = [EditRecord.from_dict(item) for item in data]


def _rule_payload(rule: SkillRule) -> dict[str, str]:
    return {
        "title": rule.title,
        "content": rule.content,
        "why": rule.why,
    }


__all__ = ["SkillState", "EditRecord"]
