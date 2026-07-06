from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


RULE_HEADER_RE = re.compile(r"^\[RULE (?P<id>\d{3})\] (?P<title>.+)$")
VALID_ID_RE = re.compile(r"^\d{3}$")


@dataclass
class SkillRule:
    id: str
    title: str
    content: str
    why: str

    def as_dict(self) -> dict[str, str]:
        return {
            "id": self.id,
            "title": self.title,
            "content": self.content,
            "why": self.why,
        }


@dataclass
class SkillOpsResult:
    ok: bool
    text: str
    error: str | None = None
    ops: list[dict[str, Any]] | None = None
    rules_before: list[SkillRule] | None = None
    rules_after: list[SkillRule] | None = None


def parse_skill_rules(text: str) -> list[SkillRule]:
    stripped = text.strip()
    if not stripped:
        return []

    blocks = re.split(r"\n\s*\n", stripped)
    rules: list[SkillRule] = []
    seen: set[str] = set()
    for block in blocks:
        lines = [line.rstrip() for line in block.splitlines() if line.strip()]
        if len(lines) != 3:
            raise ValueError(
                "each rule must have exactly header + content + why lines"
            )

        match = RULE_HEADER_RE.match(lines[0])
        if not match:
            raise ValueError(f"invalid rule header: {lines[0]!r}")

        rule_id = match.group("id")
        if rule_id in seen:
            raise ValueError(f"duplicate rule id {rule_id}")
        seen.add(rule_id)

        body = lines[1:]
        fields: dict[str, str] = {}
        for line in body:
            if line.startswith("content: "):
                fields["content"] = line[len("content: "):].strip()
            elif line.startswith("why: "):
                fields["why"] = line[len("why: "):].strip()
            else:
                raise ValueError(
                    f"unrecognized line for rule {rule_id}: {line!r}"
                )

        if "content" not in fields:
            raise ValueError(f"missing content line for rule {rule_id}")
        if "why" not in fields:
            raise ValueError(f"missing why line for rule {rule_id}")

        rules.append(
            SkillRule(
                id=rule_id,
                title=match.group("title").strip(),
                content=fields["content"],
                why=fields["why"],
            )
        )
    return rules


def render_for_critic(rules: list[SkillRule]) -> str:
    """Three-line render (header + content + why). Used for critic prompts
    and for on-disk skill snapshots."""
    if not rules:
        return ""
    blocks = [
        f"[RULE {rule.id}] {rule.title}\n"
        f"content: {rule.content}\n"
        f"why: {rule.why}"
        for rule in rules
    ]
    return "\n\n".join(blocks).rstrip() + "\n"


def render_for_agent(rules: list[SkillRule]) -> str:
    """Two-line render (header + content). Used for the agent's runtime
    rollout prompt — `why` (student-vs-teacher narrative) is intentionally
    omitted to avoid polluting agent context."""
    if not rules:
        return ""
    blocks = [
        f"[RULE {rule.id}] {rule.title}\n"
        f"content: {rule.content}"
        for rule in rules
    ]
    return "\n\n".join(blocks).rstrip() + "\n"


def canonicalize_skill_text(text: str) -> str:
    """Read any supported format (4-line, 3-line legacy, bullet-list legacy)
    and re-emit in the canonical 4-line form."""
    if not text.strip():
        return ""
    try:
        return render_for_critic(parse_skill_rules(text))
    except ValueError:
        return render_for_critic(_legacy_markdown_to_rules(text))


def extract_json_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for idx, char in enumerate(text):
        if char != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    raise ValueError("critic output does not contain a JSON object")


def parse_skill_ops(text: str) -> list[dict[str, Any]]:
    """Parse a JSON-encoded ops payload. Accepts either a string containing
    a JSON object with an `ops` key, or a string that *is* the ops list."""
    obj = extract_json_object(text)
    ops = obj.get("ops")
    if not isinstance(ops, list):
        raise ValueError("critic JSON must contain an ops list")
    _validate_ops_shape(ops)
    return ops


def _validate_ops_shape(ops: list[Any]) -> None:
    for index, op in enumerate(ops):
        if not isinstance(op, dict):
            raise ValueError(f"op {index} must be an object")
        op_name = op.get("op")
        if op_name not in {"add", "update", "delete"}:
            raise ValueError(f"op {index} has invalid op {op_name!r}")

        if op_name == "add":
            rule = _require_rule(op, index)
            if "id" in rule:
                raise ValueError(f"add op {index} must not provide rule.id")
        elif op_name == "update":
            _require_existing_id(op, index)
            _require_rule(op, index)
        else:
            _require_existing_id(op, index)


def apply_skill_ops(current_skill: str, critic_output: str | list[dict[str, Any]]) -> SkillOpsResult:
    try:
        rules = parse_skill_rules(canonicalize_skill_text(current_skill))
        if isinstance(critic_output, list):
            ops = critic_output
            _validate_ops_shape(ops)
        else:
            ops = parse_skill_ops(critic_output)
        updated = _apply_ops(rules, ops)
        return SkillOpsResult(
            True,
            render_for_critic(updated),
            ops=ops,
            rules_before=rules,
            rules_after=updated,
        )
    except Exception as exc:
        return SkillOpsResult(False, current_skill, str(exc))


def _apply_ops(rules: list[SkillRule], ops: list[dict[str, Any]]) -> list[SkillRule]:
    updated = list(rules)
    next_id_num = max((int(rule.id) for rule in updated), default=0) + 1
    for index, op in enumerate(ops):
        op_name = op["op"]
        by_id = {rule.id: pos for pos, rule in enumerate(updated)}

        if op_name == "add":
            rule_data = op["rule"]
            next_id = f"{next_id_num:03d}"
            next_id_num += 1
            updated.append(_rule_from_data(next_id, rule_data, index))
        elif op_name == "update":
            rule_id = op["id"]
            if rule_id not in by_id:
                raise ValueError(f"update op {index} references missing rule {rule_id}")
            updated[by_id[rule_id]] = _rule_from_data(rule_id, op["rule"], index)
        elif op_name == "delete":
            rule_id = op["id"]
            if rule_id not in by_id:
                raise ValueError(f"delete op {index} references missing rule {rule_id}")
            del updated[by_id[rule_id]]
    return updated


def _require_existing_id(op: dict[str, Any], index: int) -> str:
    rule_id = op.get("id")
    if not isinstance(rule_id, str) or not VALID_ID_RE.match(rule_id):
        raise ValueError(f"op {index} must provide a three-digit id")
    return rule_id


def _require_rule(op: dict[str, Any], index: int) -> dict[str, Any]:
    rule = op.get("rule")
    if not isinstance(rule, dict):
        raise ValueError(f"op {index} must provide a rule object")
    return rule


def _rule_from_data(rule_id: str, data: dict[str, Any], index: int) -> SkillRule:
    if "id" in data and data["id"] != rule_id:
        raise ValueError(f"op {index} rule.id must match target id")
    title = _require_nonempty_string(data, "title", index)
    content = _require_nonempty_string(data, "content", index)
    why = _require_nonempty_string(data, "why", index)
    return SkillRule(id=rule_id, title=title, content=content, why=why)


def _require_nonempty_string(data: dict[str, Any], field: str, index: int) -> str:
    value = data.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"op {index} rule.{field} must be a non-empty string")
    if "\n" in value:
        raise ValueError(f"op {index} rule.{field} must be a single line")
    return value.strip()


def _legacy_markdown_to_rules(text: str) -> list[SkillRule]:
    bullets = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("- "):
            bullets.append(stripped[2:].strip())

    if not bullets:
        raise ValueError("skills.md is not in canonical rule format")

    rules = []
    for idx, bullet in enumerate(bullets, start=1):
        title = _title_from_legacy_bullet(bullet)
        rules.append(
            SkillRule(
                id=f"{idx:03d}",
                title=title,
                content=bullet,
                why="Migrated from the legacy initial skill list.",
            )
        )
    return rules


def _title_from_legacy_bullet(text: str) -> str:
    words = re.sub(r"[.`]", "", text).split()
    title = " ".join(words[:8]).strip()
    return title.rstrip(".,;:") or "Migrated rule"
