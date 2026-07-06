from __future__ import annotations

import json
import textwrap
from typing import Any


ROLLOUT_SYSTEM = ""


# Minimal ALFWorld harness contract. Identical in spirit to SkillOpt's
# `You are an expert agent operating in the ALFRED Embodied Environment.`
# system prompt + per-turn `<action>` requirement, just split across system
# and user messages so chat-style multi-turn doesn't repeat the role text.
ALFWORLD_SYSTEM_ROLE = (
    "You are an expert agent operating in the ALFRED Embodied Environment."
)

ALFWORLD_USER_SUFFIX = (
    "Choose one admissible command. "
    "Respond with the action wrapped in <action>...</action> tags."
)

# SkillOpt's ALFWorld initial skill content, injected into system prompt as
# fixed background knowledge. This gives the model task-type guidance, general
# principles, and common-mistake warnings from the start.
ALFWORLD_INITIAL_SKILL = """\
## Task Types

| Type | Goal | Key Steps |
|------|------|-----------|
| Pick & Place | Put object X in/on receptacle Y | Find X -> take X -> go to Y -> put X in/on Y |
| Pick Two & Place | Put two instances of X in/on Y | Find X1 -> take -> place -> find X2 -> take -> place |
| Examine in Light | Examine object X under desklamp | Find X -> take X -> find desklamp -> use desklamp |
| Clean & Place | Clean object X and put in/on Y | Find X -> take X -> go to sink -> clean X -> go to Y -> put X |
| Heat & Place | Heat object X and put in/on Y | Find X -> take X -> go to microwave -> heat X -> go to Y -> put X |
| Cool & Place | Cool object X and put in/on Y | Find X -> take X -> go to fridge -> cool X -> go to Y -> put X |

## General Principles

1. **Decompose the task**: Parse the goal into ordered sub-goals (locate, acquire, transform, deliver). Complete each before moving to the next.
2. **Systematic exploration**: Search each surface and container exactly once before revisiting. Open closed containers (drawers, cabinets, fridge) before judging them empty.
3. **Grab immediately**: When a required object is visible and reachable, take it right away before moving elsewhere.
4. **Transform before placing**: If the task requires cleaning, heating, or cooling, perform the state change at the appropriate appliance before heading to the final destination.
5. **Direct delivery**: Once holding the transformed (or untransformed) goal object, navigate straight to the target receptacle and place it.
6. **Track progress**: Maintain an internal count of how many objects still need to be found and placed. Only stop searching when the count reaches zero.
7. **Avoid loops**: Never repeat the same action more than twice in a row. If stuck, move to a different unexplored location.
8. **Only choose admissible actions**: Always pick an action from the admissible action list. Do not invent actions.

## Common Mistakes to Avoid

- **Revisiting searched locations**: Keep track of which surfaces/containers have been checked; do not re-examine them.
- **Ignoring visible objects**: If the target object appears in the observation, pick it up immediately.
- **Skipping state changes**: Do not place an object at the destination without first cleaning/heating/cooling it when required.
- **Premature termination**: Do not stop the episode until all goal conditions are verified as met.
- **Action loops**: Repeatedly toggling or examining the same object wastes steps. Move on to new locations instead.
"""


def rollout_prompt(benchmark: str, role: str, task_text: str, skill_text: str) -> str:
    """Build the agent's runtime prompt. `skill_text` is whatever the caller
    already rendered for the agent (typically via `render_for_agent`)."""
    skill_block = skill_text.strip() if skill_text.strip() else "[None]"
    return f"""

Current skill.md:
{skill_block}

Task:
{task_text}

"""


CRITIC_SYSTEM = """You are a skill curator. You maintain a small list of reusable rules an agent uses to succeed on tasks. Each rule has three fields:

- title:   a short noun phrase (≤8 words) naming the rule so humans can scan the list.
- content: one self-contained sentence shaped "When <trigger>, do <action>". The trigger half states the type-level scenario (no specific file names, cell addresses, room names, or object nouns) and the action half states what the agent should do. This is the only field the agent sees at runtime, so it must stand on its own.
- why:     one sentence recording why this rule was needed: what the failing rollout did vs what the succeeding rollout did. This field is for your own audit trail (deciding whether a rule is still load-bearing) and is not shown to the agent.

Each decision you receive: the current skill list, the full edit log inline (every prior rule version), and a list of trajectories. Each trajectory shows the exact `skill_used` (what skill text the agent had in its prompt at that moment), the agent's `final_answer`, and the slimmed evaluation. The trajectories include the original student attempt, the teacher's attempt, and any prior retry attempts in this same task — so you can see precisely which rules were already tried in earlier rounds and how they performed.

You have two tools:
  get_original_trajectories(edit_index) — investigation tool. Returns the initial student and teacher rollouts that motivated a specific past edit, in the same {skill_used, model_output, evaluation} shape as the trajectories you already see. Use it before any update/delete to confirm the prior failure mode is or is not the same as this one. You may call this up to 10 times per decision; after that the harness will force you to submit.
  submit_ops(ops) — termination tool. Call this exactly once when you have made your decision. Pass {"ops":[]} when no change is justified.

Bias against pure `add`. The cheapest and laziest op is to add a new rule for every new failure. This produces a long, fragmented skill list with overlapping rules and degrades the agent. Before adding, you must internally check whether an existing rule is already on-topic and could be merged or sharpened instead.

Decision discipline:
1. First, walk the current skill list rule-by-rule and classify each against this failure: does it `cover` (rule, applied literally, would have prevented this failure → no op for this rule), is it `partial` (on-topic but missing a sub-condition → update is appropriate), or is it `unrelated`.
2. If any existing rule is partial, prefer `update` over `add`. Name the missed sub-condition in the new content. Before updating, investigate the most relevant prior edit for that rule_id when one exists.
3. Only emit `add` when no existing rule is partial or covers — i.e. the failure introduces a genuinely new trigger. If the new trigger overlaps an existing rule or resembles a prior failed candidate in this task, investigate the closest prior edit before adding; otherwise skip investigation and add directly.
4. Use `delete` to retire rules whose trigger overlaps with another rule and never carries its weight (fewer rules, sharper rules). Before deleting or merging rules, investigate at least one relevant prior edit for each rule being retired.
5. Before any `update` or `delete`, scan the edit log for prior edits on the same rule_id. If a similar revision was already tried and reversed, prefer no-op. A rule with 3+ prior edits is a strong signal to replace it entirely (delete + add).
6. Forbidden: cosmetic rewording of `content` without a named missing sub-condition; using specific entity names (cell addresses like "B11:B12", file names, room names, specific object nouns like "apple" or "microwave") in `content`; putting failure narrative into `content` (it belongs in `why`); proposing a new rule whose trigger paraphrases an existing rule's trigger.
7. Do not investigate just to satisfy a ritual. If the current skill list and edit log have no relevant prior edit, decide from the current trajectories and submit. Emit at most 3 ops; fewer is better. submit_ops with {"ops":[]} is a valid and often correct answer when every existing rule covers the failure or no rule meaningfully helps.
"""


GET_ORIGINAL_TRAJECTORIES_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_original_trajectories",
        "description": (
            "Fetch the initial student and teacher rollouts that motivated a specific past skill edit. "
            "Use when a prior edit looks relevant to the current failure, especially before updating, "
            "deleting, merging, or adding an overlapping rule."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "edit_index": {
                    "type": "integer",
                    "description": "Zero-based index from the edit log.",
                }
            },
            "required": ["edit_index"],
        },
    },
}

SUBMIT_OPS_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "submit_ops",
        "description": (
            "Submit your final ops list and end the decision. Call exactly once. "
            "Pass {\"ops\":[]} when no change is justified."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "ops": {
                    "type": "array",
                    "maxItems": 3,
                    "items": {
                        "type": "object",
                        "properties": {
                            "op": {
                                "type": "string",
                                "enum": ["add", "update", "delete"],
                                "description": (
                                    "Edit kind. "
                                    "'add' creates a new rule — provide `rule`, OMIT `id` (the system assigns the next three-digit id). "
                                    "'update' replaces an existing rule — provide BOTH `id` (existing three-digit rule id) and `rule`. "
                                    "'delete' removes an existing rule — provide `id` (existing three-digit rule id); `rule` is ignored."
                                ),
                            },
                            "id": {
                                "type": "string",
                                "description": "Three-digit existing rule id (update/delete only).",
                            },
                            "rule": {
                                "type": "object",
                                "properties": {
                                    "title": {"type": "string"},
                                    "content": {"type": "string"},
                                    "why": {"type": "string"},
                                },
                            },
                        },
                        "required": ["op"],
                    },
                }
            },
            "required": ["ops"],
        },
    },
}

CRITIC_TOOLS: list[dict[str, Any]] = [GET_ORIGINAL_TRAJECTORIES_TOOL, SUBMIT_OPS_TOOL]


def _indent(text: str, prefix: str = "  ") -> str:
    if not text:
        return prefix
    return textwrap.indent(text.rstrip(), prefix)


def _render_rule_payload(rule: dict[str, str], prefix: str = "    ") -> str:
    title = rule.get("title", "")
    content = rule.get("content", "")
    why = rule.get("why", "")
    lines = [
        f"{prefix}title:   {title}",
        f"{prefix}content: {content}",
        f"{prefix}why:     {why}",
    ]
    return "\n".join(lines)


def render_edit_log_for_critic(edit_log: list[dict[str, Any]]) -> str:
    """Render the whole-skills edit log inline so the critic sees every
    prior rule version (full four-tuple before/after) without a tool call."""
    if not edit_log:
        return "(no prior edits)\n"
    parts: list[str] = []
    for record in edit_log:
        header = (
            f"[edit {record['edit_index']}] "
            f"(group {record['group_index']:04d} round {record['round_index']:02d}) "
            f"{record['op'].upper()} rule {record['rule_id']}"
        )
        block_lines = [header]
        op = record["op"]
        if op == "add":
            block_lines.append("  after:")
            block_lines.append(_render_rule_payload(record["after"]))
        elif op == "update":
            block_lines.append("  before:")
            block_lines.append(_render_rule_payload(record["before"]))
            block_lines.append("  after:")
            block_lines.append(_render_rule_payload(record["after"]))
        elif op == "delete":
            block_lines.append("  before:")
            block_lines.append(_render_rule_payload(record["before"]))
        parts.append("\n".join(block_lines))
    return "\n\n".join(parts) + "\n"


def critic_initial_user(
    benchmark: str,
    current_skill_for_critic: str,
    edit_log: list[dict[str, Any]],
    task_text: str,
    trajectories: list[dict[str, Any]],
    attachments: list[dict[str, Any]] | None = None,
) -> str | list[dict[str, Any]]:
    """Build the initial user message for the agentic critic loop.

    Returns a plain string when ``attachments`` is None (text-only
    benchmarks). When attachments are provided (e.g. docvqa's document
    image), returns an OpenAI-style multimodal content list so the critic
    receives the same image bytes the agent saw. The attachments are
    spliced into the *new task* section so the image sits next to the
    question rather than at the end where it would lose context.
    """
    skill_block = current_skill_for_critic.strip() if current_skill_for_critic.strip() else "[empty skill list]"
    edit_log_block = render_edit_log_for_critic(edit_log)
    n_edits = len(edit_log)
    traj_block = json.dumps(trajectories, ensure_ascii=False, indent=2)

    head = f"""Benchmark: {benchmark}

Current skill list:
{skill_block}

Edit log ({n_edits} prior edits, oldest first):

{edit_log_block}
New task:
{task_text}
"""
    tail = f"""
Trajectories (this group, oldest first — student_initial, teacher, then any prior retry attempts in this same task). Each trajectory records the exact `skill_used` the agent had in its prompt at that moment, the agent's full `model_output` (its complete reply, thinking trace included — not just the extracted answer), and the slimmed evaluation. Earlier retry attempts in this task are already shown here, so you can see precisely which rules were tried before and how they performed:
{traj_block}

Decide whether the skill list needs an op. End your decision by calling submit_ops(ops=...). Before deciding, walk the current skill list rule-by-rule against this failure and consider update/delete/merge before defaulting to add. You may show your reasoning in prose before the tool call.
"""

    if not attachments:
        return head + tail

    parts: list[dict[str, Any]] = [{"type": "text", "text": head}]
    parts.extend(attachments)
    parts.append({"type": "text", "text": tail})
    return parts
