from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any


def extract_single_action(raw_text: str) -> str:
    """Extract the action from <action>...</action> tags in the LLM output.

    Returns an empty string when the tags are missing or empty. The empty
    string is not in any admissible-command list, so the env will treat it
    as an invalid action — consistent with SkillOpt's projection behaviour
    where unparseable responses get `valid=0`.
    """
    match = re.search(r"<action>(.*?)</action>", raw_text, flags=re.I | re.S)
    if not match:
        return ""
    return match.group(1).strip().lower()


def detect_repetitive_actions(
    actions: list[str],
    consecutive_limit: int = 5,
) -> str | None:
    """Return a stop reason when the same ALFWorld action repeats consecutively."""
    if not actions:
        return None

    last = actions[-1]
    consecutive = 0
    for action in reversed(actions):
        if action != last:
            break
        consecutive += 1
    if consecutive >= consecutive_limit:
        return f"repeated_action:{last}"

    return None


def _make_env(task: dict[str, Any], max_steps: int = 50):
    """Create a TextWorld environment for the given ALFWorld task."""
    task_path = Path(str(task.get("_path") or ""))
    game_file = Path(str(task.get("_game_file") or task_path.with_name("game.tw-pddl")))
    if not task_path.exists() or not game_file.exists():
        raise RuntimeError(
            "ALFWorld official evaluation requires downloaded ALFWorld task files and game.tw-pddl. "
            "Place the official ALFWorld data under data/alfworld/raw first."
        )

    try:
        import textworld
        import textworld.gym
        from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos
    except ImportError as exc:
        raise RuntimeError(
            "ALFWorld official evaluation requires alfworld and textworld. "
            "Install with: pip install -e '.[alfworld]'"
        ) from exc

    request_infos = textworld.EnvInfos(won=True, admissible_commands=True, extras=["gamefile"])
    env_id = textworld.gym.register_games(
        [str(game_file)],
        request_infos,
        batch_size=1,
        asynchronous=False,
        max_episode_steps=max_steps,
        wrappers=[AlfredDemangler(shuffle=False), AlfredInfos],
    )
    return textworld.gym.make(env_id)


def run_interactive_alfworld_rollout(
    task: dict[str, Any],
    skill_text: str,
    llm: Any,
    role: str = "student",
    max_steps: int = 50,
) -> dict[str, Any]:
    """Run an interactive step-by-step ALFWorld rollout.

    Uses multi-turn chat. The model keeps the full conversation history.
    Each turn the model receives only the raw environment observation.
    """
    from .prompts import ALFWORLD_SYSTEM_ROLE, ALFWORLD_USER_SUFFIX, ALFWORLD_INITIAL_SKILL

    env = _make_env(task, max_steps)
    obs, infos = env.reset()
    won = bool(infos.get("won", [False])[0])
    done = won

    # System prompt: role anchor + SkillOpt-aligned initial skill (task types,
    # principles, common mistakes) + (optional) evolved skill block.
    skill_block = skill_text.strip()
    system_parts: list[str] = [ALFWORLD_SYSTEM_ROLE, ALFWORLD_INITIAL_SKILL]
    if skill_block:
        system_parts.append(f"Current skill.md:\n{skill_block}")
    messages: list[dict[str, str]] = [
        {"role": "system", "content": "\n\n".join(system_parts)}
    ]
    env_steps: list[dict[str, Any]] = []
    actions: list[str] = []
    stop_reason: str | None = None
    repeat_action_limit = int(os.environ.get("PACT_ALFWORLD_REPEAT_ACTION_LIMIT", "5"))

    for _ in range(max_steps):
        admissible = list(infos.get("admissible_commands", [[]])[0])
        observation = obs[0] if obs else ""

        user_content = observation
        if admissible:
            user_content += "\n\nAdmissible commands: " + ", ".join(admissible)
        user_content += "\n\n" + ALFWORLD_USER_SUFFIX

        messages.append({"role": "user", "content": user_content})

        # Call LLM with the full conversation history.
        try:
            raw = llm.chat(messages)
        except Exception as exc:
            err_msg = str(exc)
            if "maximum context length" in err_msg or "input_tokens" in err_msg:
                env_steps.append(
                    {
                        "action": "CONTEXT_LIMIT_EXCEEDED",
                        "observation": observation,
                        "admissible": admissible,
                        "won": False,
                        "done": True,
                    }
                )
                break
            raise
        cleaned = llm.clean_content(raw)
        action = extract_single_action(cleaned)
        actions.append(action)
        action_is_admissible = action in admissible if admissible else True

        # Store assistant response (preserve raw thinking tags in trajectory).
        messages.append({"role": "assistant", "content": raw})

        # Execute action in environment.
        obs, scores, dones, infos = env.step([action])
        won = bool(infos.get("won", [False])[0])
        done = bool(dones[0]) or won

        env_steps.append(
            {
                "action": action,
                "observation": observation,
                "admissible": admissible,
                "admissible_action": action_is_admissible,
                "won": won,
                "done": done,
            }
        )

        if done:
            break

        stop_reason = detect_repetitive_actions(
            actions,
            consecutive_limit=repeat_action_limit,
        )

        if stop_reason is not None:
            env_steps[-1]["done"] = True
            env_steps[-1]["stop_reason"] = stop_reason
            break

    env.close()

    system_msg = next(
        (m["content"] for m in messages if m["role"] == "system"), ""
    )
    return {
        "role": role,
        "skill_used": skill_block,
        "prompt": system_msg,
        "raw": env_steps[-1]["action"] if env_steps else "",
        "final_answer": "\n".join(s["action"] for s in env_steps),
        "messages": messages,
        "env_steps": env_steps,
        "stop_reason": stop_reason,
    }
