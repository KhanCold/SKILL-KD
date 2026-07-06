"""Prompt loader vendored from SkillOpt.

Prompts are stored as ``.md`` files inside per-task subdirectories
(``src/pact/external_prompts/<env>/<name>.md``).

``load_prompt(name, env)`` reads the file and caches the contents.
This mirrors SkillOpt's `skillopt/prompts/__init__.py` so that the
prompt strings we vendor here can stay byte-identical to the upstream
versions and the per-task `_build_system` functions are drop-in
replacements.
"""
from __future__ import annotations

import os

_PROMPTS_DIR = os.path.dirname(os.path.abspath(__file__))

_cache: dict[str, str] = {}


def load_prompt(name: str, env: str) -> str:
    """Load ``src/pact/external_prompts/<env>/<name>.md``.

    Raises FileNotFoundError with a useful message if the file is missing.
    """
    path = os.path.join(_PROMPTS_DIR, env, f"{name}.md")
    if path in _cache:
        return _cache[path]
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Prompt '{name}' for env '{env}' not found at {path}"
        )
    with open(path, encoding="utf-8") as f:
        content = f.read()
    _cache[path] = content
    return content


def clear_cache() -> None:
    _cache.clear()
