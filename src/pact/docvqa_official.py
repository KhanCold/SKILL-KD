"""DocVQA prompt-assembly (multimodal) and ANLS evaluation, vendored from SkillOpt.

Sources:
- _build_system / _build_messages / _image_to_data_uri: from
  skillopt/envs/docvqa/rollout.py
- _normalize_text / _levenshtein_distance / _score_single_answer /
  _extract_answer_strings / extract_answer / evaluate: from
  skillopt/envs/docvqa/evaluator.py
"""
from __future__ import annotations

import ast
import base64
import json
import mimetypes
from collections.abc import Iterable
from typing import Any

from .external_prompts import load_prompt


DEFAULT_ANLS_THRESHOLD = 0.5


# ── Prompt assembly (vendored from SkillOpt) ─────────────────────────────────


def _build_system(skill_content: str) -> str:
    if skill_content.strip():
        skill_section = f"## Skill\n{skill_content.strip()}\n\n"
    else:
        skill_section = ""
    return load_prompt("rollout_system", env="docvqa").format(skill_section=skill_section)


def _image_to_data_uri(path: str) -> str:
    mime = mimetypes.guess_type(path)[0] or "image/png"
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _build_messages(item: dict, skill_content: str, image_detail: str = "auto") -> list[dict]:
    """Return a (system, multimodal user) message list ready for LLMClient.chat()."""
    system = _build_system(skill_content)
    user_text = item["question"] + "\n\nReturn the final answer inside <answer>...</answer>."
    image_url: dict[str, Any] = {"url": _image_to_data_uri(item["image_path"])}
    if image_detail and image_detail != "auto":
        image_url["detail"] = image_detail
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": image_url},
            ],
        },
    ]


# ── Evaluation (ANLS, vendored from SkillOpt) ────────────────────────────────


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    return " ".join(text.split())


def _levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if len(a) > len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (char_a != char_b)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]


def _score_single_answer(predicted: Any, target: Any, threshold: float) -> float:
    predicted_norm = _normalize_text(predicted)
    target_norm = _normalize_text(target)
    if not predicted_norm and not target_norm:
        return 1.0
    if not predicted_norm or not target_norm:
        return 0.0
    distance = _levenshtein_distance(predicted_norm, target_norm)
    normalized_distance = distance / max(len(predicted_norm), len(target_norm))
    if normalized_distance >= threshold:
        return 0.0
    return 1.0 - normalized_distance


def _extract_answer_strings(raw: Any) -> list[str]:
    if raw is None:
        return [""]
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return [""]
        parsed = None
        if text[0] in "[{":
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                try:
                    parsed = ast.literal_eval(text)
                except (ValueError, SyntaxError):
                    parsed = None
        if parsed is None:
            return [text]
        return _extract_answer_strings(parsed)
    if isinstance(raw, dict):
        for key in ("answers", "ground_truth", "answer"):
            if key in raw:
                return _extract_answer_strings(raw[key])
        return [str(raw)]
    if isinstance(raw, Iterable) and not isinstance(raw, (bytes, bytearray)):
        answers: list[str] = []
        for item in raw:
            if isinstance(item, dict):
                for key in ("text", "answer", "value"):
                    if key in item:
                        answers.extend(_extract_answer_strings(item[key]))
                        break
                else:
                    answers.append(str(item))
                continue
            answers.append(str(item))
        return answers or [""]
    return [str(raw)]


def extract_answer(text: str) -> str:
    lower = text.lower()
    start = lower.rfind("<answer>")
    end = lower.rfind("</answer>")
    if start != -1 and end != -1 and end > start:
        return text[start + len("<answer>"):end].strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else text.strip()


def evaluate(prediction_text: str, gold_answers: Any) -> dict:
    answer = extract_answer(prediction_text)
    answers = _extract_answer_strings(gold_answers)
    score = 0.0
    for target in answers:
        score = max(score, _score_single_answer(answer, target, DEFAULT_ANLS_THRESHOLD))
    return {
        "anls": score,
        "predicted_answer": answer,
        "gold_answers": answers,
    }
