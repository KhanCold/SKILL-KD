"""Model pricing table and cost calculation utilities.

Prices are per million tokens in USD. Applied at read time (viz_server)
so updating this table doesn't require re-running experiments.
"""

from __future__ import annotations

from typing import Any

# Prices per 1M tokens (USD). Source: provider pricing pages.
# Keep model names lowercase for case-insensitive matching.
MODEL_PRICING = [
    {
        "model": "qwen3.7-max",
        "input": 1.66,
        "output": 4.97,
        "cached_input": 0.17,
    },
    {
        "model": "qwen3.7-plus",
        "input": 0.28,
        "output": 1.10,
        "cached_input": 0.03,
    },
    {
        "model": "qwen3.5-27b",
        "input": 0.083,
        "output": 0.66,
        "cached_input": 0.008,
    },
    {
        "model": "bailian/deepseek-v4-pro",
        "input": 0.435,
        "output": 0.87,
        "cached_input": 0.004,
    },
    {
        "model": "bailian/deepseek-v4-flash",
        "input": 0.14,
        "output": 0.28,
        "cached_input": 0.003,
    },
    {
        "model": "bailian/glm-5.1",
        "input": 0.83,
        "output": 3.31,
        "cached_input": 0.15,
    },
    {
        "model": "claude-sonnet-4-6",
        "input": 3.00,
        "output": 15.00,
        "cached_input": 0.30,
    },
    {
        "model": "claude-opus-4-8",
        "input": 5.00,
        "output": 25.00,
        "cached_input": 0.50,
    },
    {
        "model": "gpt-5.5-0424-global",
        "input": 5.00,
        "output": 30.00,
        "cached_input": 0.50,
    },
]

# Build lookup: lowercase model name → price dict
_PRICING_MAP: dict[str, dict[str, float]] = {
    p["model"].lower(): p for p in MODEL_PRICING
}


def find_price(model_name: str) -> dict[str, float] | None:
    """Look up pricing for a model name.

    Case-insensitive. Tries exact match first, then checks if the
    model name contains a known pricing key (handles variants like
    ``Qwen3.7-Max-0808`` matching ``qwen3.7-max``).
    """
    if not model_name:
        return None
    name_lower = model_name.lower()

    # Exact match
    if name_lower in _PRICING_MAP:
        return _PRICING_MAP[name_lower]

    # Substring match: find the longest pricing key that appears in the name
    best_match = None
    best_len = 0
    for key, price in _PRICING_MAP.items():
        if key in name_lower and len(key) > best_len:
            best_match = price
            best_len = len(key)
    return best_match


def calculate_cost(usage: dict[str, Any], model_name: str) -> float | None:
    """Calculate cost in USD for a usage dict and model name.

    Returns ``None`` if the model is not in the pricing table.

    Formula (prices are per 1M tokens)::

        non_cached_input = input_tokens - cached_tokens
        cost = (non_cached_input * input_price
                + output_tokens * output_price
                + cached_tokens * cached_input_price) / 1_000_000
    """
    price = find_price(model_name)
    if price is None:
        return None

    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    cached_tokens = usage.get("cached_tokens", 0)

    non_cached_input = max(0, input_tokens - cached_tokens)

    cost = (
        non_cached_input * price["input"]
        + output_tokens * price["output"]
        + cached_tokens * price["cached_input"]
    ) / 1_000_000

    return round(cost, 6)


def compute_run_cost(usage: dict[str, Any]) -> dict[str, Any]:
    """Compute per-role and total cost from a run-level usage.json dict.

    Returns a dict like::

        {
            "student": {"cost": 0.12, "model": "qwen3-4b", "priced": True},
            "teacher": {"cost": 0.34, "model": "qwen3.7-plus", "priced": True},
            "critic":  {"cost": 0.56, "model": "qwen3.7-plus", "priced": True},
            "total":   1.02,
            "all_priced": True,
        }

    If a model has no pricing entry, its ``cost`` is ``None`` and
    ``priced`` is False; ``total`` excludes unpriced roles.
    """
    result: dict[str, Any] = {}
    total = 0.0
    all_priced = True

    for role in ("student", "teacher", "critic"):
        role_usage = usage.get(role)
        if not role_usage:
            continue
        model = role_usage.get("model", "")
        cost = calculate_cost(role_usage, model)
        result[role] = {
            "cost": cost,
            "model": model,
            "priced": cost is not None,
            "input_tokens": role_usage.get("input_tokens", 0),
            "output_tokens": role_usage.get("output_tokens", 0),
            "cached_tokens": role_usage.get("cached_tokens", 0),
            "requests": role_usage.get("requests", 0),
        }
        if cost is not None:
            total += cost
        else:
            all_priced = False

    result["total"] = round(total, 6)
    result["all_priced"] = all_priced
    return result
