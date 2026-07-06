from __future__ import annotations

import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import datetime

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    BadRequestError,
    OpenAI,
    RateLimitError,
)

ROOT = Path(__file__).resolve().parents[2]


def _clean_base_url(base_url: str | None) -> str | None:
    if base_url is None:
        return None
    cleaned = base_url.strip()
    return cleaned or None


def load_env_file(path: Path | None = None) -> None:
    env_path = path or ROOT / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value


@dataclass
class TokenUsage:
    """Accumulated token usage across LLM API calls."""

    input_tokens: int = 0  # prompt_tokens (total, including cached)
    output_tokens: int = 0  # completion_tokens
    cached_tokens: int = 0  # prompt_tokens_details.cached_tokens
    requests: int = 0


@dataclass
class LLMConfig:
    model: str
    base_url: str | None = None
    api_key_env: str = "DASHSCOPE_API_KEY"
    # Per-request timeout in seconds. Overridable via env PACT_LLM_TIMEOUT.
    # 600s gives enough headroom for vLLM queue stalls under heavy concurrent
    # load (e.g. LiveMathC with 12 workers on a saturated GPU). Transient
    # timeout handling is implemented by LLMClient._call_with_retry below.
    timeout: float = float(os.environ.get("PACT_LLM_TIMEOUT", "600"))
    # Auto-retries on transient errors. Overridable via env
    # PACT_LLM_MAX_RETRIES. This is the number of retries after the first
    # failed attempt, so 10 means at most 11 total attempts.
    max_retries: int = int(os.environ.get("PACT_LLM_MAX_RETRIES", "10"))
    retry_base_delay: float = float(os.environ.get("PACT_LLM_RETRY_BASE_DELAY", "1"))
    retry_max_delay: float = float(os.environ.get("PACT_LLM_RETRY_MAX_DELAY", "60"))
    # For Qwen3+ family on DashScope. Default OFF: thinking mode adds 30-100s
    # per call and breaks `tool_choice="required"`. Enable per-call via the
    # `enable_thinking=True` keyword on complete/chat/chat_raw if you need it.
    enable_thinking: bool = bool(int(os.environ.get("PACT_ENABLE_THINKING", "0")))


class LLMClient:
    def __init__(self, config: LLMConfig):
        load_env_file()
        api_key = os.environ.get(config.api_key_env)
        base_url = (
            config.base_url
            or os.environ.get("PACT_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or os.environ.get("DASHSCOPE_BASE_URL")
            or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        base_url = _clean_base_url(base_url)
        if not api_key:
            api_key = "not-needed"
        self.config = config
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=config.timeout,
            # Keep retry policy in this wrapper so behavior is explicit and
            # consistent across chat(), chat_raw(), and complete().
            max_retries=0,
        )
        self.total_usage = TokenUsage()

    # Keywords in an error body that indicate the gateway is rate-limiting
    # the request even though it returned a 400 instead of a standard 429.
    _THROTTLE_KEYWORDS = ("Throttling", "AllocationQuota", "MPE-429", "限流")
    _NON_RETRYABLE_BAD_REQUEST_KEYWORDS = (
        "BadRequest.TooLarge",
        "Exceeded limit on max bytes",
        "maximum context length",
        "context length",
        "input_tokens",
        "invalid_request",
    )

    def _is_retryable_exception(self, exc: Exception) -> bool:
        if isinstance(exc, (APITimeoutError, APIConnectionError, RateLimitError)):
            return True
        if isinstance(exc, BadRequestError):
            msg = str(exc)
            if any(kw in msg for kw in self._NON_RETRYABLE_BAD_REQUEST_KEYWORDS):
                return False
            return any(kw in msg for kw in self._THROTTLE_KEYWORDS)
        if isinstance(exc, APIStatusError):
            status_code = getattr(getattr(exc, "response", None), "status_code", None)
            return isinstance(status_code, int) and status_code >= 500
        return False

    def _retry_after_delay(self, exc: Exception) -> float | None:
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", None)
        if not headers:
            return None
        retry_after = headers.get("retry-after") or headers.get("Retry-After")
        if not retry_after:
            return None
        try:
            delay = float(retry_after)
        except (TypeError, ValueError):
            return None
        return max(0.0, min(delay, self.config.retry_max_delay))

    def _retry_delay(self, exc: Exception, retry_index: int) -> float:
        retry_after = self._retry_after_delay(exc)
        if retry_after is not None:
            return retry_after
        base = max(0.0, self.config.retry_base_delay)
        cap = max(base, self.config.retry_max_delay)
        delay = min(cap, base * (2 ** retry_index))
        return delay * random.uniform(0.8, 1.2)

    def _call_with_retry(self, **kwargs):
        """Wrap chat.completions.create with project-level retry.

        Some gateways (e.g. idealab/DashScope) return HTTP 400 with a body
        containing throttling information instead of the standard 429. The
        OpenAI SDK also has its own retry policy, but we disable it and keep
        all transient-error handling here so experiment behavior is visible.
        """
        for attempt in range(self.config.max_retries + 1):
            try:
                return self.client.chat.completions.create(**kwargs)
            except Exception as exc:
                if not self._is_retryable_exception(exc) or attempt == self.config.max_retries:
                    raise
                delay = self._retry_delay(exc, attempt)
                err_preview = str(exc)
                if len(err_preview) > 120:
                    err_preview = err_preview[:117] + "..."
                now = datetime.datetime.now().strftime("%H:%M:%S")
                print(
                    f"[{now}][llm] retryable {type(exc).__name__}: {err_preview}, "
                    f"retry {attempt + 1}/{self.config.max_retries} in {delay:.1f}s",
                    flush=True,
                )
                time.sleep(delay)
        raise RuntimeError("unreachable retry loop exit")

    def _record_usage(self, response: Any) -> None:
        """Capture token usage from an OpenAI ChatCompletion response."""
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        cached = 0
        details = getattr(usage, "prompt_tokens_details", None)
        if details:
            cached = getattr(details, "cached_tokens", 0) or 0
        self.total_usage.input_tokens += usage.prompt_tokens or 0
        self.total_usage.output_tokens += usage.completion_tokens or 0
        self.total_usage.cached_tokens += cached
        self.total_usage.requests += 1

    def get_usage(self) -> dict[str, Any]:
        """Return a snapshot of accumulated token usage."""
        return {
            "model": self.config.model,
            "input_tokens": self.total_usage.input_tokens,
            "output_tokens": self.total_usage.output_tokens,
            "cached_tokens": self.total_usage.cached_tokens,
            "requests": self.total_usage.requests,
        }

    def reset_usage(self) -> None:
        """Reset the accumulated token usage counters."""
        self.total_usage = TokenUsage()

    def _is_local(self) -> bool:
        """Check if the base_url points to a local endpoint (localhost/127.0.0.1)."""
        host = self.client.base_url.host
        return host in ("localhost", "127.0.0.1", "0.0.0.0")

    def _is_qwen(self) -> bool:
        return "qwen" in self.config.model.lower()

    def _extra_body(self, enable_thinking: bool | None = None) -> dict[str, Any]:
        """Build extra_body for Qwen models.

        Remote (DashScope): sends ``{"enable_thinking": bool}`` at the top
        level.  Local (vLLM): sends ``{"chat_template_kwargs": {"enable_thinking": bool}}``
        so the chat template can render thinking tags.

        For non-Qwen models returns an empty dict (which OpenAI SDK ignores).
        When ``enable_thinking`` is False, returns an empty dict to avoid
        sending unnecessary provider-specific fields.
        """
        if not self._is_qwen():
            return {}
        flag = self.config.enable_thinking if enable_thinking is None else enable_thinking
        if not flag:
            return {}
        if self._is_local():
            return {"chat_template_kwargs": {"enable_thinking": True}}
        return {"enable_thinking": True}

    def clean_content(self, content: str) -> str:
        """Strip thinking tags from model outputs (Qwen3 / DeepSeek style)."""
        model_lower = self.config.model.lower()
        if "qwen3" in model_lower or "deepseek" in model_lower:
            if "</think>" in content:
                content = content.split("</think>")[-1].strip()
            elif content.strip().startswith("<think>"):
                content = content.strip()[len("<think>"):].strip()
        return content

    def complete(self, system: str, user: str, *, enable_thinking: bool | None = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        response = self._call_with_retry(
            model=self.config.model,
            messages=messages,
            extra_body=self._extra_body(enable_thinking),
        )
        self._record_usage(response)
        return response.choices[0].message.content or ""

    def chat(self, messages: list[dict[str, Any]], *, enable_thinking: bool | None = None) -> str:
        """Multi-turn chat completion. Returns raw response (thinking tags preserved)."""
        response = self._call_with_retry(
            model=self.config.model,
            messages=messages,
            extra_body=self._extra_body(enable_thinking),
        )
        self._record_usage(response)
        return response.choices[0].message.content or ""

    def chat_raw(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
        enable_thinking: bool | None = None,
        max_completion_tokens: int | None = None,
    ) -> Any:
        """Multi-turn chat completion returning the full ChatCompletionMessage
        so callers can inspect `tool_calls`. Pass tools/tool_choice through
        when set."""
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "extra_body": self._extra_body(enable_thinking),
        }
        if tools is not None:
            kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        if max_completion_tokens is not None:
            kwargs["max_completion_tokens"] = max_completion_tokens
        response = self._call_with_retry(**kwargs)
        self._record_usage(response)
        return response.choices[0].message


@dataclass
class ToolLoopResult:
    """Outcome of `run_tool_loop`.

    `messages` is the full message trajectory ready to persist to disk
    (system + user + every assistant turn + every tool result, in order).
    `final_tool_call` is the terminator call the model made (or `None` if
    the harness force-defaulted). `investigations` and `loop_stats` mirror
    what gets written into `critic.json`.
    """

    messages: list[dict[str, Any]]
    final_tool_call: dict[str, Any] | None
    investigations: list[dict[str, Any]]
    loop_stats: dict[str, Any]
    error: str | None = None


def run_tool_loop(
    client: "LLMClient",
    *,
    system: str,
    initial_user: str,
    tools: list[dict[str, Any]],
    investigation_tools: set[str],
    terminator_tool: str,
    investigation_handler,
    max_investigations: int = 5,
    max_turns: int = 7,
    max_no_tool_nudges: int = 2,
    force_terminator_after_investigations: bool = True,
) -> ToolLoopResult:
    """Run an agentic loop where the model is invited to think in plain
    text and emit tool calls.

    Default `tool_choice` is ``"auto"`` so the model is free to reason in
    `content` alongside (or before) any tool call. Empirically Qwen3 with
    ``tool_choice="required"`` returns empty `content` even when the prompt
    permits it — see the smoke run analysis. With ``"auto"`` the model
    will usually do both; when it skips the tool entirely we append a user
    nudge and consume another turn (capped by ``max_no_tool_nudges``
    consecutive nudges, so a confused model surfaces fast instead of
    burning all of ``max_turns``).

    Once ``max_investigations`` have been made the next request normally
    pins ``tool_choice`` to ``terminator_tool`` — we want the hard guarantee
    on the *terminator* specifically for the agentic critic. Callers that have
    no investigation tools can disable this so the first turn still uses
    ``"auto"``. If ``max_turns`` is reached without a valid terminator call,
    ``final_tool_call`` is ``None`` and ``error`` is set.
    """
    import json as _json

    messages: list[dict[str, Any]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": initial_user})

    investigations: list[dict[str, Any]] = []
    investigation_calls = 0
    forced_terminate = False
    final_call: dict[str, Any] | None = None
    error: str | None = None
    turn = 0
    no_tool_streak = 0
    total_nudges = 0

    for turn in range(1, max_turns + 1):
        if force_terminator_after_investigations and investigation_calls >= max_investigations:
            forced_terminate = True
            tool_choice = {
                "type": "function",
                "function": {"name": terminator_tool},
            }
        else:
            # "auto" lets the model produce content (reasoning) alongside
            # any tool call. The nudge below handles the case where the
            # model writes prose but forgets the tool.
            tool_choice = "auto"

        try:
            assistant_message = client.chat_raw(
                messages, tools=tools, tool_choice=tool_choice
            )
        except Exception as exc:
            error = f"chat_raw failed on turn {turn}: {exc}"
            break

        assistant_dict = _assistant_to_dict(assistant_message)
        messages.append(assistant_dict)

        tool_calls = assistant_dict.get("tool_calls") or []
        if not tool_calls:
            no_tool_streak += 1
            total_nudges += 1
            if no_tool_streak > max_no_tool_nudges:
                error = (
                    f"turn {turn}: model produced no tool_calls after "
                    f"{max_no_tool_nudges} consecutive nudges"
                )
                break
            # Append a user-role nudge and re-enter the loop. The model's
            # prose (if any) stays in the transcript so the audit log
            # captures what it was thinking before being asked to commit.
            if investigation_tools:
                nudge_content = (
                    "You didn't call a tool in the previous turn. Please call exactly one tool now: "
                    "`get_original_trajectories(edit_index=...)` to investigate a prior edit, "
                    "or `submit_ops(ops=[...])` to finish "
                    "(use `{\"ops\": []}` if no change is justified). "
                    "You may still include reasoning text in the same turn."
                )
            else:
                nudge_content = (
                    "You didn't call a tool in the previous turn. Please call "
                    "`submit_ops(ops=[...])` now "
                    "(use `{\"ops\": []}` if no change is justified). "
                    "You may still include reasoning text in the same turn."
                )
            messages.append({
                "role": "user",
                "content": nudge_content,
            })
            continue
        no_tool_streak = 0

        # The agentic protocol expects exactly one tool call per turn. If a
        # model emits several, we honor only the first and ignore the rest;
        # the SDK still sends them all back to the model as the assistant
        # turn so future requests stay consistent. We do dispatch results
        # for any investigation calls though, to keep the transcript valid.
        terminated_this_turn = False
        for call in tool_calls:
            name = call["function"]["name"]
            raw_args = call["function"].get("arguments", "") or ""
            try:
                args = _json.loads(raw_args) if raw_args else {}
            except _json.JSONDecodeError as exc:
                args = {}
                tool_result_str = _json.dumps({"error": f"could not parse arguments: {exc}"})
                messages.append({
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "name": name,
                    "content": tool_result_str,
                })
                continue

            if name == terminator_tool:
                final_call = {"id": call["id"], "name": name, "arguments": args}
                terminated_this_turn = True
                break

            if name in investigation_tools:
                investigation_calls += 1
                try:
                    result_str = investigation_handler(name, args)
                    if not isinstance(result_str, str):
                        result_str = _json.dumps(result_str, ensure_ascii=False)
                except Exception as exc:
                    result_str = _json.dumps({"error": str(exc)})
                investigations.append({
                    "turn": turn,
                    "name": name,
                    "args": args,
                    "result_bytes": len(result_str),
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "name": name,
                    "content": result_str,
                })
            else:
                # Unknown tool — reply with an error so the model can recover.
                err_str = _json.dumps({"error": f"unknown tool {name!r}"})
                messages.append({
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "name": name,
                    "content": err_str,
                })

        if terminated_this_turn:
            break

    loop_stats = {
        "total_turns": turn,
        "investigation_calls": investigation_calls,
        "forced_terminate": forced_terminate,
        "max_investigations": max_investigations,
        "max_turns": max_turns,
        "no_tool_nudges": total_nudges,
        "max_no_tool_nudges": max_no_tool_nudges,
    }

    if final_call is None and error is None:
        error = f"hit hard cap of {max_turns} turns without {terminator_tool} call"

    return ToolLoopResult(
        messages=messages,
        final_tool_call=final_call,
        investigations=investigations,
        loop_stats=loop_stats,
        error=error,
    )


def usage_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """Compute the token usage delta between two get_usage() snapshots."""
    return {
        "model": after.get("model", ""),
        "input_tokens": after.get("input_tokens", 0) - before.get("input_tokens", 0),
        "output_tokens": after.get("output_tokens", 0) - before.get("output_tokens", 0),
        "cached_tokens": after.get("cached_tokens", 0) - before.get("cached_tokens", 0),
        "requests": after.get("requests", 0) - before.get("requests", 0),
    }


def merge_usage(accumulated: dict[str, Any] | None, delta: dict[str, Any]) -> dict[str, Any]:
    """Add a usage delta into an accumulated usage dict (mutates accumulated)."""
    if accumulated is None:
        return {
            "model": delta.get("model", ""),
            "input_tokens": delta.get("input_tokens", 0),
            "output_tokens": delta.get("output_tokens", 0),
            "cached_tokens": delta.get("cached_tokens", 0),
            "requests": delta.get("requests", 0),
        }
    accumulated["input_tokens"] += delta.get("input_tokens", 0)
    accumulated["output_tokens"] += delta.get("output_tokens", 0)
    accumulated["cached_tokens"] += delta.get("cached_tokens", 0)
    accumulated["requests"] += delta.get("requests", 0)
    return accumulated


def _assistant_to_dict(message: Any) -> dict[str, Any]:
    """Convert an OpenAI ChatCompletionMessage to a JSON-safe dict suitable
    both for the message history sent back to the model and for persisting
    in critic.json."""
    payload: dict[str, Any] = {"role": "assistant", "content": message.content}
    tool_calls = getattr(message, "tool_calls", None) or []
    if tool_calls:
        payload["tool_calls"] = [
            {
                "id": call.id,
                "type": getattr(call, "type", "function"),
                "function": {
                    "name": call.function.name,
                    "arguments": call.function.arguments or "",
                },
            }
            for call in tool_calls
        ]
    return payload


def parse_final_answer(text: str) -> str:
    match = re.search(r"FINAL_ANSWER\s*:\s*(.+)", text, flags=re.I | re.S)
    if match:
        return match.group(1).strip()
    return text.strip()


def parse_json_object(text: str) -> dict[str, Any] | None:
    import json

    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None
