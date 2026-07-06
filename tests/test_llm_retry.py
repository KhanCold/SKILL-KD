from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx
import pytest
from openai import APIStatusError, APITimeoutError, BadRequestError

from pact import llm as llm_module
from pact.llm import LLMClient, LLMConfig


@dataclass
class FakeCompletionClient:
    outcomes: list[Any]
    calls: int = 0

    def create(self, **_kwargs):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeChatClient:
    def __init__(self, completions: FakeCompletionClient):
        self.completions = completions


class FakeOpenAIClient:
    def __init__(self, completions: FakeCompletionClient):
        self.chat = FakeChatClient(completions)


def _request() -> httpx.Request:
    return httpx.Request("POST", "https://example.test/v1/chat/completions")


def _response(status_code: int, *, headers: dict[str, str] | None = None) -> httpx.Response:
    return httpx.Response(status_code, request=_request(), headers=headers or {})


def _client(outcomes: list[Any], *, max_retries: int = 10) -> tuple[LLMClient, FakeCompletionClient]:
    completions = FakeCompletionClient(outcomes)
    client = LLMClient.__new__(LLMClient)
    client.config = LLMConfig("test-model", max_retries=max_retries)
    client.client = FakeOpenAIClient(completions)
    return client, completions


def test_timeout_retries_with_exponential_backoff_until_success(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr(llm_module.time, "sleep", sleeps.append)

    response = object()
    client, completions = _client(
        [APITimeoutError(_request()), APITimeoutError(_request()), response]
    )

    assert client._call_with_retry(model="test-model", messages=[]) is response
    assert completions.calls == 3
    assert len(sleeps) == 2
    assert 0.8 <= sleeps[0] <= 1.2
    assert 1.6 <= sleeps[1] <= 2.4


def test_status_500_retries_but_status_400_does_not(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr(llm_module.time, "sleep", sleeps.append)

    response = object()
    client_500, completions_500 = _client(
        [APIStatusError("server error", response=_response(500), body=None), response]
    )
    assert client_500._call_with_retry(model="test-model", messages=[]) is response
    assert completions_500.calls == 2
    assert len(sleeps) == 1

    client_400, completions_400 = _client(
        [APIStatusError("bad request", response=_response(400), body=None), response]
    )
    with pytest.raises(APIStatusError):
        client_400._call_with_retry(model="test-model", messages=[])
    assert completions_400.calls == 1


def test_throttle_bad_request_retries_but_too_large_does_not(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr(llm_module.time, "sleep", sleeps.append)

    response = object()
    throttled = BadRequestError(
        "Error code: 400 - {'code': 'MPE-429', 'message': 'Throttling.AllocationQuota'}",
        response=_response(400),
        body=None,
    )
    client_throttle, completions_throttle = _client([throttled, response])
    assert client_throttle._call_with_retry(model="test-model", messages=[]) is response
    assert completions_throttle.calls == 2

    too_large = BadRequestError(
        "BadRequest.TooLarge: Exceeded limit on max bytes to request body",
        response=_response(400),
        body=None,
    )
    client_too_large, completions_too_large = _client([too_large, response])
    with pytest.raises(BadRequestError):
        client_too_large._call_with_retry(model="test-model", messages=[])
    assert completions_too_large.calls == 1


def test_llm_client_disables_sdk_retries(monkeypatch):
    constructed: list[dict[str, Any]] = []

    class FakeOpenAI:
        def __init__(self, **kwargs):
            constructed.append(kwargs)

    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    monkeypatch.setattr(llm_module, "OpenAI", FakeOpenAI)

    LLMClient(LLMConfig("test-model", max_retries=10))

    assert constructed
    assert constructed[0]["max_retries"] == 0
