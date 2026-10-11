"""LLM clients (Anthropic / OpenAI), FakeLLM for tests, and make_llm()."""
from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Callable

from iz.types import LLM, LLMResponse

# ESTIMATES: USD per 1M tokens (input, output). Not authoritative billing data.
PRICES: dict[str, tuple[float, float]] = {
    "claude-opus": (15.0, 75.0),
    "claude-sonnet": (3.0, 15.0),
    "claude-haiku": (1.0, 5.0),
    "gpt-4o-mini": (0.15, 0.6),
    "gpt-4o": (2.5, 10.0),
    "gpt-4.1-mini": (0.4, 1.6),
    "gpt-4.1": (2.0, 8.0),
    "default": (3.0, 15.0),
}


class LLMError(Exception):
    pass


def estimate_cost(model: str, in_tok: int, out_tok: int) -> float:
    """Estimated USD cost; longest-prefix match on PRICES, fallback to default."""
    best, price = "", PRICES["default"]
    for prefix, p in PRICES.items():
        if prefix != "default" and model.startswith(prefix) and len(prefix) > len(best):
            best, price = prefix, p
    return (in_tok * price[0] + out_tok * price[1]) / 1_000_000


def _report(model: str, cost: float, i: int, o: int) -> None:
    """Record spend in the process budget. BudgetExceeded propagates (the MAX_LLM_USD guard)."""
    try:
        from iz.obs.budget import get_budget
    except Exception:  # noqa: BLE001
        return
    get_budget().add(cost, model)


def _with_retry(call: Callable[[], LLMResponse]) -> LLMResponse:
    last: Exception | None = None
    for attempt in range(1, 4):
        try:
            resp = call()
        except Exception as e:  # noqa: BLE001
            last = e
            if attempt < 3:
                time.sleep(0.5 * attempt)
            continue
        _report(resp.model, resp.cost_usd, resp.input_tokens, resp.output_tokens)
        return resp
    raise LLMError(f"LLM call failed after 3 attempts: {type(last).__name__}: {last}")


class AnthropicLLM:
    def __init__(self) -> None:
        import anthropic
        self._anthropic = anthropic
        self.client = anthropic.Anthropic()

    def complete_json(self, model: str, system: str, user: str, schema: dict[str, Any]) -> LLMResponse:
        def call() -> LLMResponse:
            kw: dict[str, Any] = dict(
                model=model, max_tokens=1024, system=system,
                messages=[{"role": "user", "content": user}],
                tools=[{"name": "respond", "description": "Return the answer", "input_schema": schema}],
                tool_choice={"type": "tool", "name": "respond"},
            )
            msg = self._create(kw)
            data = _extract_json(msg)
            i, o = msg.usage.input_tokens, msg.usage.output_tokens
            return LLMResponse(data=dict(data), model=model, input_tokens=i, output_tokens=o,
                               cost_usd=estimate_cost(model, i, o))
        return _with_retry(call)

    # Some models reject `temperature` (TypeError in newer SDKs, 400 in others) and some
    # reject forced tool_choice. Remember what failed so later calls skip it.
    _no_temp: bool = False
    _auto_tool: bool = False

    def _create(self, kw: dict[str, Any]) -> Any:
        for _ in range(3):
            args = dict(kw)
            if self._auto_tool:
                args["tool_choice"] = {"type": "auto"}
                args["system"] = (kw["system"] + "\n\nAlways answer by calling the `respond` tool "
                                  "(or, failing that, with a single JSON object only).")
            if not self._no_temp:
                args["temperature"] = 0
            try:
                return self.client.messages.create(**args)
            except TypeError as e:
                if "temperature" in str(e) and not self._no_temp:
                    self._no_temp = True
                    continue
                raise
            except self._anthropic.BadRequestError as e:
                s = str(e)
                if "temperature" in s and not self._no_temp:
                    self._no_temp = True
                    continue
                if "tool_choice" in s and not self._auto_tool:
                    self._auto_tool = True
                    continue
                raise
        return self.client.messages.create(**args)


def _extract_json(msg: Any) -> dict:
    for b in msg.content:
        if getattr(b, "type", "") == "tool_use":
            return dict(b.input)
    text = "".join(getattr(b, "text", "") for b in msg.content if getattr(b, "type", "") == "text")
    s, e = text.find("{"), text.rfind("}")
    if s >= 0 and e > s:
        return json.loads(text[s:e + 1])
    raise LLMError("no tool_use block or JSON object in response")


class OpenAILLM:
    def __init__(self) -> None:
        import openai
        self.client = openai.OpenAI()

    def complete_json(self, model: str, system: str, user: str, schema: dict[str, Any]) -> LLMResponse:
        def call() -> LLMResponse:
            r = self.client.chat.completions.create(
                model=model, temperature=0,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                response_format={"type": "json_schema",
                                 "json_schema": {"name": "respond", "schema": schema, "strict": False}},
            )
            data = json.loads(r.choices[0].message.content)
            i = getattr(r.usage, "prompt_tokens", 0) or 0
            o = getattr(r.usage, "completion_tokens", 0) or 0
            return LLMResponse(data=data, model=model, input_tokens=i, output_tokens=o,
                               cost_usd=estimate_cost(model, i, o))
        return _with_retry(call)


class FakeLLM:
    """Scripted or policy-driven LLM for tests. Thread-safe."""

    def __init__(self, responses: list[dict] | None = None,
                 policy: Callable[[str, str, str, dict], dict] | None = None) -> None:
        self.responses = list(responses or [])
        self.policy = policy
        self.calls: list[dict] = []
        self.fail_times = 0
        self._i = 0
        self._lock = threading.Lock()

    def complete_json(self, model: str, system: str, user: str, schema: dict[str, Any]) -> LLMResponse:
        with self._lock:
            self.calls.append({"model": model, "system": system, "user": user, "schema": schema})
            if self.fail_times > 0:
                self.fail_times -= 1
                raise LLMError("FakeLLM scripted failure")
            if self.policy is not None:
                data = self.policy(model, system, user, schema)
            elif self.responses:
                data = self.responses[min(self._i, len(self.responses) - 1)]
                self._i += 1
            else:
                data = {}
        return LLMResponse(data=data, model=model, cost_usd=0)


def robot_model() -> str:
    return os.getenv("ROBOT_MODEL", "claude-sonnet-4-5")


def fixer_model() -> str:
    return os.getenv("FIXER_MODEL", "claude-opus-4-1")


def make_llm(cached: bool = True) -> LLM:
    from dotenv import load_dotenv
    load_dotenv()
    provider = os.getenv("LLM_PROVIDER", "anthropic").lower()
    if provider == "anthropic":
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise LLMError("ANTHROPIC_API_KEY is not set (LLM_PROVIDER=anthropic)")
        llm: LLM = AnthropicLLM()
    elif provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise LLMError("OPENAI_API_KEY is not set (LLM_PROVIDER=openai)")
        llm = OpenAILLM()
    else:
        raise LLMError(f"unknown LLM_PROVIDER {provider!r}")
    if cached:
        from iz.llm.cache import CachedLLM
        return CachedLLM(llm)
    return llm
