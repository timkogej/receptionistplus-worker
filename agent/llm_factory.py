"""Builds the conversation LLM (primary + fallback) for the worker and evals.

Why a wrapper around livekit-plugins-anthropic (1.6.9) is needed for Claude
Haiku 5.5, all checked against the plugin source:

- Thinking is ON by default on Haiku 5.5 and the plugin has no parameter for
  it. Thinking before every reply costs latency a phone call can't afford,
  and the plugin drops thinking blocks instead of sending them back — which
  Haiku 5.5 expects on tool-use turns. So thinking is switched off
  explicitly, and `effort` (default "medium" on 5.5) is set from config.
  Both go in `extra_body`, the plugin's passthrough to the request JSON.
- A safety-classifier decline comes back as a normal stream ending in
  stop_reason "refusal", with no text. The plugin ignores stop_reason, so the
  caller would hear nothing. Here it is raised as an error before any text is
  sent, which lets FallbackAdapter answer with the fallback model instead.
- Haiku 5.5 rejects temperature/top_p/top_k; they are dropped per request.
- Haiku 5.5 rejects a trailing assistant message (prefill) with a 400. The
  plugin only knows that about Sonnet/Opus 4.6 and adds the trailing user
  message for those; the list is extended below.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from livekit.agents import llm
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given
from livekit.plugins import anthropic, openai
from livekit.plugins.anthropic import llm as _anthropic_llm

logger = logging.getLogger("receptionistplus-worker.llm")

DEFAULT_PRIMARY_MODEL = "claude-haiku-5-5"
DEFAULT_EFFORT = "low"
DEFAULT_FALLBACK_MODEL = "openai:gpt-5.4-mini"

# Models that take `thinking` / `output_config.effort` in the request. Haiku
# 4.5 rejects `effort`, so the old model keeps getting a plain request.
_THINKING_CONTROL_PREFIXES = ("claude-haiku-5", "claude-sonnet-5", "claude-opus-5")

# See the module docstring: models that reject assistant prefill.
_extra_no_prefill = ("claude-haiku-5", "claude-sonnet-5", "claude-opus-5", "claude-fable-5")
_anthropic_llm._NO_PREFILL_PATTERNS = tuple(
    dict.fromkeys(_anthropic_llm._NO_PREFILL_PATTERNS + _extra_no_prefill)
)


class ModelRefusal(RuntimeError):
    """The model's safety classifier declined the request (stop_reason
    "refusal"). Raised so FallbackAdapter moves on to the next model."""


class _RefusalAwareStream(_anthropic_llm.LLMStream):
    def _parse_event(self, event):
        if event.type == "message_delta" and getattr(event.delta, "stop_reason", None) == "refusal":
            logger.warning("model declined the request (stop_reason=refusal)")
            raise ModelRefusal("stop_reason=refusal")
        return super()._parse_event(event)


class ClaudeLLM(anthropic.LLM):
    """anthropic.LLM with thinking switched off, effort set, and refusals
    surfaced as errors. For models outside _THINKING_CONTROL_PREFIXES it
    behaves exactly like the plugin's LLM."""

    def __init__(self, *, model: str, effort: str | None = DEFAULT_EFFORT, **kwargs) -> None:
        super().__init__(model=model, **kwargs)
        self._request_body: dict = {}
        if model.startswith(_THINKING_CONTROL_PREFIXES):
            self._request_body["thinking"] = {"type": "disabled"}
            if effort:
                self._request_body["output_config"] = {"effort": effort}

    def chat(self, *, extra_kwargs: NotGivenOr[dict] = NOT_GIVEN, **kwargs):
        extra = dict(extra_kwargs) if is_given(extra_kwargs) else {}
        if self._request_body:
            # The 5.x models reject non-default sampling parameters with a
            # 400 ("`temperature` is deprecated for this model"); LiveKit's
            # own judge() sends temperature=0, and other framework helpers may.
            for key in ("temperature", "top_p", "top_k"):
                if extra.pop(key, None) is not None:
                    logger.debug("dropped %s for %s", key, self.model)
            extra["extra_body"] = {**self._request_body, **(extra.get("extra_body") or {})}
        stream = super().chat(extra_kwargs=extra, **kwargs)
        stream.__class__ = _RefusalAwareStream
        return stream


@dataclass(frozen=True)
class LLMConfig:
    primary_model: str = DEFAULT_PRIMARY_MODEL
    effort: str | None = DEFAULT_EFFORT
    fallback_model: str | None = DEFAULT_FALLBACK_MODEL

    @classmethod
    def from_env(cls) -> "LLMConfig":
        fallback = os.environ.get("LLM_FALLBACK_MODEL", DEFAULT_FALLBACK_MODEL).strip()
        effort = os.environ.get("LLM_EFFORT", DEFAULT_EFFORT).strip()
        return cls(
            primary_model=os.environ.get("LLM_MODEL", DEFAULT_PRIMARY_MODEL).strip(),
            effort=effort or None,
            fallback_model=fallback or None,
        )


def build_single_llm(
    model: str,
    *,
    effort: str | None = DEFAULT_EFFORT,
    anthropic_api_key: str | None = None,
    openai_api_key: str | None = None,
) -> llm.LLM:
    """One LLM from a model string. "openai:<model>" selects OpenAI; anything
    else is a Claude model id."""
    if model.startswith("openai:"):
        kwargs = {"api_key": openai_api_key} if openai_api_key else {}
        return openai.LLM(model=model.split(":", 1)[1], **kwargs)
    kwargs = {"api_key": anthropic_api_key} if anthropic_api_key else {}
    return ClaudeLLM(model=model, effort=effort, caching="ephemeral", **kwargs)


def build_llm(
    config: LLMConfig,
    *,
    anthropic_api_key: str | None = None,
    openai_api_key: str | None = None,
) -> llm.LLM:
    primary = build_single_llm(
        config.primary_model,
        effort=config.effort,
        anthropic_api_key=anthropic_api_key,
        openai_api_key=openai_api_key,
    )
    fallback_model = config.fallback_model
    if not fallback_model:
        return primary
    if fallback_model.startswith("openai:") and not (
        openai_api_key or os.environ.get("OPENAI_API_KEY")
    ):
        logger.warning("no OPENAI_API_KEY — running without an LLM fallback")
        return primary
    fallback = build_single_llm(
        fallback_model,
        effort=config.effort,
        anthropic_api_key=anthropic_api_key,
        openai_api_key=openai_api_key,
    )
    # Outage/refusal safety net only, not load balancing: the fallback runs
    # only after the primary has failed outright. attempt_timeout is how long
    # to wait for Claude's first token before giving up on it: 5 s (the SDK
    # default) cost a live call five silent seconds on 2026-10-17. Claude's
    # first token is ~1-2 s even with a cold cache, so 3.5 s still leaves room.
    return llm.FallbackAdapter([primary, fallback], attempt_timeout=3.5)
