"""Per-call latency and reliability numbers for receptionist_calls.metrics.

Fed from the session's metrics_collected events (the same LiveKit metrics
the worker already logs) and its error events; summarised once at the end of
the call. Duck-typed on the metric objects' fields, so tests can use plain
namespaces.
"""

from __future__ import annotations

from collections import Counter


def _percentiles(values: list[float]) -> dict | None:
    if not values:
        return None
    v = sorted(values)

    def at(q: float) -> float:
        return round(v[min(len(v) - 1, int(round(q * (len(v) - 1))))], 3)

    return {"p50": at(0.5), "p90": at(0.9), "max": round(v[-1], 3), "n": len(v)}


class CallMetrics:
    def __init__(self) -> None:
        self.llm_ttft: list[float] = []
        self.tts_ttfb: list[float] = []
        self.eou_delay: list[float] = []
        self.llm_models: Counter = Counter()
        self.prompt_tokens = 0
        self.cached_tokens = 0
        self.completion_tokens = 0
        self.provider_errors: Counter = Counter()

    def record(self, m) -> None:
        kind = type(m).__name__
        if kind == "LLMMetrics":
            if getattr(m, "ttft", None) is not None and m.ttft >= 0:
                self.llm_ttft.append(m.ttft)
            model = getattr(getattr(m, "metadata", None), "model_name", None) or getattr(m, "label", "?")
            self.llm_models[str(model)] += 1
            self.prompt_tokens += int(getattr(m, "prompt_tokens", 0) or 0)
            self.cached_tokens += int(getattr(m, "prompt_cached_tokens", 0) or 0)
            self.completion_tokens += int(getattr(m, "completion_tokens", 0) or 0)
        elif kind == "TTSMetrics":
            if getattr(m, "ttfb", None) is not None and m.ttfb >= 0:
                self.tts_ttfb.append(m.ttfb)
        elif kind == "EOUMetrics":
            if getattr(m, "end_of_utterance_delay", None) is not None:
                self.eou_delay.append(m.end_of_utterance_delay)

    def record_error(self, error_type: str) -> None:
        self.provider_errors[error_type] += 1

    def summary(self, **extra) -> dict:
        out = {
            "llm_ttft": _percentiles(self.llm_ttft),
            "tts_ttfb": _percentiles(self.tts_ttfb),
            "eou_delay": _percentiles(self.eou_delay),
            "llm_models": dict(self.llm_models),
            "tokens": {
                "prompt": self.prompt_tokens,
                "cached": self.cached_tokens,
                "completion": self.completion_tokens,
            },
            "provider_errors": dict(self.provider_errors),
        }
        out.update(extra)
        return out
