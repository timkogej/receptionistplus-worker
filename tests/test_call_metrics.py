"""Per-call metrics summary (agent/call_metrics.py). stdlib only."""

from __future__ import annotations

import unittest
from types import SimpleNamespace as NS

from agent.call_metrics import CallMetrics


def LLMMetrics(**kw):
    return type("LLMMetrics", (NS,), {})(**kw)


def TTSMetrics(**kw):
    return type("TTSMetrics", (NS,), {})(**kw)


class Summary(unittest.TestCase):
    def test_percentiles_models_and_tokens(self):
        m = CallMetrics()
        for t in (0.5, 0.7, 0.9, 2.0):
            m.record(LLMMetrics(ttft=t, metadata=NS(model_name="claude-haiku-5-5"),
                                prompt_tokens=100, prompt_cached_tokens=90, completion_tokens=10))
        m.record(LLMMetrics(ttft=1.1, metadata=NS(model_name="gpt-5.4-mini"),
                            prompt_tokens=100, prompt_cached_tokens=0, completion_tokens=5))
        m.record(TTSMetrics(ttfb=0.3))
        m.record_error("llm_error")
        s = m.summary(turns=3)
        self.assertEqual(s["llm_ttft"]["n"], 5)
        self.assertEqual(s["llm_ttft"]["max"], 2.0)
        self.assertEqual(s["llm_ttft"]["p50"], 0.9)
        self.assertEqual(s["llm_models"], {"claude-haiku-5-5": 4, "gpt-5.4-mini": 1})
        self.assertEqual(s["tokens"], {"prompt": 500, "cached": 360, "completion": 45})
        self.assertEqual(s["provider_errors"], {"llm_error": 1})
        self.assertEqual(s["turns"], 3)
        self.assertIsNone(s["eou_delay"])


if __name__ == "__main__":
    unittest.main()
