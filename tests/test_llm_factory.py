"""LLM wrapper rules (agent/llm_factory.py). Needs the project venv; no
network: nothing here sends a request."""

from __future__ import annotations

import os
import unittest
from types import SimpleNamespace

os.environ.setdefault("ANTHROPIC_API_KEY", "test")
os.environ.setdefault("OPENAI_API_KEY", "test")

try:
    from agent import llm_factory as F
    from livekit.plugins.anthropic import llm as anthropic_llm
except ImportError:  # pragma: no cover
    F = None


@unittest.skipIf(F is None, "livekit-agents not installed")
class RequestShape(unittest.TestCase):
    def test_haiku_55_gets_thinking_off_and_effort(self):
        m = F.ClaudeLLM(model="claude-haiku-5-5", effort="low")
        self.assertEqual(
            m._request_body,
            {"thinking": {"type": "disabled"}, "output_config": {"effort": "low"}},
        )

    def test_haiku_45_request_unchanged(self):
        self.assertEqual(F.ClaudeLLM(model="claude-haiku-4-5", effort="low")._request_body, {})

    def test_5x_models_never_prefill(self):
        self.assertTrue(anthropic_llm._model_disables_prefill("claude-haiku-5-5"))
        self.assertFalse(anthropic_llm._model_disables_prefill("claude-haiku-4-5"))

    def test_refusal_is_raised(self):
        stream = object.__new__(F._RefusalAwareStream)
        event = SimpleNamespace(type="message_delta", delta=SimpleNamespace(stop_reason="refusal"))
        with self.assertRaises(F.ModelRefusal):
            stream._parse_event(event)


@unittest.skipIf(F is None, "livekit-agents not installed")
class Config(unittest.TestCase):
    def test_env(self):
        env = {"LLM_MODEL": "claude-haiku-4-5", "LLM_EFFORT": "", "LLM_FALLBACK_MODEL": ""}
        old = {k: os.environ.get(k) for k in env}
        os.environ.update(env)
        try:
            c = F.LLMConfig.from_env()
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        self.assertEqual((c.primary_model, c.effort, c.fallback_model), ("claude-haiku-4-5", None, None))

    def test_openai_prefix_and_fallback_chain(self):
        built = F.build_llm(F.LLMConfig())
        models = [i.model for i in built._llm_instances]
        self.assertEqual(models, ["claude-haiku-5-5", "gpt-5.4-mini"])


if __name__ == "__main__":
    unittest.main()
