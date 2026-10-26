"""The "Trenutek." a slow reply starts with (agent/worker.py). Needs the
worker's runtime dependencies; skipped without them."""

from __future__ import annotations

import asyncio
import unittest

try:
    from agent import worker as W
except ImportError:  # pragma: no cover - no livekit in this interpreter
    W = None


async def _stream(delay: float, parts):
    await asyncio.sleep(delay)
    for p in parts:
        yield p


def _collect(stream):
    async def go():
        out = []
        async for item in stream:
            out.append(item if isinstance(item, str) else item.delta.content)
        return out
    return asyncio.run(go())


@unittest.skipIf(W is None, "livekit not installed")
class ThinkingFiller(unittest.TestCase):
    def test_slow_reply_starts_with_the_filler(self):
        out = _collect(W._with_thinking_filler(_stream(0.2, ["Ja."]), lambda: "Trenutek.", 0.05))
        self.assertEqual(out, ["Trenutek. ", "Ja."])

    def test_fast_reply_has_none(self):
        out = _collect(W._with_thinking_filler(_stream(0.0, ["Ja.", " Seveda."]), lambda: "Trenutek.", 0.5))
        self.assertEqual(out, ["Ja.", " Seveda."])

    def test_empty_slow_stream(self):
        out = _collect(W._with_thinking_filler(_stream(0.1, []), lambda: "Trenutek.", 0.02))
        self.assertEqual(out, ["Trenutek. "])

    def test_watchdog_ignores_the_filler(self):
        self.assertEqual(W._without_thinking_filler("Samo trenutek. Masaža traja trideset minut."),
                         "Masaža traja trideset minut.")
        self.assertEqual(W._without_thinking_filler("Preverim termine."), "Preverim termine.")


if __name__ == "__main__":
    unittest.main()
