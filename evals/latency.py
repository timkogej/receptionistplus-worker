"""Time-to-first-token of the conversation LLM with the real prompt and tools.

Phone latency is dominated by how fast the first words (or the first tool
call) come back, so this measures exactly that per model config, on typical
first caller turns, with prompt caching warm after the first request.

    .venv/bin/python -m evals.latency
    .venv/bin/python -m evals.latency --configs claude-haiku-5-5@low,claude-haiku-5-5@medium --rounds 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

from livekit.agents import llm

from agent.call_notes import CallNotes
from agent.llm_factory import build_single_llm
from agent.tools import BookingTools
from agent.worker import build_receptionist_agent
from evals.run_evals import FIXTURE, _load_env

TURNS = [
    "Živjo, kako si?",
    "Koliko stane pedikura?",
    "Kaj vse ponujate?",
    "Rada bi masažo glave pri Maji.",
    "Imate jutri dopoldan prosto za pedikuro? Vseeno mi je, kdo.",
]

DEFAULT_CONFIGS = "claude-haiku-4-5,claude-haiku-5-5@low,claude-haiku-5-5@medium,openai:gpt-5.4-mini"


def _parse(config: str) -> tuple[str, str | None]:
    model, _, effort = config.partition("@")
    return model, (effort or None)


async def _first_token(model_llm, agent, user_text: str) -> tuple[float, float, str]:
    ctx = llm.ChatContext()
    ctx.add_message(role="system", content=agent.instructions)
    ctx.add_message(role="user", content=user_text)
    started = time.monotonic()
    first = None
    kind = "text"
    async with model_llm.chat(chat_ctx=ctx, tools=list(agent.tools)) as stream:
        async for chunk in stream:
            if chunk.delta and (chunk.delta.content or chunk.delta.tool_calls):
                if first is None:
                    first = time.monotonic() - started
                    kind = "tool" if chunk.delta.tool_calls else "text"
    return first if first is not None else float("nan"), time.monotonic() - started, kind


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", default=DEFAULT_CONFIGS)
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()
    _load_env()

    company = json.loads(Path(FIXTURE).read_text())
    agent = build_receptionist_agent(
        company, "sl", booking_tools=BookingTools("eval-company", company), notes=CallNotes()
    )

    for config in args.configs.split(","):
        model, effort = _parse(config)
        model_llm = build_single_llm(model, effort=effort)
        await _first_token(model_llm, agent, TURNS[0])  # warm the prompt cache
        ttfts, totals = [], []
        for _ in range(args.rounds):
            for turn in TURNS:
                try:
                    ttft, total, _kind = await _first_token(model_llm, agent, turn)
                except Exception as exc:
                    print(f"  {config}: error on {turn!r}: {exc!r}")
                    continue
                ttfts.append(ttft)
                totals.append(total)
        if not ttfts:
            continue
        ttfts.sort()
        p90 = ttfts[max(0, int(len(ttfts) * 0.9) - 1)]
        print(
            f"{config:32s} first token p50={statistics.median(ttfts):.2f}s "
            f"p90={p90:.2f}s max={max(ttfts):.2f}s  full reply p50={statistics.median(totals):.2f}s"
            f"  (n={len(ttfts)})"
        )


if __name__ == "__main__":
    asyncio.run(main())
