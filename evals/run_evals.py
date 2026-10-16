"""Run the receptionist against evals/scenarios.py, text only.

Uses the real prompt, tools and LLM (so it costs a little: a full run on
Claude Haiku is a few cents), with the booking webhook replaced by
evals/fake_booking.py — nothing is ever booked for real.

    .venv/bin/python -m evals.run_evals                       # all, current model
    .venv/bin/python -m evals.run_evals --model claude-haiku-4-5 --repeat 3
    .venv/bin/python -m evals.run_evals --model claude-haiku-5-5@medium
    .venv/bin/python -m evals.run_evals --model openai:gpt-5.4-mini
    .venv/bin/python -m evals.run_evals --only sl-vikanje,en-price

Needs ANTHROPIC_API_KEY (read from .env if present). Results are printed and
written to evals/results/<timestamp>-<model>.json for before/after
comparisons (e.g. a model or prompt change).
"""

from __future__ import annotations

import argparse
import re
import asyncio
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path

import anthropic as anthropic_sdk
from livekit.agents import AgentSession
from livekit.agents import llm as llm_mod
from livekit.agents.voice.run_result import ChatMessageEvent, FunctionCallEvent
from livekit.plugins import anthropic

from agent.call_notes import CallNotes
from agent.llm_factory import build_single_llm
from agent.tools import BookingTools
from agent.worker import build_receptionist_agent
from evals.fake_booking import FakeBooking
from evals.scenarios import SCENARIOS, Scripted, Simulated

ROOT = Path(__file__).resolve().parent
FIXTURE = ROOT / "fixtures" / "company_jedroplus.json"
RESULTS = ROOT / "results"

# "<model>[@effort]"; "openai:<model>" for OpenAI. Same strings as LLM_MODEL.
DEFAULT_MODEL = "claude-haiku-5-5@low"
# A judge that misreads Slovenian passes or fails the wrong things (Haiku 4.5
# read "petinštirideset minut" as "thirty-four minutes").
DEFAULT_JUDGE = "claude-haiku-5-5@medium"
DEFAULT_CALLER = "claude-haiku-4-5"

CALLER_RULES = (
    "\n\nOdgovarjaš kot klicatelj po telefonu: samo to, kar rečeš, en kratek "
    "stavek ali dva, brez opisov dejanj. Ko je rezervacija potrjena in se "
    "receptorka poslovi, odgovori natanko: KONEC"
)


# Deterministic checks on every Slovenian reply, independent of the judge.
# "štirideset pet" is how an English-thinking model builds 45; Slovenian puts
# the unit first and joins it: "petinštirideset". Same for every tens word.
_TENS = "dvajset|trideset|štirideset|petdeset|šestdeset|sedemdeset|osemdeset|devetdeset"
_UNITS = "ena|en|dva|dve|tri|štiri|pet|šest|sedem|osem|devet"
SL_TEXT_CHECKS = [
    (re.compile(rf"\b(?:{_TENS})\s+(?:in\s+)?(?:{_UNITS})\b", re.I),
     "number word in English order (e.g. 'štirideset pet' instead of 'petinštirideset')"),
    (re.compile(r"\d"), "digits in spoken text (TTS reads them badly)"),
    (re.compile(r"[*#]|^\s*-\s", re.M), "markdown in spoken text"),
]


def _text_failures(language: str, turn_no: int, text: str) -> list[str]:
    if language != "sl":
        return []
    return [
        f"turn {turn_no}: {why}: {m.group(0)!r}"
        for pattern, why in SL_TEXT_CHECKS
        if (m := pattern.search(text))
    ]


_JUDGE_SYSTEM = (
    "You grade one reply of an AI phone receptionist against an intent. The "
    "reply may be in Slovenian; read it carefully in that language. First "
    "reason in two to four sentences, checking each requirement of the "
    "intent against the reply. Then end with exactly one final line: "
    "VERDICT: PASS or VERDICT: FAIL."
)


async def _judge(judge_llm, reply: str, intent: str) -> tuple[bool, str]:
    """Reason first, verdict last. LiveKit's built-in judge() forces a tool
    call, which makes Claude 5.x decide before reasoning — it then failed
    correct replies and argued itself back to "success" in the reason."""
    ctx = llm_mod.ChatContext()
    ctx.add_message(role="system", content=_JUDGE_SYSTEM)
    ctx.add_message(role="user", content=f"Intent:\n{intent}\n\nReply:\n{reply}")
    parts = []
    async with judge_llm.chat(chat_ctx=ctx) as stream:
        async for chunk in stream:
            if chunk.delta and chunk.delta.content:
                parts.append(chunk.delta.content)
    text = "".join(parts).strip()
    verdict = text.rsplit("VERDICT:", 1)[-1].strip().upper() if "VERDICT:" in text else ""
    return verdict.startswith("PASS"), text


def _load_env() -> None:
    env = ROOT.parent / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key, value)


def _build(language: str, model: str, context: dict | None = None):
    company = json.loads(FIXTURE.read_text())
    booking_tools = BookingTools("eval-company", company, language=language)
    notes = CallNotes()
    agent = build_receptionist_agent(
        company, language, booking_tools=booking_tools, notes=notes, **(context or {})
    )
    name, _, effort = model.partition("@")
    llm = build_single_llm(name, effort=effort or None)
    return agent, llm, booking_tools, notes


def _tool_names(result) -> list[str]:
    return [e.item.name for e in result.events if isinstance(e, FunctionCallEvent)]


def _assistant_text(result) -> str:
    return " ".join(
        e.item.text_content or ""
        for e in result.events
        if isinstance(e, ChatMessageEvent) and e.item.role == "assistant"
    ).strip()


async def run_scripted(sc: Scripted, model: str, judge_llm) -> dict:
    agent, llm, _tools, _notes = _build(sc.language, model, sc.context)
    fake = FakeBooking()
    failures: list[str] = []
    transcript: list[dict] = []
    with fake.installed():
        async with llm, AgentSession(llm=llm) as session:
            await session.start(agent)
            for n, turn in enumerate(sc.turns, 1):
                result = await session.run(user_input=turn.user)
                names = _tool_names(result)
                reply = _assistant_text(result)
                transcript.append({"user": turn.user, "tools": names, "agent": reply})
                failures += _text_failures(sc.language, n, reply)
                for t in turn.expect_tools:
                    if t not in names:
                        failures.append(f"turn {n}: expected tool {t}, got {names}")
                for t in turn.forbid_tools:
                    if t in names:
                        failures.append(f"turn {n}: forbidden tool {t} was called")
                if turn.judge:
                    if not reply:
                        failures.append(f"turn {n}: no assistant reply to judge")
                    else:
                        ok, reasoning = await _judge(judge_llm, reply, turn.judge)
                        if not ok:
                            failures.append(f"turn {n}: judge: {' '.join(reasoning.split())}")
    return {"failures": failures, "transcript": transcript}


async def run_simulated(sc: Simulated, model: str, caller_model: str) -> dict:
    agent, llm, _tools, notes = _build(sc.language, model, sc.context)
    fake = FakeBooking()
    caller = anthropic_sdk.AsyncAnthropic()
    # From the simulated caller's side, the receptionist is the "user".
    caller_msgs: list[dict] = [{"role": "user", "content": "(telefon zazvoni, receptorka se oglasi)"}]
    transcript: list[dict] = []
    user_line = sc.opening
    text_failures: list[str] = []
    with fake.installed():
        async with llm, AgentSession(llm=llm) as session:
            await session.start(agent)
            for _ in range(sc.max_turns):
                caller_msgs.append({"role": "assistant", "content": user_line})
                result = await session.run(user_input=user_line)
                reply = _assistant_text(result) or "(tišina)"
                transcript.append({"user": user_line, "tools": _tool_names(result), "agent": reply})
                text_failures += _text_failures(sc.language, len(transcript), reply)
                stored = ((sc.context.get("customer") or {}).get("Email stranke") or "").lower()
                if stored and stored.split("@")[0] in reply.lower():
                    text_failures.append(f"turn {len(transcript)}: read the stored email out loud")
                caller_msgs.append({"role": "user", "content": reply})
                if fake.created and any(
                    w in reply.lower() for w in ("lep dan", "nasvidenje", "goodbye", "have a")
                ):
                    break
                resp = await caller.messages.create(
                    model=caller_model,
                    max_tokens=200,
                    system=sc.persona + CALLER_RULES,
                    messages=caller_msgs,
                )
                user_line = "".join(b.text for b in resp.content if b.type == "text").strip()
                if user_line == "KONEC":
                    break
    return {"failures": sc.check(fake, notes) + text_failures, "transcript": transcript}


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.environ.get("EVAL_MODEL", DEFAULT_MODEL))
    parser.add_argument("--judge", default=os.environ.get("EVAL_JUDGE_MODEL", DEFAULT_JUDGE))
    parser.add_argument("--caller", default=os.environ.get("EVAL_CALLER_MODEL", DEFAULT_CALLER))
    parser.add_argument("--only", default="", help="comma-separated scenario ids")
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()

    _load_env()
    logging.basicConfig(level=getattr(logging, os.environ.get("EVAL_LOG", "WARNING")))
    only = {s for s in args.only.split(",") if s}
    scenarios = [s for s in SCENARIOS if not only or s.id in only]
    judge_name, _, judge_effort = args.judge.partition("@")
    judge_llm = build_single_llm(judge_name, effort=judge_effort or None)

    rows = []
    for sc in scenarios:
        for rep in range(args.repeat):
            started = time.monotonic()
            try:
                if isinstance(sc, Scripted):
                    out = await run_scripted(sc, args.model, judge_llm)
                else:
                    out = await run_simulated(sc, args.model, args.caller)
            except Exception as exc:  # a crash is a failure, not an abort
                out = {"failures": [f"crashed: {exc!r}"], "transcript": []}
            out.update(id=sc.id, rep=rep, seconds=round(time.monotonic() - started, 1))
            rows.append(out)
            status = "PASS" if not out["failures"] else "FAIL"
            print(f"{status}  {sc.id}  ({out['seconds']}s)")
            for f in out["failures"]:
                print(f"      - {f}")

    passed = sum(1 for r in rows if not r["failures"])
    print(f"\n{passed}/{len(rows)} passed  model={args.model}")

    RESULTS.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = RESULTS / f"{stamp}-{args.model.replace(':', '_').replace('@', '_')}.json"
    out_path.write_text(
        json.dumps(
            {"model": args.model, "judge": args.judge, "caller": args.caller, "results": rows},
            ensure_ascii=False,
            indent=1,
        )
    )
    print(f"written {out_path.relative_to(ROOT.parent)}")
    return 0 if passed == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
