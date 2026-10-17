"""What the call leaves behind for the business owner: a message and/or a
callback request.

The take_message tool lives here (not as a closure inside worker.entrypoint)
so the eval harness builds the agent with the exact same tool the live call
uses.
"""

from __future__ import annotations

import logging

from livekit.agents import RunContext, function_tool

from agent.tools import _sanitize_free_text

logger = logging.getLogger("receptionistplus-worker.notes")


class CallNotes:
    def __init__(self) -> None:
        self.message: str | None = None
        self.callback_requested: bool = False

    @property
    def has_note(self) -> bool:
        return bool(self.message) or self.callback_requested

    def request_callback(self, reason: str) -> None:
        """Code paths that tell the caller "we'll call you back" (technical
        failure, time limit, ambiguous booking) call this so the promise
        reaches the owner even though the model never called take_message.
        A message the model already took is kept."""
        self.callback_requested = True
        if not self.message:
            self.message = reason

    @function_tool
    async def take_message(
        self,
        context: RunContext,
        message: str,
        callback_requested: bool,
        callback_phone: str = "",
    ) -> dict:
        """Pass a message or callback request to the business owner — the
        only way it reaches them.

        Args:
            message: One or two plain sentences for the owner, in the
                caller's language, with their name if given.
            callback_requested: True if they want, or were promised, a call
                back.
            callback_phone: Only if they want to be called on a different
                number than the one they are calling from.
        """
        text = _sanitize_free_text(message)
        if callback_phone.strip():
            text = f"{text} (Povratni klic na: {_sanitize_free_text(callback_phone)})"
        self.message = text
        self.callback_requested = self.callback_requested or bool(callback_requested)
        logger.info(
            "message taken: callback_requested=%s len=%d",
            self.callback_requested,
            len(text),
        )
        return {
            "success": True,
            "message": (
                "Saved for the owner. Tell the caller briefly that you've "
                "passed it on"
                + (" and that they will be called back." if callback_requested else ".")
                + " Don't read the message back word for word."
            ),
        }
