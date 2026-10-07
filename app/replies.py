"""What a model's reply goes through before it is stored and sent.

- A thinking block (`<think>...</think>`, as some models write their reasoning into the reply
  itself) is removed; a reply that is only an unfinished thinking block becomes empty and fails
  like any empty reply. Models whose engine separates the reasoning (the installed one:
  `reasoning_content`) are not changed.
- A degenerate loop (a model repeating the same sequence of words) is detected: some sequence
  of `period` words (1 to MAX_PERIOD) follows itself so that at least REPEATS copies in a row
  span MIN_LOOP_WORDS words or more. The turn then fails instead of sending the loop.

Words stand in for tokens: a word is one to a few tokens, so 20 repeated tokens are at least a
few repeated words and the thresholds are set in words.
"""

import re

THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", re.S | re.I)
UNFINISHED_THINK = re.compile(r"^\s*<think>(?!.*</think>).*", re.S | re.I)
MAX_PERIOD = 200
REPEATS = 5
MIN_LOOP_WORDS = 60


# On a long request the installed model can spend the engine's whole token cap
# (`--predict 4096`) thinking and never write the answer. Measured 2026-10-04 on a request for a
# 1500-word essay: finish_reason "length", 4096 tokens, 0 visible characters, 17,868 reasoning
# characters, 422 s; the email turn failed, was run again in full twice, then filed as failed.
# Such a reply is asked for once more without thinking (`NO_THINKING`).
NO_THINKING = {"chat_template_kwargs": {"enable_thinking": False}}


def thought_to_the_cap(body: dict) -> bool:
    """The engine stopped at its token cap before any visible text: no tool call, nothing left
    once a thinking block is removed, finish_reason "length"."""
    choice = body["choices"][0]
    message = choice.get("message") or {}
    return (
        choice.get("finish_reason") == "length"
        and not message.get("tool_calls")
        and not strip_thinking(message.get("content") or "")
    )


def thinking_is_off(fields: dict) -> bool:
    return (fields.get("chat_template_kwargs") or {}).get("enable_thinking") is False


class EmptyReplyError(RuntimeError):
    """The model's reply has no visible text: the turn is failed, never sent empty."""


class DegenerateReplyError(RuntimeError):
    """The model's reply repeats itself in a loop; the turn fails instead of sending it."""


def strip_thinking(text: str) -> str:
    text = THINK_BLOCK.sub("", text or "")
    return UNFINISHED_THINK.sub("", text).strip()


def repetition(text: str) -> tuple[int, int] | None:
    """(period, copies) of the longest loop of the reply, or None when there is none."""
    words = (text or "").split()
    best = None
    for period in range(1, min(MAX_PERIOD, len(words) // REPEATS) + 1):
        run = 0
        for i in range(period, len(words)):
            run = run + 1 if words[i] == words[i - period] else 0
            copies = run // period + 1
            if copies >= REPEATS and copies * period >= MIN_LOOP_WORDS:
                if best is None or copies * period > best[0] * best[1]:
                    best = (period, copies)
    return best


def check(text: str) -> str:
    """The reply to store and send: thinking removed; DegenerateReplyError on a loop."""
    cleaned = strip_thinking(text)
    loop = repetition(cleaned)
    if loop is not None:
        raise DegenerateReplyError(
            f"the reply repeats a sequence of {loop[0]} words {loop[1]} times"
        )
    return cleaned
