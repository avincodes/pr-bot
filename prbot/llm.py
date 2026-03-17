"""LLM client abstraction.

Supports two real providers (anthropic, openai) and a ``dry-run`` mode
that plays back scripted responses so the full agent loop can execute
end-to-end without network or API keys. The scripted mode is what the
CLI demo uses.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol


@dataclass
class LLMResponse:
    text: str
    tokens_in: int = 0
    tokens_out: int = 0


class LLMClient(Protocol):
    def complete(self, system: str, user: str) -> LLMResponse: ...


# ----------------------------------------------------------------------
# Scripted (dry-run) client
# ----------------------------------------------------------------------
@dataclass
class ScriptedClient:
    """Replays a queue of scripted responses keyed by role tag.

    The agent asks the LLM for three kinds of outputs during a run:
    ``plan``, ``step`` and ``critic``. The scripted client simply pops
    the next response matching the requested tag. This keeps the dry
    run deterministic and lets tests assert on exact behavior.
    """

    scripts: dict[str, list[str]] = field(default_factory=dict)
    tokens_in: int = 0
    tokens_out: int = 0

    def complete(self, system: str, user: str) -> LLMResponse:
        tag = _infer_tag(system)
        queue = self.scripts.get(tag, [])
        if not queue:
            raise RuntimeError(
                f"ScriptedClient exhausted for tag={tag!r}. "
                f"Available tags: {sorted(self.scripts)}"
            )
        text = queue.pop(0)
        self.tokens_in += len(user) // 4
        self.tokens_out += len(text) // 4
        return LLMResponse(text=text, tokens_in=len(user) // 4, tokens_out=len(text) // 4)


def _infer_tag(system: str) -> str:
    lower = system.lower()
    if "planner" in lower:
        return "plan"
    if "critic" in lower:
        return "critic"
    return "step"


# ----------------------------------------------------------------------
# Real providers (thin wrappers, optional imports)
# ----------------------------------------------------------------------
class AnthropicClient:
    def __init__(self, model: str = "claude-sonnet-4-5") -> None:
        try:
            import anthropic  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dep
            raise RuntimeError(
                "anthropic package not installed; `pip install anthropic`"
            ) from exc
        self._client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        self.model = model

    def complete(self, system: str, user: str) -> LLMResponse:  # pragma: no cover - network
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(block.text for block in msg.content if getattr(block, "type", "") == "text")
        return LLMResponse(
            text=text,
            tokens_in=getattr(msg.usage, "input_tokens", 0),
            tokens_out=getattr(msg.usage, "output_tokens", 0),
        )


class OpenAIClient:
    def __init__(self, model: str = "gpt-4o-mini") -> None:
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dep
            raise RuntimeError("openai package not installed; `pip install openai`") from exc
        self._client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.model = model

    def complete(self, system: str, user: str) -> LLMResponse:  # pragma: no cover - network
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        choice = resp.choices[0].message.content or ""
        usage = resp.usage
        return LLMResponse(
            text=choice,
            tokens_in=getattr(usage, "prompt_tokens", 0) if usage else 0,
            tokens_out=getattr(usage, "completion_tokens", 0) if usage else 0,
        )


# ----------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------
def build_client(
    provider: str,
    *,
    scripts: dict[str, list[str]] | None = None,
    model: str | None = None,
) -> LLMClient:
    provider = provider.lower()
    if provider in {"dry-run", "dry_run", "scripted"}:
        return ScriptedClient(scripts=scripts or {})
    if provider == "anthropic":
        return AnthropicClient(model=model or "claude-sonnet-4-5")
    if provider == "openai":
        return OpenAIClient(model=model or "gpt-4o-mini")
    raise ValueError(f"unknown provider {provider!r}")


# ----------------------------------------------------------------------
# Default demo script
# ----------------------------------------------------------------------
def default_demo_scripts() -> dict[str, list[str]]:
    """Scripted responses that drive the sample-repo demo end-to-end.

    The scripted agent reads ``strutils.py``, writes a ``reverse_string``
    implementation, re-runs the tests, and the critic approves.
    """

    plan = {
        "goal": "Add a reverse_string(s) function to strutils.py and make tests pass",
        "steps": [
            "Inspect the repo layout",
            "Read strutils.py to see what exists",
            "Implement reverse_string in strutils.py",
            "Run the test suite to confirm green",
        ],
    }

    step_list_dir = {
        "thought": "Start by seeing what files exist at the repo root.",
        "tool": "list_dir",
        "args": {"path": "."},
    }
    step_read = {
        "thought": "Read strutils.py to understand the current module.",
        "tool": "read_file",
        "args": {"path": "strutils.py"},
    }
    step_write = {
        "thought": "Append a reverse_string function that uses slicing.",
        "tool": "write_file",
        "args": {
            "path": "strutils.py",
            "content": (
                '"""Small string utilities used by tests."""\n\n\n'
                "def shout(s: str) -> str:\n"
                '    """Return ``s`` upper-cased with an exclamation mark."""\n'
                '    return f"{s.upper()}!"\n\n\n'
                "def reverse_string(s: str) -> str:\n"
                '    """Return ``s`` reversed.\n\n'
                "    Implemented with slicing so it works for any sequence\n"
                '    of characters, including unicode."""\n'
                "    return s[::-1]\n"
            ),
        },
    }
    step_tests = {
        "thought": "Run pytest to confirm the new function makes the failing test pass.",
        "tool": "run_tests",
        "args": {"target": "tests"},
    }
    step_diff = {
        "thought": "Capture the diff so the critic can review the change.",
        "tool": "git_diff",
        "args": {},
    }

    critic_continue = lambda note: {"decision": "continue", "summary": note}
    critic_approve = {
        "decision": "approve",
        "summary": "reverse_string implemented with slicing; test suite is green.",
    }

    return {
        "plan": [json.dumps(plan)],
        "step": [
            json.dumps(step_list_dir),
            json.dumps(step_read),
            json.dumps(step_write),
            json.dumps(step_tests),
            json.dumps(step_diff),
        ],
        "critic": [
            json.dumps(critic_continue("layout noted; read strutils.py next.")),
            json.dumps(critic_continue("current module inspected; implement reverse_string.")),
            json.dumps(critic_continue("file written; run the tests to verify.")),
            json.dumps(critic_continue("tests green; capture the diff for review.")),
            json.dumps(critic_approve),
        ],
    }
