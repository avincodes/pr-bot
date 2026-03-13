"""LLM client abstraction.

First cut: a Protocol and a scripted (dry-run) client that plays back
canned responses keyed by role. Real providers land next.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class LLMResponse:
    text: str
    tokens_in: int = 0
    tokens_out: int = 0


class LLMClient(Protocol):
    def complete(self, system: str, user: str) -> LLMResponse: ...


@dataclass
class ScriptedClient:
    """Replays a queue of scripted responses keyed by role tag."""

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
