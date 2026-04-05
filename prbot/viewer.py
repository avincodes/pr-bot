"""Render a JSONL trace as a human-readable timeline.

Usage:
    python -m prbot.viewer traces/<run_id>.jsonl
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

COLORS = {
    "run_start": "\033[1;36m",
    "run_end": "\033[1;36m",
    "plan": "\033[1;35m",
    "tool_call": "\033[1;33m",
    "tool_result": "\033[0;32m",
    "critic_decision": "\033[1;34m",
    "repair": "\033[1;31m",
    "budget_exceeded": "\033[1;31m",
}
RESET = "\033[0m"


def _color(event: str) -> str:
    return COLORS.get(event, "")


def _fmt(value: Any, width: int = 80) -> str:
    text = json.dumps(value, default=str) if not isinstance(value, str) else value
    text = text.replace("\n", " ")
    if len(text) > width:
        text = text[: width - 1] + "..."
    return text


def render(path: Path) -> None:
    if not path.exists():
        print(f"trace not found: {path}", file=sys.stderr)
        raise SystemExit(2)
    lines = path.read_text().splitlines()
    start_ts = None
    print(f"=== trace {path.name} ({len(lines)} events) ===")
    for raw in lines:
        if not raw.strip():
            continue
        rec = json.loads(raw)
        ts = rec.get("ts", 0.0)
        if start_ts is None:
            start_ts = ts
        offset = ts - start_ts
        event = rec.get("event", "?")
        color = _color(event)
        head = f"{color}[{offset:6.2f}s] {event:<16}{RESET}"
        if event == "plan":
            plan = rec.get("plan", {})
            print(f"{head} goal: {plan.get('goal', '')}")
            for i, step in enumerate(plan.get("steps", []), start=1):
                print(f"             {i}. {step}")
        elif event == "tool_call":
            print(f"{head} {rec.get('tool')}({_fmt(rec.get('args', {}), 60)})")
        elif event == "tool_result":
            data = rec.get("data", {})
            summary = _fmt({k: data[k] for k in list(data)[:3]}, 70)
            ok = "ok" if rec.get("ok") else "FAIL"
            print(f"{head} {rec.get('tool')} -> {ok} {summary}")
        elif event == "critic_decision":
            dec = rec.get("decision", {})
            print(f"{head} {dec.get('decision', '?')}: {dec.get('summary', '')}")
        elif event == "repair":
            print(f"{head} {rec.get('hint', '')}")
        elif event == "run_start":
            print(f"{head} issue: {rec.get('issue', '')}")
        elif event == "run_end":
            print(
                f"{head} approved={rec.get('approved')} "
                f"iters={rec.get('iterations')} "
                f"tokens={rec.get('tokens_in', 0)}+{rec.get('tokens_out', 0)}"
            )
        else:
            print(f"{head} {_fmt({k: v for k, v in rec.items() if k not in {'ts','run_id','event'}})}")


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        print("usage: python -m prbot.viewer <trace.jsonl>", file=sys.stderr)
        return 2
    render(Path(argv[0]))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
