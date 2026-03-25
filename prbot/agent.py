"""The agent loop: planner -> executor.

First cut: produce a plan, then step through tool calls until a
fixed iteration cap. No critic, no repair, no token budget yet.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .llm import LLMClient, LLMResponse
from .tools import Toolbox
from .trace import TraceLogger

PLANNER_SYSTEM = (
    "You are the planner module of an autonomous coding agent. "
    "Given an issue and repo layout, output a JSON object with keys "
    "`goal` (string) and `steps` (list of short strings). "
    "Do not include prose outside the JSON."
)

EXECUTOR_SYSTEM = (
    "You are the executor module of an autonomous coding agent. "
    "Pick exactly ONE tool to call next. Output a JSON object with keys "
    "`thought`, `tool`, and `args`. Valid tools: read_file, write_file, "
    "list_dir, grep, run_tests, git_diff, git_commit, gh_open_pr."
)


@dataclass
class RunResult:
    iterations: int
    plan: dict[str, Any]
    history: list[dict[str, Any]]
    trace_path: Path


class Agent:
    MAX_ITERS = 8

    def __init__(self, llm: LLMClient, tools: Toolbox, tracer: TraceLogger) -> None:
        self.llm = llm
        self.tools = tools
        self.tracer = tracer

    def _ask(self, system: str, user: str, *, event: str) -> LLMResponse:
        resp = self.llm.complete(system, user)
        self.tracer.emit(event, text=resp.text)
        return resp

    def _parse_json(self, text: str) -> dict[str, Any]:
        text = text.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text)

    def plan(self, issue: str, layout: list[str]) -> dict[str, Any]:
        user = json.dumps({"issue": issue, "layout": layout})
        resp = self._ask(PLANNER_SYSTEM, user, event="plan_request")
        plan = self._parse_json(resp.text)
        self.tracer.emit("plan", plan=plan)
        return plan

    def next_step(self, plan: dict[str, Any], history: list[dict[str, Any]]) -> dict[str, Any]:
        user = json.dumps({"plan": plan, "history": history[-6:]})
        resp = self._ask(EXECUTOR_SYSTEM, user, event="step_request")
        step = self._parse_json(resp.text)
        self.tracer.emit("tool_call", tool=step.get("tool"), args=step.get("args", {}))
        return step

    def run(self, issue: str) -> RunResult:
        layout = sorted(
            p.relative_to(self.tools.sandbox.workdir).as_posix()
            for p in self.tools.sandbox.workdir.rglob("*")
            if p.is_file() and ".git" not in p.parts
        )
        self.tracer.emit("run_start", issue=issue, layout=layout[:50])
        plan = self.plan(issue, layout[:50])
        history: list[dict[str, Any]] = []

        for i in range(self.MAX_ITERS):
            step = self.next_step(plan, history)
            tool = step.get("tool", "")
            args = step.get("args", {}) or {}
            result = self.tools.call(tool, args)
            self.tracer.emit("tool_result", tool=tool, ok=result.ok, data=result.data)
            history.append({"tool": tool, "args": args, "result": result.to_json()})

        self.tracer.emit("run_end", iterations=len(history))
        return RunResult(
            iterations=len(history),
            plan=plan,
            history=history,
            trace_path=self.tracer.path,
        )
