"""Tool interface exposed to the agent.

First cut: read/write/list/grep on top of the sandbox. Git and gh
helpers land in a follow-up.
"""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .sandbox import Sandbox, SandboxError


@dataclass
class ToolResult:
    ok: bool
    data: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return {"ok": self.ok, **self.data}


class Toolbox:
    def __init__(self, sandbox: Sandbox) -> None:
        self.sandbox = sandbox

    def read_file(self, path: str) -> ToolResult:
        try:
            target = self.sandbox.resolve_inside(path)
            text = target.read_text()
        except (SandboxError, FileNotFoundError, OSError) as exc:
            return ToolResult(False, {"error": str(exc)})
        return ToolResult(True, {"path": path, "content": text})

    def write_file(self, path: str, content: str) -> ToolResult:
        try:
            target = self.sandbox.resolve_inside(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
        except (SandboxError, OSError) as exc:
            return ToolResult(False, {"error": str(exc)})
        return ToolResult(True, {"path": path, "bytes": len(content)})

    def list_dir(self, path: str = ".") -> ToolResult:
        try:
            target = self.sandbox.resolve_inside(path)
            entries = sorted(p.name for p in target.iterdir())
        except (SandboxError, OSError) as exc:
            return ToolResult(False, {"error": str(exc)})
        return ToolResult(True, {"path": path, "entries": entries})

    def grep(self, pattern: str, glob: str = "*.py") -> ToolResult:
        try:
            regex = re.compile(pattern)
        except re.error as exc:
            return ToolResult(False, {"error": f"bad regex: {exc}"})
        hits: list[dict[str, Any]] = []
        root = self.sandbox.workdir
        for file in root.rglob("*"):
            if not file.is_file():
                continue
            rel = file.relative_to(root).as_posix()
            if not fnmatch.fnmatch(rel, glob) and not fnmatch.fnmatch(file.name, glob):
                continue
            try:
                for i, line in enumerate(file.read_text().splitlines(), start=1):
                    if regex.search(line):
                        hits.append({"path": rel, "line": i, "text": line})
            except OSError:
                continue
        return ToolResult(True, {"pattern": pattern, "glob": glob, "hits": hits})

    def call(self, name: str, args: dict[str, Any]) -> ToolResult:
        method = getattr(self, name, None)
        if not callable(method) or name.startswith("_"):
            return ToolResult(False, {"error": f"unknown tool {name!r}"})
        try:
            return method(**args)
        except TypeError as exc:
            return ToolResult(False, {"error": f"bad args for {name}: {exc}"})
