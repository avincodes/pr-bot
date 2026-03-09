"""Tool interface exposed to the agent.

Each tool is a small, strongly-typed function that the executor can
dispatch by name. Tools return JSON-serializable dicts so every call
can be recorded in the trace log.
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
    """Bundle of tools bound to a single :class:`Sandbox`."""

    def __init__(self, sandbox: Sandbox) -> None:
        self.sandbox = sandbox

    # ------------------------------------------------------------------
    # File IO
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Tests and git
    # ------------------------------------------------------------------
    def run_tests(self, target: str = "tests") -> ToolResult:
        import sys as _sys

        result = self.sandbox.run([_sys.executable, "-m", "pytest", target, "-q"], timeout=120)
        return ToolResult(
            result.ok,
            {
                "returncode": result.returncode,
                "stdout": result.stdout[-4000:],
                "stderr": result.stderr[-2000:],
            },
        )

    def git_diff(self) -> ToolResult:
        result = self.sandbox.run(["git", "diff", "--no-color"])
        return ToolResult(
            result.ok,
            {"diff": result.stdout, "stderr": result.stderr},
        )

    def git_commit(self, message: str) -> ToolResult:
        add = self.sandbox.run(["git", "add", "-A"])
        if not add.ok:
            return ToolResult(False, {"stage": "add", "stderr": add.stderr})
        commit = self.sandbox.run(["git", "commit", "-m", message])
        return ToolResult(
            commit.ok,
            {"stdout": commit.stdout, "stderr": commit.stderr},
        )

    def gh_open_pr(self, title: str, body: str) -> ToolResult:
        # Opening a real PR requires auth + network. In dry-run / tests we
        # just record the intent so the full loop is observable.
        return ToolResult(
            True,
            {
                "simulated": True,
                "title": title,
                "body": body,
            },
        )

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------
    def call(self, name: str, args: dict[str, Any]) -> ToolResult:
        method = getattr(self, name, None)
        if not callable(method) or name.startswith("_"):
            return ToolResult(False, {"error": f"unknown tool {name!r}"})
        try:
            return method(**args)
        except TypeError as exc:
            return ToolResult(False, {"error": f"bad args for {name}: {exc}"})


def ensure_git_repo(path: str | Path) -> None:
    """Initialize a disposable git repo if one does not exist."""
    root = Path(path)
    if (root / ".git").is_dir():
        return
    sb = Sandbox(root)
    sb.run(["git", "init", "-q"])
    sb.run(["git", "add", "-A"])
    # Use -c flags would require shell; keep identity via env vars instead.
    import os

    env_cmd = ["git", "commit", "-q", "-m", "seed"]
    # Inject a throwaway identity via env so commits work even on a fresh box.
    os.environ.setdefault("GIT_AUTHOR_NAME", "prbot")
    os.environ.setdefault("GIT_AUTHOR_EMAIL", "prbot@example.com")
    os.environ.setdefault("GIT_COMMITTER_NAME", "prbot")
    os.environ.setdefault("GIT_COMMITTER_EMAIL", "prbot@example.com")
    sb.run(env_cmd)
