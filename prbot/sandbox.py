"""Sandboxed shell execution.

Adds a command allowlist on top of the workdir+timeout runner. The
first token of any argv must be in ALLOWED_COMMANDS or the sandbox
refuses to run.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

ALLOWED_COMMANDS: frozenset[str] = frozenset(
    {
        "python",
        "python3",
        "pytest",
        "ls",
        "cat",
        "grep",
        "rg",
        "git",
        "gh",
        "echo",
        "pwd",
        "true",
        "false",
    }
)


class SandboxError(RuntimeError):
    """Raised when a command is rejected by the sandbox."""


@dataclass
class SandboxResult:
    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0


class Sandbox:
    def __init__(
        self,
        workdir: str | os.PathLike[str],
        *,
        allowed: Iterable[str] = ALLOWED_COMMANDS,
        timeout: float = 30.0,
    ) -> None:
        self.workdir = Path(workdir).resolve()
        if not self.workdir.is_dir():
            raise SandboxError(f"workdir does not exist: {self.workdir}")
        self.allowed = frozenset(allowed)
        self.timeout = timeout

    def run(self, argv: Sequence[str], *, timeout: float | None = None) -> SandboxResult:
        if not argv:
            raise SandboxError("empty command")
        head = os.path.basename(argv[0])
        normalized = head
        if head.startswith("python"):
            normalized = "python3" if head.startswith("python3") else "python"
        if normalized not in self.allowed and head not in self.allowed:
            raise SandboxError(
                f"command {head!r} not in allowlist {sorted(self.allowed)}"
            )
        try:
            proc = subprocess.run(
                list(argv),
                cwd=str(self.workdir),
                capture_output=True,
                text=True,
                timeout=timeout if timeout is not None else self.timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise SandboxError(f"command timed out after {exc.timeout}s") from exc
        return SandboxResult(
            command=tuple(argv),
            returncode=proc.returncode,
            stdout=proc.stdout or "",
            stderr=proc.stderr or "",
        )
