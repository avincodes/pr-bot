"""Sandboxed shell execution.

The sandbox:
  - Whitelists allowed commands (first token must be in ALLOWED).
  - Pins execution to a working directory and refuses to escape it.
  - Enforces a wall-clock timeout and captures stdout/stderr.
  - Rejects shell metacharacters so callers must pass args explicitly.

It is NOT a security boundary for hostile code, but it catches the
common classes of agent mistakes (runaway loops, rm -rf, curl | sh).
"""

from __future__ import annotations

import os
import shlex
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

FORBIDDEN_TOKENS: tuple[str, ...] = ("&&", "||", ";", "|", ">", "<", "`", "$(")


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
    """Restricted command runner rooted at ``workdir``."""

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

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------
    def resolve_inside(self, relpath: str | os.PathLike[str]) -> Path:
        """Resolve ``relpath`` under the workdir, raising on escape."""
        candidate = (self.workdir / relpath).resolve()
        try:
            candidate.relative_to(self.workdir)
        except ValueError as exc:
            raise SandboxError(
                f"path escapes workdir: {candidate} not under {self.workdir}"
            ) from exc
        return candidate

    # ------------------------------------------------------------------
    # Command execution
    # ------------------------------------------------------------------
    def run(self, argv: Sequence[str], *, timeout: float | None = None) -> SandboxResult:
        if not argv:
            raise SandboxError("empty command")
        for token in argv:
            if not isinstance(token, str):
                raise SandboxError(f"non-string arg: {token!r}")
            for bad in FORBIDDEN_TOKENS:
                if bad in token:
                    raise SandboxError(f"forbidden token {bad!r} in arg {token!r}")
        head = os.path.basename(argv[0])
        # Accept versioned python binaries like python3.11, python3.13 by
        # normalizing the head down to its non-version prefix.
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
            raise SandboxError(
                f"command timed out after {exc.timeout}s: {shlex.join(argv)}"
            ) from exc
        return SandboxResult(
            command=tuple(argv),
            returncode=proc.returncode,
            stdout=proc.stdout or "",
            stderr=proc.stderr or "",
        )
