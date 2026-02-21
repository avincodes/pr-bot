"""Sandboxed shell execution (workdir + timeout).

First cut: just pins execution to a working directory and enforces a
wall-clock timeout. Allowlisting and path-escape protection come next.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


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
    def __init__(self, workdir: str | os.PathLike[str], *, timeout: float = 30.0) -> None:
        self.workdir = Path(workdir).resolve()
        if not self.workdir.is_dir():
            raise SandboxError(f"workdir does not exist: {self.workdir}")
        self.timeout = timeout

    def run(self, argv: Sequence[str], *, timeout: float | None = None) -> SandboxResult:
        if not argv:
            raise SandboxError("empty command")
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
