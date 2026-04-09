import os
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from prbot.sandbox import Sandbox, SandboxError  # noqa: E402


@pytest.fixture
def workdir(tmp_path: Path) -> Path:
    (tmp_path / "hello.txt").write_text("hi\n")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "nested.txt").write_text("nested\n")
    return tmp_path


def test_allowed_command_runs(workdir: Path) -> None:
    sb = Sandbox(workdir)
    result = sb.run(["echo", "hello"])
    assert result.ok
    assert result.stdout.strip() == "hello"


def test_disallowed_command_rejected(workdir: Path) -> None:
    sb = Sandbox(workdir)
    with pytest.raises(SandboxError):
        sb.run(["rm", "-rf", "/"])


def test_metacharacters_rejected(workdir: Path) -> None:
    sb = Sandbox(workdir)
    with pytest.raises(SandboxError):
        sb.run(["echo", "hi && rm -rf /"])


def test_resolve_inside_blocks_escape(workdir: Path) -> None:
    sb = Sandbox(workdir)
    with pytest.raises(SandboxError):
        sb.resolve_inside("../etc/passwd")


def test_resolve_inside_ok(workdir: Path) -> None:
    sb = Sandbox(workdir)
    p = sb.resolve_inside("sub/nested.txt")
    assert p.read_text() == "nested\n"


def test_timeout_raises(workdir: Path) -> None:
    sb = Sandbox(workdir, timeout=0.2)
    py = shutil.which("python") or shutil.which("python3")
    assert py is not None
    with pytest.raises(SandboxError):
        sb.run([py, "-c", "import time; time.sleep(2)"])
