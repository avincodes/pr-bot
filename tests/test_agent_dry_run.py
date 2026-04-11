"""End-to-end test: run the agent against the shipped sample-repo in dry-run."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from prbot.agent import Agent, RunConfig  # noqa: E402
from prbot.llm import build_client, default_demo_scripts  # noqa: E402
from prbot.sandbox import Sandbox  # noqa: E402
from prbot.tools import Toolbox, ensure_git_repo  # noqa: E402
from prbot.trace import TraceLogger  # noqa: E402


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    src = ROOT / "examples" / "sample-repo"
    dst = tmp_path / "sample-repo"
    shutil.copytree(src, dst)
    ensure_git_repo(dst)
    return dst


def test_dry_run_approves_and_writes_file(repo: Path, tmp_path: Path) -> None:
    sandbox = Sandbox(repo, timeout=60.0)
    tools = Toolbox(sandbox)
    llm = build_client("dry-run", scripts=default_demo_scripts())
    with TraceLogger(root=tmp_path / "traces") as tracer:
        agent = Agent(llm, tools, tracer, config=RunConfig(max_iters=10))
        result = agent.run("Add a function to reverse a string")

    assert result.approved, f"agent did not approve: {result.history}"
    assert (repo / "strutils.py").read_text().count("reverse_string") == 1
    # Trace file exists and has events
    assert result.trace_path.exists()
    assert result.trace_path.read_text().count("\n") > 5
