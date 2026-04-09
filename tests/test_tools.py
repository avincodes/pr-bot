import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from prbot.sandbox import Sandbox  # noqa: E402
from prbot.tools import Toolbox  # noqa: E402


@pytest.fixture
def tools(tmp_path: Path) -> Toolbox:
    (tmp_path / "a.py").write_text("def foo():\n    return 1\n")
    (tmp_path / "b.py").write_text("def bar():\n    return 2\n")
    (tmp_path / "notes.txt").write_text("hello\n")
    return Toolbox(Sandbox(tmp_path))


def test_read_file_ok(tools: Toolbox) -> None:
    r = tools.read_file("a.py")
    assert r.ok
    assert "foo" in r.data["content"]


def test_read_file_missing(tools: Toolbox) -> None:
    r = tools.read_file("nope.py")
    assert not r.ok


def test_write_file_roundtrip(tools: Toolbox) -> None:
    w = tools.write_file("c.py", "x = 1\n")
    assert w.ok
    r = tools.read_file("c.py")
    assert r.data["content"] == "x = 1\n"


def test_list_dir(tools: Toolbox) -> None:
    r = tools.list_dir(".")
    assert r.ok
    assert "a.py" in r.data["entries"]


def test_grep_finds_match(tools: Toolbox) -> None:
    r = tools.grep(r"def\s+foo", glob="*.py")
    assert r.ok
    assert any("a.py" in hit["path"] for hit in r.data["hits"])


def test_dispatch_unknown(tools: Toolbox) -> None:
    r = tools.call("definitely_not_a_tool", {})
    assert not r.ok


def test_dispatch_bad_args(tools: Toolbox) -> None:
    r = tools.call("read_file", {"wrong": "arg"})
    assert not r.ok
