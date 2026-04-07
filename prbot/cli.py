"""Command-line entry point for prbot."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .agent import Agent, RunConfig
from .llm import build_client, default_demo_scripts
from .sandbox import Sandbox
from .tools import Toolbox, ensure_git_repo
from .trace import TraceLogger


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="prbot", description="Autonomous coding agent.")
    p.add_argument("--repo", required=True, help="path to the target repository")
    p.add_argument("--issue", required=True, help="natural-language issue to solve")
    p.add_argument(
        "--provider",
        default="dry-run",
        choices=["dry-run", "anthropic", "openai"],
        help="LLM provider (default: dry-run scripted)",
    )
    p.add_argument("--model", default=None, help="override model name")
    p.add_argument("--max-iters", type=int, default=12)
    p.add_argument("--token-budget", type=int, default=50_000)
    p.add_argument("--dry-run", action="store_true", help="force dry-run scripted mode")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo = Path(args.repo).resolve()
    if not repo.is_dir():
        print(f"error: repo not found: {repo}", file=sys.stderr)
        return 2

    ensure_git_repo(repo)
    sandbox = Sandbox(repo, timeout=60.0)
    tools = Toolbox(sandbox)

    provider = "dry-run" if args.dry_run else args.provider
    scripts = default_demo_scripts() if provider == "dry-run" else None
    llm = build_client(provider, scripts=scripts, model=args.model)

    config = RunConfig(max_iters=args.max_iters, token_budget=args.token_budget)
    with TraceLogger(root=Path("traces")) as tracer:
        agent = Agent(llm, tools, tracer, config=config)
        print(f"[prbot] run_id={tracer.run_id} repo={repo} provider={provider}")
        result = agent.run(args.issue)

    print("-" * 60)
    print(f"approved       : {result.approved}")
    print(f"iterations     : {result.iterations}")
    print(f"tokens in/out  : {result.tokens_in}/{result.tokens_out}")
    print(f"trace          : {result.trace_path}")
    print(f"plan goal      : {result.plan.get('goal', '')}")
    print("-" * 60)
    if result.approved:
        print("[prbot] critic approved the change. Review the diff with:")
        print(f"    git -C {repo} diff")
    else:
        print("[prbot] run ended without approval. Inspect the trace for details.")
    return 0 if result.approved else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
