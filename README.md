# prbot — an autonomous coding agent that ships PRs

`prbot` is a compact autonomous coding agent (mini-Devin) that takes a
natural-language issue, plans a fix, edits files inside a sandboxed
working directory, runs the test suite, self-critiques the diff, and
opens a pull request when it is happy. The whole thing is under 2k
lines of Python and has no third-party runtime dependencies in
dry-run mode.

It exists to make the agent loop itself legible: every decision,
tool call, tool result, and critic verdict is written to a
structured JSONL trace that you can replay or diff against a
previous run.

## Why

Most agent demos are a single prompt with a tool-use tantrum wrapped
around it. I wanted something closer to how I would actually ship a
fix: decompose the problem, take one concrete step at a time, check
your work against the test suite, and repair when the critic
complains. `prbot` is a small but honest implementation of that
loop with enough seams (sandbox, LLM abstraction, scripted dry-run)
to run it offline end-to-end.

## Architecture

```
                    +----------------+
   issue ---------> |    Planner     |---+
                    +----------------+   |
                                         v
                    +----------------+   steps, history
              +---->|    Executor    |-----------+
              |     +----------------+           |
              |             |                    |
              |             v                    |
              |     +----------------+            |
              |     |    Toolbox     |            |
              |     | read/write/    |            |
              |     | grep/tests/git |            |
              |     +----------------+            |
              |             |                    |
              |             v                    |
              |     +----------------+            |
              |     |     Critic     |<-----------+
              |     +----------------+
              |             |
              |  repair/continue
              +-------------+
                            |
                         approve
                            v
                      gh_open_pr
```

Every arrow is also a JSON event in `traces/<run_id>.jsonl`.

## Agent loop

1. **Planner.** Given the issue and the repo layout, emit a
   `goal` and a list of concrete `steps`.
2. **Executor.** On each iteration, pick exactly one tool call
   (`read_file`, `write_file`, `list_dir`, `grep`, `run_tests`,
   `git_diff`, `git_commit`, `gh_open_pr`).
3. **Tool dispatch.** The tool runs through the sandbox and the
   result is appended to the history and the trace.
4. **Critic.** Reads the last tool result in context of the plan.
   It either approves (loop terminates), asks for another step, or
   issues a `repair` hint that gets pushed back into the history
   for the executor to react to.
5. **Budget.** The loop has a hard iteration cap and a token
   budget; exceeding either ends the run without approval.

Planner, executor and critic are three calls into the same
`LLMClient`, distinguished only by the system prompt. The scripted
dry-run client uses that to route each role to its own response
queue, so the demo is 100% deterministic and offline.

## Sandbox and safety

`prbot.sandbox.Sandbox` is the only way tools touch the filesystem
or spawn a process. It:

- pins execution to a `workdir` and refuses to resolve paths that
  escape it (`../../etc/passwd` is a hard error),
- only runs commands whose head is in a small allowlist
  (`python`, `pytest`, `git`, `gh`, `ls`, `grep`, ...),
- rejects shell metacharacters (`&&`, `|`, `;`, backticks, `$(`,
  redirections) so callers have to pass argv lists,
- enforces a wall-clock timeout on every subprocess.

It is *not* a security boundary against hostile code — it is a
guardrail against the common failure modes of an LLM-driven
executor: runaway loops, `rm -rf`, pipes into `curl | sh`, writing
outside the repo. For real-world deployment you would want to run
the whole thing inside a disposable container or VM on top of this.

## Running it

### Dry-run demo (no API keys)

```
python -m prbot.cli \
    --repo ./examples/sample-repo \
    --issue "Add a function to reverse a string" \
    --dry-run
```

The sample repo ships with a failing test for `reverse_string`.
The scripted LLM responses walk the agent through inspecting the
file, implementing the function, running the tests, and approving
the diff. After the run:

```
python -m prbot.viewer traces/<run_id>.jsonl
git -C examples/sample-repo diff
```

### Real providers

Install the optional dep and export a key:

```
pip install -e .[anthropic]   # or .[openai]
export ANTHROPIC_API_KEY=sk-...
python -m prbot.cli --repo path/to/repo --issue "..." --provider anthropic
```

### Tests

```
pip install -e .[dev]
pytest -q
```

The suite covers the sandbox allowlist, path-escape protection,
tool layer dispatch, and a full end-to-end dry-run against the
sample repo that asserts the agent actually produces the expected
diff.

## Project layout

```
prbot/
  agent.py     planner/executor/critic loop
  cli.py       command-line entry point
  llm.py       LLMClient + scripted dry-run + real providers
  sandbox.py   allowlisted, workdir-pinned, timed subprocess runner
  tools.py     read/write/grep/tests/git/gh tools on top of the sandbox
  trace.py     structured JSONL event logger
  viewer.py    human-readable timeline renderer
examples/
  sample-repo/ tiny python project with a failing test (the demo fixture)
tests/
  test_sandbox.py, test_tools.py, test_agent_dry_run.py
traces/        run logs land here
```

## Roadmap

- Real GitHub integration: open an actual PR with `gh pr create`
  and drop the `simulated` flag from `gh_open_pr`.
- Multi-file refactors: let the executor propose a change-set
  instead of a single write, and have the critic diff-review it.
- Cached tool-result memoization keyed on content hashes, so
  repair loops are cheap.
- Pluggable critics (lint, typecheck, perf benchmarks).
- Replay mode: take a previous trace and re-run it against a new
  model to compare plans.

## License

MIT.
