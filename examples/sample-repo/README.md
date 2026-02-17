# sample-repo

A deliberately tiny Python project used as a playground for `prbot`.

`strutils.py` ships with `shout()` only. The test suite references a
`reverse_string()` function that does not exist yet, so `pytest` fails
on a clean checkout. The agent's job is to implement that function.

```
pytest tests -q   # fails on a clean checkout
```
