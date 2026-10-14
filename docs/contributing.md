# Contributing

Contributions are welcome — issues, suggestions and pull requests alike.

```bash
git clone https://github.com/patrk/pyPTE && cd pyPTE
uv sync --all-extras
uv run pytest
```

Before opening a PR: `uv run ruff check .`, `uv run ruff format .`,
`uv run mypy pyPTE tests`.
