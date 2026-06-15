# Contributing to floe-guard

`floe-guard` is a local budget guardrail for AI agents: it hard-stops an agent
before its next LLM call when it would cross a spend ceiling. It runs in-process
with no account. Hosted Floe is the upgrade path — enforcement moves server-side
so the ceiling becomes un-bypassable and cross-vendor.

Contributions are welcome.

## Development setup

```bash
git clone https://github.com/Floe-Labs/floe-guard.git
cd floe-guard
pip install -e ".[dev]"
```

Run the checks before opening a PR:

```bash
pytest               # tests
ruff check .         # lint
ruff format .        # format
```

The optional adapters need their extras to run/import:

```bash
pip install -e ".[crewai]"    # CrewAI adapter
pip install -e ".[litellm]"   # LiteLLM adapter
```

## Contribution flow

1. Fork the repo and create a branch off `main` (e.g. `feat/your-change`).
2. Make your change with tests, and keep `pytest` + `ruff` green.
3. Open a **draft pull request** against `main` and describe what changed and why.

## Code style

- Python 3.10+ with type hints.
- Formatting and linting are handled by `ruff` (config in `pyproject.toml`) — run
  `ruff format .` and `ruff check .` before pushing.
- Keep the core (`src/floe_guard/`) dependency-free; framework integrations belong
  in `src/floe_guard/integrations/` behind an optional extra.
- Prefer small, focused changes with tests that describe behavior.

## Areas we'd love help with

- **LangChain adapter** (callback-based enforcement).
- **Vercel AI SDK adapter** (TypeScript middleware / port).
- **Cost-map coverage** — keeping the bundled pricing map current and broad.

Open an issue first if you want to discuss a larger change.
