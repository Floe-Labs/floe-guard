# Contributing to floe-guard

`floe-guard` is a local budget guardrail for AI agents: it hard-stops an agent
before its next LLM call when it would cross a spend ceiling. It runs in-process
with no account. Hosted Floe is the upgrade path — enforcement moves server-side
so the ceiling becomes un-bypassable and cross-vendor.

Contributions are welcome. The repo holds two packages: the Python package at
the root (`src/floe_guard/`, published to PyPI) and the TypeScript package for
the Vercel AI SDK in [`js/`](js/) (published to npm).

## Development setup (Python)

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
pip install -e ".[crewai]"      # CrewAI adapter
pip install -e ".[litellm]"     # LiteLLM adapter
pip install -e ".[langchain]"   # LangChain adapter
pip install -e ".[openai]"      # OpenAI adapter
pip install -e ".[anthropic]"   # Anthropic adapter
```

The core must keep working with **no** extras installed — CI runs the suite
bare, and adapter tests skip cleanly when their framework is absent.

## Development setup (TypeScript, `js/`)

```bash
cd js
npm ci
npm run build        # tsup
npm test             # vitest
npm run typecheck    # tsc --noEmit
```

The middleware supports both `ai@4` and `ai@5` from one build; it deliberately
imports nothing from `ai` (see `js/src/middleware.ts`) — keep it that way.

## Contribution flow

1. Fork the repo and create a branch off `main` (e.g. `feat/your-change`).
2. Make your change with tests, and keep the checks above green.
3. Open a **draft pull request** against `main` and describe what changed and why.

## Code style

- Python 3.10+ with type hints.
- Formatting and linting are handled by `ruff` (config in `pyproject.toml`) — run
  `ruff format .` and `ruff check .` before pushing.
- Keep the core (`src/floe_guard/`, `js/src/guard.ts`) dependency-free; framework
  integrations belong in `src/floe_guard/integrations/` behind an optional extra.
- The cost map is vendored twice (`src/floe_guard/cost_map.json` and
  `js/src/cost_map.json`) and must stay byte-identical — CI fails on drift.
  Refresh both together (`scripts/update-cost-map.mjs`).
- Prefer small, focused changes with tests that describe behavior.

## Areas we'd love help with

- **Cost-map coverage** — keeping the bundled pricing map current and broad
  (new models, new providers).
- **New adapters** — e.g. Google Gemini SDK, AutoGen, Pydantic AI, Semantic
  Kernel. Follow the reserve-before / settle-after contract the OpenAI and
  Anthropic adapters use.
- **Docs and examples** — small, runnable, no-API-key demos like
  `examples/runaway_loop.py`.

Open an issue first if you want to discuss a larger change.

## Releases

Versions bump per package (`pyproject.toml` for PyPI, `js/package.json` +
`js/package-lock.json` for npm). Publishing is automated: merging to `main`
runs tests and publishes any version not yet on the registry. Add a
[CHANGELOG.md](CHANGELOG.md) entry with the version bump, and tag the release
(`git tag py-vX.Y.Z` / `js-vX.Y.Z`) so it shows up under GitHub Releases.
