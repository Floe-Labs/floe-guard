# Codebase conventions

This guide records the engineering conventions already present in `floe-guard`.
Use it with [`CONTRIBUTING.md`](CONTRIBUTING.md), which remains the source of
truth for setup, contribution flow, and release mechanics.

The goal is not to make every contribution look identical. It is to preserve the
contracts that make this package trustworthy: stop paid work before it crosses a
limit, never silently under-meter, remain safe under concurrency, keep the core
dependency-free, and be explicit about what cannot be guaranteed.

## Start with the contract

Before changing code, write down:

1. Where enforcement happens **before** the paid operation.
2. Where actual usage is recorded after success.
3. How an in-flight reservation is released on failure, cancellation, missing
   usage, or an unsupported response.
4. Whether the behavior exists in both Python and TypeScript and, if so, which
   names and wire shapes must remain equivalent.
5. Which public docs, examples, exports, versions, and changelog entries the
   change affects.

For anything safety-sensitive, prefer the conservative outcome. A false low
price or a swallowed error weakens the product's core promise; an explicit
refusal is visible and recoverable.

## Repository map

| Path | Responsibility | Constraints |
|---|---|---|
| `src/floe_guard/` | Python package and framework-agnostic core | Python 3.10+; no required runtime dependencies |
| `src/floe_guard/integrations/` | Optional framework and provider adapters | Imports must not make the bare core require the framework extra |
| `tests/` | Python unit, regression, adapter, concurrency, and example tests | Must pass both with CI's adapter extras and with no extras installed |
| `examples/` | Small runnable demonstrations | Prefer no API key, no account, and no network |
| `js/src/` | TypeScript core and Vercel AI SDK middleware | Node 18+, ES2022, strict TypeScript |
| `js/test/` | Vitest behavioral and regression tests | Keep shared contracts aligned with Python |
| `scripts/update-cost-map.mjs` | Curates the vendored provider cost map | Update both package copies together |
| `.github/workflows/` | CI, version guards, publishing, and map refresh | Treat these workflows as enforced contribution constraints |

## Invariants that take priority over local style

### Enforce in the call path

A listener that observes a completed call cannot prevent that call. New
integrations should wrap the actual operation and follow the established
reserve/settle lifecycle:

```text
validate unsupported modes
    -> estimate when the request can be sized
    -> reserve before external work
    -> perform the operation
    -> settle from actual usage
```

If the operation fails before `settle()` takes ownership, release its
reservation and re-raise. Once `settle()` is called, do not release the same
reservation again: `settle()` owns its disposal even when it raises.

The sequential `check()` / `record()` API remains useful, but it is not a
replacement for `reserve()` / `settle()` when calls can overlap.

### Fail closed when spend cannot be measured

Unknown models, missing model IDs, absent or malformed usage, non-finite values,
and provider-specific prices that cannot be identified must not silently become
zero-cost work. Route completed spend through the guard's configured policy so
the default raises an explicit error. Only honor fail-open behavior when the
caller has deliberately configured it.

Validate inputs that could poison comparisons or totals. In particular, reject
`NaN`, infinities, negative costs, invalid reservation handles, and booleans
where a real integer is required.

### Preserve concurrency safety

State shared by parallel callers belongs under the existing lock or synchronous
reservation boundary. Never split a check and its state mutation across an
`await`, external call, or separately acquired lock.

Every reservation must have exactly one terminal path:

- settled after valid usage;
- released after an exception;
- released when a response or stream ends without usage; or
- retained only while the corresponding operation is genuinely in flight.

Tests should prove both the ceiling and cleanup. An assertion such as
`remaining >= 0` is not enough when the property is clamped; assert the exact
restored balance or inspect the in-flight value in a regression test.

### Keep pricing conservative

The cost map is a safety input, not ordinary reference data. Never infer a cheap
rate across billing backends when the model ID cannot identify the backend.
When aliases or duplicate entries disagree, choose behavior that cannot
under-meter. Document deliberate exclusions.

The following files must remain byte-identical:

```text
src/floe_guard/cost_map.json
js/src/cost_map.json
```

Refresh them through `scripts/update-cost-map.mjs`; do not hand-edit one copy.

### Keep the core dependency-free

The Python project's required dependency list is intentionally empty.
Framework integrations belong in `src/floe_guard/integrations/` and their
packages belong in optional extras in `pyproject.toml`.

The TypeScript middleware deliberately does not import `ai`, including for
types. It uses the structural surface shared by AI SDK v4 and v5 so one build
supports both peer versions. Preserve that compatibility unless the project
explicitly changes its support policy.

## Python conventions

### Layout and imports

- Begin modules with a docstring that explains purpose, contract, and important
  limitations. Integration modules also identify the optional extra.
- Use `from __future__ import annotations`.
- Order imports as standard library, third party, then local; Ruff enforces the
  ordering.
- Use relative imports within `floe_guard`.
- Keep public exports explicit in `src/floe_guard/__init__.py` and module-level
  `__all__` lists where the module defines a public surface.
- Prefix implementation helpers and internal state with `_`.

### Typing and data shapes

- Type public functions, methods, constructor parameters, return values, and
  meaningful internal helpers.
- Prefer built-in generics (`dict[str, object]`, `tuple[int, int]`) and `X | None`.
- Use frozen dataclasses for immutable public value objects.
- Use `Literal` when a field has a small closed vocabulary.
- Preserve stable serialization schemas. Optional fields are omitted when the
  established wire contract says they are absent; do not casually replace
  omission with `null`.
- When an optional SDK returns both objects and dictionaries, normalize both
  shapes in a small private helper.

### Naming

- Modules, functions, methods, parameters, and local variables use
  `snake_case`.
- Classes, exceptions, dataclasses, and type-like objects use `PascalCase`.
- Constants use `UPPER_SNAKE_CASE`; private constants add a leading underscore.
- Test names describe behavior and outcome, for example
  `test_usageless_response_releases_the_reservation`.
- Prefer domain terms already used by the public contract: `limit`, `spent`,
  `remaining`, `reserve`, `settle`, `release`, `advisory`, `prompt_tokens`, and
  `completion_tokens`.

### Functions and control flow

- Keep provider-shape extraction, model selection, validation, and lifecycle
  wiring in separate focused helpers.
- Return early for unsupported or no-usage cases.
- Catch the narrowest hierarchy that preserves lifecycle correctness. Adapter
  wrappers catch `BaseException` when cleanup must also run for cancellation or
  interruption, then immediately re-raise. Retry helpers deliberately catch
  `Exception` so process-level exceptions are not retried.
- Invoke callbacks and raise user-visible errors outside locks.
- Avoid clever abstractions when a short explicit branch makes ownership of a
  reservation or fallback easier to audit.

### Documentation and comments

Public classes and functions use docstrings that explain behavior, parameters,
returns, raised errors, and honest limitations where relevant. Use Sphinx roles
such as `:class:`, `:meth:`, and `:func:` when cross-referencing Python APIs.

Comments explain **why**, especially for:

- provider quirks;
- counterintuitive token accounting;
- concurrency ownership;
- fail-closed decisions;
- compatibility workarounds; and
- regression context or issue numbers.

Do not narrate straightforward syntax. If a subtle invariant depends on a line,
put the explanation next to that line.

### Formatting and lint

Ruff is authoritative:

- target Python: 3.10;
- line length: 100;
- lint families: `E`, `F`, `I`, `UP`, `B`, and `W`.

Do not reformat unrelated files. Use a targeted `# type: ignore[code]` only when
an external type surface requires it, and explain non-obvious ignores.

## TypeScript conventions

### Module and API style

- Use ES modules and include `.js` in relative import specifiers.
- Export the package surface explicitly from `js/src/index.ts`.
- Export public interfaces and type aliases with their implementations.
- Use `camelCase` for variables, functions, methods, properties, and option
  fields; use `PascalCase` for classes, interfaces, and exported types.
- Mirror shared Python concepts idiomatically:
  `spent_usd` becomes `spentUsd`, `reserve_tool` becomes `reserveTool`, and so
  on. Stable cross-runtime wire formats such as exported JSONL remain
  `snake_case`.
- Prefer `unknown` at untrusted boundaries and narrow it before use. `any` is
  reserved for framework compatibility surfaces where multiple supported SDK
  versions cannot share a useful imported type.

### Documentation and compatibility

Use JSDoc for public APIs and module comments for compatibility or lifecycle
contracts. Explain casts that compensate for a lagging library type definition.

The compiler runs in strict mode. The package builds ESM and CommonJS outputs,
declarations, and source maps targeting ES2022. A TypeScript change is not
complete until it builds, type-checks, and passes Vitest.

## Integration conventions

A provider adapter should generally contain:

1. a module docstring describing the optional extra and enforcement surface;
2. private helpers to extract the served model and usage from supported response
   shapes;
3. a model-selection helper that prefers the served model but uses a priceable
   requested alias when documented and safe;
4. explicit rejection of modes that cannot be metered honestly;
5. synchronous and asynchronous wrappers when the SDK exposes both; and
6. an explicit `__all__` for its supported public entry points.

Do not import an optional framework from the package root. Adapter tests should
use small fakes where possible and `pytest.importorskip` when the real framework
is necessary, so the bare installation remains valid.

New adapters must test at least:

- successful accrual and return-value passthrough;
- blocking before the provider method is reached;
- synchronous and asynchronous behavior, when both exist;
- provider exceptions releasing the reservation;
- missing usage releasing or failing according to the documented contract;
- unpriceable or missing model behavior;
- served-model versus requested-alias behavior;
- unsupported streaming or other modes; and
- concurrency when the integration can fan out.

## Testing conventions

### Python

- Put tests in `tests/test_<area>.py`.
- Use plain `test_<behavior>()` functions and explicit return annotations.
- Prefer small local fakes over network calls or credentials.
- Use `pytest.raises(..., match=...)` for both error type and meaningful message
  when the message is contractual.
- Use `pytest.warns` for fail-open/fail-closed warning behavior.
- Test boundary values, invalid types, non-finite numbers, and cleanup paths.
- Add a regression comment or issue reference when the test protects a subtle
  previously broken invariant.
- Async tests use `pytest-asyncio`; keep synchronous and asynchronous contracts
  equivalent.

The pytest configuration promotes `UnpriceableModelWarning` to an error unless
a test handles it explicitly. Do not suppress it globally.

### TypeScript

- Put tests in `js/test/<area>.test.ts`.
- Use Vitest's `describe`, `it`, `expect`, and `vi`.
- Keep fakes local and deterministic.
- Use `Promise.all` / `Promise.allSettled` to reproduce concurrency behavior,
  and assert that excess work was actually blocked.
- Add parity tests when a shared Python/TypeScript behavior or serialized shape
  changes.

### Test the public contract

Prefer driving behavior through public methods instead of mutating private state.
Inspect private state only when no public value can prove the invariant—for
example, verifying that a clamped balance did not hide a leaked reservation.

Examples are part of the tested product surface. Keep them runnable and verify
important examples through tests when practical.

## Cross-language parity

Python and TypeScript do not have identical feature sets, but shared features
should agree on:

- validation and fail-closed behavior;
- boundary and floating-point tolerance semantics;
- reservation ownership and concurrency guarantees;
- advisory meaning and defaults;
- spend-event fields and JSONL schema;
- pricing resolution and the vendored map; and
- equivalent documentation examples.

When changing a shared concept, search both trees before editing:

```bash
rg "concept_name|conceptName" src tests js/src js/test README.md js/README.md
```

If parity is intentionally not provided, say so in the main README and the
package-specific documentation instead of implying support.

## User-facing changes

For a public behavior or API change, review all of the following:

- `src/floe_guard/__init__.py` and/or `js/src/index.ts`;
- the relevant package README;
- root `README.md` and its adapter matrix;
- a small runnable example, if the feature benefits from one;
- `CHANGELOG.md`;
- package versions; and
- tests in every affected language.

The changelog follows Keep a Changelog and Semantic Versioning. Python and
TypeScript versions are independent, and entries identify `py`, `js`, or
`py + js`. Add user-facing changes under the current `Unreleased` entry and use
the established `Added`, `Changed`, or `Fixed` sections.

Package source or packaging metadata changes trigger the version guard:

- Python: changes under `src/floe_guard/` or to `pyproject.toml` require a new
  `pyproject.toml` version unless the PR is deliberately labeled
  `skip-version-guard`.
- TypeScript: changes under `js/src/` or to `js/package.json` require a new
  `js/package.json` version under the same exception.

When bumping Python, keep `src/floe_guard/__init__.py::__version__` in lockstep.
When bumping npm, update both `js/package.json` and `js/package-lock.json`.
Publishing is automated after CI succeeds on `main`; do not add ad hoc publish
steps to a feature contribution.

## Documentation and example style

- Lead with the concrete guarantee or limitation.
- Use direct language: “raises before the call,” “fails closed,” and “no API
  key” are preferred over broad marketing claims.
- Distinguish guaranteed enforcement from advisory or estimate-based behavior.
- State unsupported cases explicitly.
- Keep examples small, copyable, and focused on one behavior.
- Label partial framework snippets as fragments; do not present undefined
  surrounding objects as a complete runnable program.
- Preserve the project's terms: “local,” “in-process,” “no telemetry,”
  “estimate-based,” and “hosted Floe” have specific documented meanings.

## Scope and change discipline

- Prefer small, focused changes with behavior-describing tests.
- Reuse existing helpers and lifecycle contracts before introducing a parallel
  abstraction.
- Avoid unrelated cleanup, renaming, dependency upgrades, or formatting.
- Preserve backward compatibility unless the change explicitly proposes a
  breaking release.
- Open an issue before a large architectural change, as requested in
  `CONTRIBUTING.md`.
- Never weaken an existing check merely to make a provider response pass. First
  establish why the response can be metered safely.

## Verification checklist

Run the checks relevant to the changed surface. Before a full run, focused tests
are useful while iterating.

### Python

```bash
python -m pip install -e ".[dev]"
ruff check .
ruff format --check .
pytest -q
```

If an adapter changed, install its optional extra and run its focused test file.
Also consider the bare-core contract: optional imports must not break a test run
with only `.[dev]` installed.

### TypeScript

```bash
cd js
npm ci
npm run build
npm run typecheck
npm test
```

### Repository-wide checks

```bash
git diff --check
git diff --exit-code -- src/floe_guard/cost_map.json js/src/cost_map.json
```

Then review the diff as a maintainer would:

- Is the paid operation blocked before it starts?
- Can any error, cancellation, or missing-usage path leak a reservation?
- Can any input make spend appear lower than it is?
- Does parallel execution preserve the ceiling?
- Does the bare Python core still import without extras?
- Are Python and TypeScript still aligned where the feature is shared?
- Do tests prove the failure path, not only the happy path?
- Are public exports, docs, changelog, and versions complete?
- Are limitations described honestly?

## Source-of-truth hierarchy

When this guide and the repository disagree, use this order:

1. tests and CI-enforced behavior;
2. current implementation contracts and public API;
3. `CONTRIBUTING.md`, package metadata, and READMEs;
4. this guide.

Update this document when an accepted contribution establishes a new recurring
convention. Do not preserve a stale rule merely because it is written here.
