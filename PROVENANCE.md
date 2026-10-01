# Provenance

Every result in this project must be traceable to the exact code and data below. Update this file whenever a pinned input changes.

## APG (Adaptive Prompt Graph)

- **Source:** `~/Documents/Work/Code/EvolvingWisdomAgents`
- **Pinned commit:** annotated tag **`apg-eval-baseline`** → `d47f7f3749dac38e9f917429810136e4ec6ebb37` (2026-09-30). This is the repo's first commit.
  - Before committing, a scan confirmed: `.env` is ignored and unstaged; no staged file names look like secrets; there are no key-like strings in staged content; and the tree hashes below matched.
- **Tree hashes at the pinned commit** (method: sha256 of the sorted `<sha256>  <path>` listing, excluding `__pycache__`, `.pytest_cache`, `.DS_Store`):

| Component | Paths hashed | sha256 |
|---|---|---|
| apg-core 0.1.0 | `python/packages/apg-core/src`, `python/packages/apg-core/pyproject.toml` | `22fb44ef882a4e6b71e720b8de25159a05a9109af3e4ff15f6564103dac040a3` |
| APG schema | `schema/` | `78b4568e6459e2efef3591b3d8da2bdc3d73ad62cd508881d9fe216bf4be9490` |
| Vendored `src/ape/apg/apg.schema.json` | single file | `ba4f4170c19d956d3b049011ee6b6b13dc2d03e4f5566cb81d0f5deaa05bdbda` |

- **How it's installed:** a git source pinned to the tag (`[tool.uv.sources]` in `pyproject.toml`); `uv.lock` records the commit SHA.
  - The URL is a local `file://` path until APG has a remote. When it does, swap the URL and keep the tag.
- **Verified at the pin:** APG's suite (123 tests, including 66 conformance fixtures) and this project's suite pass against the installed commit.

## Environment

- **Python 3.14** (see `DECISIONS.md` D-001 for why not 3.12).
- **Resolved versions:** `uv.lock`. Key packages:
  - inspect-ai 0.3.273
  - lightrag-hku 1.5.7
  - openai 3.22.1
  - tiktoken 0.14.0

## Models and prices

To be filled in by readiness items E3/E5: model IDs per role, honoured parameters, and the price table with source URL and date.
