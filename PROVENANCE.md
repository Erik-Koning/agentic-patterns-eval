# Provenance

Every result in this project must be traceable to the exact code and data below. Update this file whenever a pinned input changes.

## APG (Adaptive Prompt Graph)

- **Source:** `~/Documents/Work/Code/EvolvingWisdomAgents`
- **Git state:** no commits (`git rev-parse HEAD` fails), so the tree is pinned by content hash.
  - **Preferred:** the owner commits and tags `apg-eval-baseline`, then records the SHA here.
- **Tree-hash method:** for each file, `shasum -a 256` of the sorted file list (excluding `__pycache__`, `.pytest_cache`, `.DS_Store`), formatted as `<sha>  <path>`. Then take `shasum -a 256` of that whole listing.

| Component | Paths hashed | sha256 | Recorded (UTC) |
|---|---|---|---|
| apg-core 0.1.0 | `python/packages/apg-core/src`, `python/packages/apg-core/pyproject.toml` | `22fb44ef882a4e6b71e720b8de25159a05a9109af3e4ff15f6564103dac040a3` | 2026-09-30T02:27Z |
| APG schema | `schema/` | `78b4568e6459e2efef3591b3d8da2bdc3d73ad62cd508881d9fe216bf4be9490` | 2026-09-30T02:27Z |

- **How it's installed:** non-editable path dependency (`[tool.uv.sources]` in `pyproject.toml`). The copy in `.venv` is frozen at install time. Re-run `uv sync --reinstall-package apg-core` after any upstream change, then update the hashes above.

## Environment

- **Python 3.14** (see `DECISIONS.md` D-001 for why not 3.12).
- **Resolved versions:** `uv.lock`. Key packages:
  - inspect-ai 0.3.273
  - lightrag-hku 1.5.7
  - openai 3.22.1
  - tiktoken 0.14.0

## Models and prices

To be filled in by readiness items E3/E5: model IDs per role, honoured parameters, and the price table with source URL and date.
