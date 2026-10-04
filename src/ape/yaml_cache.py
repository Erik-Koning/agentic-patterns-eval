"""Parsed YAML files, cached by path, size and modification time: the orchestrators read run_plan.yaml, models.yaml and
the price table hundreds of times per phase (every projection, group and fingerprint), and parsing the plan takes ~0.1 s.
Each call returns a deep copy, so a caller that edits what it got never changes another's view; a file edited on disk
(a test's copy, a hand edit between phases) is parsed again."""

import copy
from pathlib import Path
from typing import Any

import yaml

_CACHE: dict[str, tuple[tuple[int, int], Any]] = {}


def load(path: str | Path) -> Any:
    """`yaml.safe_load` of the file at `path` (a fresh copy of the cached parse while the file is unchanged)."""
    p = Path(path)
    st = p.stat()
    key, stamp = str(p.resolve()), (st.st_size, st.st_mtime_ns)
    hit = _CACHE.get(key)
    if hit is None or hit[0] != stamp:
        hit = (stamp, yaml.safe_load(p.read_text()))
        _CACHE[key] = hit
    return copy.deepcopy(hit[1])
