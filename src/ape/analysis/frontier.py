"""The S8 compute-matched frontier (ORCHESTRATOR_BRIEF_v2 §4.5): self-consistency over the S1 pool, built post hoc.

S1 runs K times per task (K = 8 in Studies A and B; S1's 3 main epochs are the pool's first 3). S8(k), k = 1..K, is
a k-run ensemble whose answer is the **plurality of the runs' canonical answer keys** (`main_load.answer_key`: the
part of the answer the scorer reads, normalised as it normalises, so equal keys always score alike):

- **Exhaustive subsampling.** S8(k)'s success on a task is the mean over all C(K, k) subsets of k of its K runs.
  That is the U-statistic of a k-run ensemble drawn from the task's run distribution, so it is unbiased for the
  success of a fresh k-run ensemble, and it uses every run (no resampling noise).
- **Ties** count as the expected success of a uniform random choice among the tied answers. No LLM aggregator is
  run post hoc. The live S8k3 arm (Study C, `agent/multi`) votes on the same keys but breaks ties with an LLM
  aggregator (the earliest tied attempt as its fallback), and its aggregator call is part of its cost, so the live
  arm and the post-hoc S8(3) are not interchangeable; the report compares them descriptively (`s8k3_check`).
- **Abstentions.** A run without an answer (cap hit, harness error; an F3 run that neither finished nor made a call;
  key None, coded -1) casts no vote, as in the live arm; it still costs. When every run of a subset abstains, the
  ensemble fails.
- **Cost.** S8(k)'s realised cost is the sum of its runs' costs, per meter, so its expected cost over the subsets is
  exactly k times the task's mean run cost. For wall-clock, `aggregate="max"` prices members run in parallel (the
  expected maximum over a random k-subset, from order statistics) instead of serially (`"sum"`, the default).
- **Interpolation** (`interpolate`): the condition-level curve (cost_k, success_k), k = 1..K, is linear between
  adjacent k. Another arm is compared with S8 at that arm's realised mean cost; per task, S8's value there is the
  same convex combination of the task's S8(k) and S8(k+1), so the paired difference arm - S8 stays per task. An arm
  cheaper than one S1 run is compared with S8(1) and flagged `below` (conservative for a claim that the arm beats
  S8); an arm costlier than S8(K) is `beyond` the frontier: reported, never extrapolated.
"""

from dataclasses import dataclass, field
from functools import cache
from itertools import combinations
from math import comb

import numpy as np

ABSTAIN = -1
STATUSES = ("inside", "below", "beyond", "degenerate")


@cache
def subset_masks(K: int) -> tuple[np.ndarray, np.ndarray]:
    """Every non-empty subset of K runs as a (S, K) 0/1 matrix, and each subset's size (S,)."""
    rows = [np.isin(np.arange(K), c) for k in range(1, K + 1) for c in combinations(range(K), k)]
    masks = np.array(rows, dtype=float)
    return masks, masks.sum(axis=1).astype(int)


def key_codes(keys) -> np.ndarray:
    """Per task (row), each run's answer key as an integer code (first occurrence order), None as ABSTAIN."""
    out = np.full((len(keys), max((len(r) for r in keys), default=0)), ABSTAIN, dtype=int)
    for i, row in enumerate(keys):
        seen: dict = {}
        for j, k in enumerate(row):
            if k is not None and not (isinstance(k, float) and np.isnan(k)):
                out[i, j] = seen.setdefault(k, len(seen))
    return out


def s8_success(codes: np.ndarray, success: np.ndarray) -> tuple[np.ndarray, int]:
    """(T, K) expected success of S8(k) per task (column k - 1) by exhaustive subsampling, and the number of tasks
    where runs with one answer key disagree on success (should be 0: the key is what the scorer reads; such a key
    scores as its runs' mean success).

    `codes` (T, K): non-negative answer codes per run, ABSTAIN for no answer; `success` (T, K) in {0, 1}."""
    codes, success = np.asarray(codes, dtype=int), np.asarray(success, dtype=float)
    T, K = codes.shape
    if T == 0 or K == 0:
        return np.zeros((T, K)), 0
    U = int(codes.max()) + 1 if (codes >= 0).any() else 1
    onehot = (codes[:, :, None] == np.arange(U)[None, None, :]).astype(float)  # (T, K, U); abstentions all-zero
    n_key = onehot.sum(axis=1)  # (T, U)
    s_key = np.einsum("tk,tku->tu", success, onehot)
    key_succ = np.divide(s_key, n_key, out=np.zeros_like(s_key), where=n_key > 0)
    inconsistent = int(np.any((n_key > 0) & (s_key > 0) & (s_key < n_key), axis=1).sum())
    masks, sizes = subset_masks(K)
    votes = np.einsum("sk,tku->tsu", masks, onehot)  # (T, S, U)
    top = votes.max(axis=2, keepdims=True)
    tied = (votes == top) & (top > 0)
    n_tied = tied.sum(axis=2)
    exp = np.divide((tied * key_succ[:, None, :]).sum(axis=2), n_tied, out=np.zeros((T, len(sizes))), where=n_tied > 0)
    by_size = (sizes[:, None] == np.arange(1, K + 1)[None, :]).astype(float)  # (S, K)
    return exp @ by_size / by_size.sum(axis=0), inconsistent


def s8_cost(cost: np.ndarray, aggregate: str = "sum") -> np.ndarray:
    """(T, K) expected realised cost of S8(k) per task over its k-subsets: k x the mean run cost (`sum`), or the
    expected maximum of k runs drawn without replacement (`max`: members run in parallel; wall-clock only)."""
    cost = np.asarray(cost, dtype=float)
    T, K = cost.shape
    if aggregate == "sum":
        return cost.mean(axis=1, keepdims=True) * np.arange(1, K + 1)[None, :]
    if aggregate != "max":
        raise ValueError(f"aggregate must be 'sum' or 'max', not {aggregate!r}")
    srt = np.sort(cost, axis=1)
    # P(the i-th smallest (1-based) is the maximum of a random k-subset) = C(i - 1, k - 1) / C(K, k)
    w = np.array([[comb(i - 1, k - 1) / comb(K, k) for k in range(1, K + 1)] for i in range(1, K + 1)])
    return srt @ w


@dataclass
class Frontier:
    """One condition's S8 curve on one meter: S8(k) for k = 1..K, per task and as condition means."""

    task_success: np.ndarray  # (T, K)
    cost: np.ndarray  # (K,) condition mean realised cost of S8(k)
    run_cost: float  # condition mean cost of one S1 run
    inconsistent_keys: int = 0
    info: dict = field(default_factory=dict)

    @property
    def K(self) -> int:
        return int(self.cost.shape[0])

    @property
    def success(self) -> np.ndarray:
        return self.task_success.mean(axis=0) if len(self.task_success) else np.full(self.K, np.nan)

    def points(self) -> list[dict]:
        return [{"k": k + 1, "cost": float(self.cost[k]), "success": float(self.success[k])} for k in range(self.K)]


def build(codes: np.ndarray, success: np.ndarray, cost: np.ndarray, aggregate: str = "sum", info: dict | None = None) -> Frontier:
    """The frontier of one condition from its pool: answer codes, success and one meter's cost, each (T, K)."""
    ts, bad = s8_success(codes, success)
    c = s8_cost(cost, aggregate)
    run = float(np.mean(cost)) if np.size(cost) else float("nan")
    return Frontier(ts, c.mean(axis=0) if len(c) else np.zeros(np.shape(cost)[1]), run, bad, info or {})


def interpolate(frontier: Frontier, target_cost: float) -> tuple[np.ndarray | None, dict]:
    """S8 at `target_cost` per task (T,), linear in cost between adjacent k, and how it was matched:
    {status, k, lambda, target_cost, cost_lo, cost_hi, success}. `status` is `inside`; `below` (cheaper than one run:
    S8(1), flagged); `beyond` (costlier than S8(K): None, never extrapolated); or `degenerate` (no positive cost on this
    meter, e.g. an unmetered mock run: None)."""
    cost, K = frontier.cost, frontier.K
    info: dict = {"target_cost": float(target_cost), "max_cost": float(cost[-1]) if K else None, "K": K}
    if K == 0 or not np.isfinite(target_cost) or not np.all(np.isfinite(cost)) or cost[-1] <= 0:
        return None, info | {"status": "degenerate"}
    tol = 1e-12 * max(1.0, abs(float(cost[-1])))
    if target_cost > cost[-1] + tol:
        return None, info | {"status": "beyond"}
    if target_cost < cost[0] - tol:
        vals = frontier.task_success[:, 0]
        return vals, info | {"status": "below", "k": 1, "lambda": 0.0, "success": float(vals.mean()) if len(vals) else None}
    j = int(np.searchsorted(cost, target_cost + tol, side="right")) - 1  # cost[j] <= target
    j = min(max(j, 0), K - 1)
    if j == K - 1:
        lam = 0.0
        vals = frontier.task_success[:, j]
    else:
        span = cost[j + 1] - cost[j]
        lam = float(np.clip((target_cost - cost[j]) / span, 0.0, 1.0)) if span > 0 else 0.0
        vals = (1 - lam) * frontier.task_success[:, j] + lam * frontier.task_success[:, j + 1]
    return vals, info | {"status": "inside", "k": j + 1, "lambda": lam, "cost_lo": float(cost[j]), "cost_hi": float(cost[min(j + 1, K - 1)]), "success": float(vals.mean()) if len(vals) else None}
