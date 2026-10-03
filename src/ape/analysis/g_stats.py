"""Study G statistics (BUILD_PLAN B10; D-029 is the method decision; CONTEXT_MANAGEMENT_AUDIT §2, §5, §7).

Pure functions on tidy data: the session table and item table of `ape.analysis.g_load` (or `g_power`'s simulated
ones, which share the schema), and measured capability per point ({point: value} or g_load's capability frame).

**Inference is design-based and clustered by session** (D-029). A session is one F8 world; its epochs are pooled
first. Every confirmatory estimand is a linear combination L = Σ_c a_c ḡ_c of per-point session means ḡ_c of a
per-session value (a paired difference, a linear form of several arms, or a linearised ratio):
- **Interval** (`lincomb`): variance Σ_w ψ_w² with ψ_w = Σ_c (a_c / n_c) √(n_c / (n_c − 1)) (d_wc − ḡ_c), so a world
  that appears at several points (same seeds) is one cluster across them; with distinct worlds this is exactly the
  Welch variance Σ_c a_c² s_c² / n_c. Welch–Satterthwaite df, capped at G − 1 when worlds are shared. This is the
  gate's validated world-clustered t-interval (D-023) generalised from equal-weight cell means to any weights; on
  a pooled mean it reproduces `gate_stats.cluster_t` (tests).
- **Sign-flip** (`flip_test`): the wild sign-flip with null-restricted residuals: the per-point means are moved to
  the nearest values satisfying the null, residuals are flipped per world, and the studentised statistic is
  recomputed. All 2^G sign vectors are enumerated up to G = 16 (an exact test for a single point), so the minimum
  attainable one-sided p is 2^−G: 4 sessions cannot go below 0.0625. Random flips above that.
- **Bootstrap** (`bootstrap`): worlds resampled (within point when no world is shared), percentile intervals;
  reported, never decisive (the gate found it anti-conservative with few clusters).
- **Ratios** (`ratio`): R = N / D of two linear combinations, with the delta-method interval, Fieller's interval
  (unbounded when D is not significantly away from 0) and the bootstrap with its share of draws at D ≤ `MIN_DEN`.
  A ratio is not estimated when D ≤ `MIN_DEN` (2 pp: no headroom). Thresholds on ratios (shares ≥ 0.5, recovery
  ≥ 0.8) are tested as linear forms, (N − r D) > 0, which needs no ratio inference; a per-point ratio that enters
  a slope (G-H2b) is linearised per session (`ratio_pseudo`).

**Hypotheses** (`ape.analysis.g_hypotheses` is the table): `gh1` (topology gap slope on measured capability, for any
single-agent reference), `gap_iut`, `headroom`, `tost` and `gh2` (context management), `gh3` (isolation and
specialisation shares, S-CM* recovery at cost), and the descriptive `degradation`, `short_control`,
`crossing_split`, `probe_table`, `probe_behaviour`, `taxonomy_table`, `cost_table` and `glmm`.

Nothing here raises on missing arms, points or sessions: a test that cannot run returns `testable: False` and a
`reason`.
"""

import math
import warnings
from collections.abc import Mapping, Sequence
from functools import cache
from itertools import product

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.stats import t as student_t

from .g_hypotheses import ALPHA, COST_METER, COST_RATIO, ISOLATION_SHARE, RECOVERY_SHARE, TOST_ESTIMAND, TOST_MARGIN
from .gate_stats import holm_test, t_bounds, t_p_greater, t_p_less

REPS = 10_000
SEED = 0
EXACT_FLIP_MAX = 16  # enumerate every sign vector up to this many clusters (65,536)
MIN_DEN = 0.02  # a ratio's denominator must exceed this (2 pp) to be estimated
SHORT_N = 10  # sessions of at most this many items are the short-session control
LOGIT_EPS = 0.5  # empirical logit: log((k + ε) / (n − k + ε))
MIN_SESSIONS = 2  # per point, for a variance
COST_METERS = ("cost_usd", "tokens", "calls", "wall_clock")
TOPO_ARMS = ("S1", "M1", "M2", "S-CM*")


# ---------- per-session values ----------


def capability_map(capability) -> dict[str, float]:
    """{point: measured capability} from a mapping or a frame with a `capability` column (index or `point` column)."""
    if capability is None:
        return {}
    if isinstance(capability, pd.DataFrame):
        df = capability.set_index("point") if "point" in capability.columns else capability
        return {str(k): float(v) for k, v in df["capability"].items() if v is not None and np.isfinite(v)}
    return {str(k): float(v) for k, v in dict(capability).items() if v is not None and np.isfinite(float(v))}


def _rows(sessions: pd.DataFrame, block: str | None, length: str | None, points: Sequence[str] | None) -> pd.DataFrame:
    if sessions is None or not len(sessions):
        return pd.DataFrame()
    df = sessions
    if block is not None and "block" in df:
        df = df[df["block"] == block]
    if length is not None and "N" in df:
        df = df[(df["N"] <= SHORT_N) == (length == "short")]
    if points is not None:
        df = df[df["point"].isin(list(points))]
    return df


def session_values(
    sessions: pd.DataFrame,
    arms: Sequence[str],
    *,
    block: str | None = None,
    length: str | None = "long",
    points: Sequence[str] | None = None,
    outcome: str = "item_success",
    scale: str = "prob",
    pre_overflow_of: str | None = None,
    positions: str | None = None,
) -> pd.DataFrame:
    """One row per (point, session) with one column per arm: the arm's outcome pooled over its epochs (NaN where it
    did not run the session).

    `item_success` pools items over epochs (k solved of n); `scale="logit"` takes the empirical logit of k / n. Other
    outcomes (dependency_success, report_exact, probe_f1, ...) are epoch means on the probability scale.
    `pre_overflow_of="S1"` keeps only the items before that arm's earliest overflow in the session, for every arm (the
    S1-pre reference); `positions="pre"/"post"` keeps the items before / from the session's reference W crossing."""
    if scale not in ("prob", "logit"):
        raise ValueError(f"scale {scale!r}: prob or logit")
    if scale == "logit" and outcome != "item_success":
        raise ValueError("the logit scale needs item counts (outcome='item_success')")
    df = _rows(sessions, block, length, points)
    cols = ["point", "session", *arms]
    if not len(df):
        return pd.DataFrame(columns=cols)
    df = df[df["arm"].isin(list(arms))]
    if not len(df):
        return pd.DataFrame(columns=cols)
    keys = ["point", "session", "arm"]
    if outcome != "item_success":
        vals = df.assign(_v=pd.to_numeric(df[outcome], errors="coerce")).dropna(subset=["_v"])
        agg = vals.groupby(keys)["_v"].mean()
    else:
        if pre_overflow_of is None and positions is None:
            k = df.groupby(keys)["items_solved"].sum()
            n = df.groupby(keys)["n_items"].sum()
        else:
            k_rows, n_rows, keep = _restricted_counts(df, pre_overflow_of, positions)
            sub = df.loc[keep].assign(_k=k_rows[keep], _n=n_rows[keep])
            k, n = sub.groupby(keys)["_k"].sum(), sub.groupby(keys)["_n"].sum()
        k, n = k[n > 0], n[n > 0]
        agg = np.log((k + LOGIT_EPS) / (n - k + LOGIT_EPS)) if scale == "logit" else k / n
    if not len(agg):
        return pd.DataFrame(columns=cols)
    res = agg.unstack("arm").reset_index()
    res.columns.name = None
    for a in arms:
        if a not in res:
            res[a] = np.nan
    return res[cols].sort_values(["point", "session"]).reset_index(drop=True)


def _restricted_counts(df: pd.DataFrame, pre_overflow_of: str | None, positions: str | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per row: items solved and items kept after the S1-pre cut and/or the pre / post crossing split; and which rows
    are usable (a session without the cut arm, or without a reference crossing, is not)."""
    cut = pd.Series(np.nan, index=df.index)
    keep = np.ones(len(df), dtype=bool)
    if pre_overflow_of is not None:
        ref = df[df["arm"] == pre_overflow_of]
        first = pd.to_numeric(ref["overflow_at"], errors="coerce").groupby([ref["point"], ref["session"]]).min()
        has = ref.groupby(["point", "session"]).size()
        idx = pd.MultiIndex.from_arrays([df["point"], df["session"]])
        keep &= has.reindex(idx).notna().to_numpy()
        cut = pd.Series(first.reindex(idx).to_numpy(), index=df.index) - 1  # items 1..cut survive; NaN: no overflow
    if positions is not None:
        wc = pd.to_numeric(df["w_crossing_item"], errors="coerce") if "w_crossing_item" in df else pd.Series(np.nan, index=df.index)
        keep &= wc.notna().to_numpy()
    k_out, n_out = np.zeros(len(df)), np.zeros(len(df))
    for i, (o, c, x) in enumerate(zip(df["outcomes"], cut, df["w_crossing_item"] if positions is not None else cut, strict=True)):
        if not keep[i]:
            continue
        o = np.asarray(o, dtype=float)
        if np.isfinite(c):
            o = o[: int(c)]
        if positions is not None:
            o = o[: int(x) - 1] if positions == "pre" else o[int(x) - 1 :]
        k_out[i], n_out[i] = o.sum(), len(o)
    return k_out, n_out, keep


def long_values(wide: pd.DataFrame, expr: Mapping[str, float]) -> pd.DataFrame:
    """[point, session, value] for a linear form Σ coef × arm of a `session_values` frame (rows missing any arm dropped)."""
    if not len(wide) or any(a not in wide for a in expr):
        return pd.DataFrame(columns=["point", "session", "value"])
    v = sum(c * wide[a] for a, c in expr.items())
    out = pd.DataFrame({"point": wide["point"], "session": wide["session"], "value": v})
    return out.dropna(subset=["value"]).reset_index(drop=True)


# ---------- linear combinations of per-point session means ----------


def _matrix(v: pd.DataFrame, weights: Mapping[str, float]) -> tuple[list, list, np.ndarray, np.ndarray]:
    """(worlds, points, D, a): D[w, c] the value of world w at point c (NaN where it did not run)."""
    pts = [c for c, a in weights.items() if a != 0]
    a = np.array([float(weights[c]) for c in pts])
    if not len(v) or not pts:
        return [], pts, np.full((0, len(pts)), np.nan), a
    point = v["point"].to_numpy()
    val = v["value"].to_numpy(dtype=float)
    sel = np.isin(point, pts) & ~np.isnan(val)
    if not sel.any():
        return [], pts, np.full((0, len(pts)), np.nan), a
    wcode, worlds = pd.factorize(v["session"].to_numpy()[sel], sort=True)
    ccode = pd.Index(pts).get_indexer(point[sel])
    D = np.full((len(worlds), len(pts)), np.nan)
    flat = wcode * len(pts) + ccode
    if len(np.unique(flat)) != len(flat):
        raise ValueError("lincomb: one value per (point, session); pool epochs first (session_values)")
    D[wcode, ccode] = val[sel]
    return list(worlds), pts, D, a


def _core(D: np.ndarray, M: np.ndarray, a: np.ndarray):
    """Vectorised over leading axes: D (..., G, C) with mask M (G, C). Returns est, V, means, s2, n, psi."""
    n = M.sum(0).astype(float)
    Dm = np.where(M, D, 0.0)
    g = Dm.sum(-2) / n
    e = np.where(M, D - g[..., None, :], 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        s2 = (e**2).sum(-2) / (n - 1)
        b = a / n * np.sqrt(n / (n - 1))
    psi = (e * b[..., None, :]).sum(-1)
    return (a * g).sum(-1), (psi**2).sum(-1), g, s2, n, psi


def _df(a: np.ndarray, s2: np.ndarray, n: np.ndarray, shared: bool, G: int) -> np.ndarray:
    comps = a**2 * s2 / n
    tot = comps.sum(-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        df = tot**2 / (comps**2 / (n - 1)).sum(-1)
    df = np.where(np.isfinite(df), df, np.nan)
    return np.minimum(df, G - 1) if shared else df


def lincomb(v: pd.DataFrame, weights: Mapping[str, float]) -> dict:
    """L = Σ_c a_c ḡ_c over per-(point, session) values `v` [point, session, value], with the session-clustered
    variance and Satterthwaite df (module docstring). `se` is None (with `reason`) when a weighted point has fewer than
    2 sessions; se 0 with df None when every session agrees. The dict works with gate_stats.t_bounds / t_p_*."""
    worlds, pts, D, a = _matrix(v, weights)
    out: dict = {"est": None, "se": None, "df": None, "points": {}, "sessions": len(worlds), "shared": False}
    if not pts:
        return out | {"reason": "no weighted points"}
    M = ~np.isnan(D)
    n = M.sum(0)
    shared = bool((M.sum(1) > 1).any())
    out["shared"] = shared
    for j, c in enumerate(pts):
        col = D[M[:, j], j]
        out["points"][c] = {"n": int(n[j]), "mean": float(col.mean()) if len(col) else None, "sd": float(col.std(ddof=1)) if len(col) > 1 else None, "weight": float(a[j])}
    if (empty := [c for j, c in enumerate(pts) if n[j] == 0]):
        return out | {"reason": f"no sessions at {empty}"}
    est = float(sum(a[j] * out["points"][c]["mean"] for j, c in enumerate(pts)))
    out["est"] = est
    if (short := [c for j, c in enumerate(pts) if n[j] < MIN_SESSIONS]):
        return out | {"reason": f"fewer than {MIN_SESSIONS} sessions at {short}"}
    _, V, _, s2, nn, _ = _core(D, M, a)
    out["se"] = float(math.sqrt(max(float(V), 0.0)))
    if V > 0:
        df = float(_df(a, s2, nn, shared, len(worlds)))
        out["df"] = df if np.isfinite(df) else None
    return out


@cache
def _signs(G: int) -> np.ndarray:
    """Every sign vector of length G, the identity first."""
    eta = np.array(list(product((1.0, -1.0), repeat=G)))
    eta.setflags(write=False)
    return eta


def flip_test(v: pd.DataFrame, weights: Mapping[str, float], null: float = 0.0, alternative: str = "greater", reps: int = REPS, seed: int = SEED, exact_max: int = EXACT_FLIP_MAX) -> dict:
    """Wild sign-flip p-value for L vs `null` (one-sided, `alternative` greater | less), studentised, with residuals
    restricted to the null. Exact (every sign vector) up to `exact_max` clusters while 2^G ≤ 8 × `reps` (so the
    report's 10,000 resamples enumerate up to 16 clusters and a power run's 999 up to 12), else `reps` random sign
    vectors plus the identity. `min_p` is the floor 2^−G no sign-flip over G clusters can go below; `resolution` is
    1 / (sign vectors used)."""
    worlds, pts, D, a = _matrix(v, weights)
    M = ~np.isnan(D)
    G = len(worlds)
    if not pts or G == 0 or (M.sum(0) < MIN_SESSIONS).any():
        return {"p": None, "exact": False, "flips": 0, "min_p": None, "reason": "fewer than 2 sessions at a weighted point"}
    est, V, g, s2, n, _ = _core(D, M, a)
    sign = 1.0 if alternative == "greater" else -1.0
    if not V > 0:  # every session agrees: as gate_stats.t_p_greater with se 0
        return {"p": 0.0 if sign * (est - null) > 0 else 1.0, "exact": True, "flips": 1, "min_p": float(2.0**-G), "clusters": G, "degenerate": True}
    lam = (est - null) / float((a**2 / n).sum())
    mu = g - lam * a / n  # restricted means: Σ a μ = null
    resid = np.where(M, D - mu, 0.0)
    if G <= exact_max and 2**G <= 8 * max(reps, 1):
        eta = _signs(G)
        exact = True
    else:
        rng = np.random.default_rng(seed)
        eta = np.vstack([np.ones(G), rng.choice((-1.0, 1.0), size=(reps, G))])
        exact = False
    Ds = np.where(M, mu + eta[:, :, None] * resid, np.nan)
    est_s, V_s, *_ = _core(Ds, M, a)
    t_obs = sign * (est - null) / math.sqrt(V)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_s = sign * (est_s - null) / np.sqrt(V_s)
    t_s = np.where(np.isnan(t_s), 0.0, t_s)
    p = float(np.sum(t_s >= t_obs - 1e-9 * max(1.0, abs(t_obs))) / len(eta))  # the identity flip is in both enumerations
    return {"p": p, "exact": exact, "flips": int(len(eta)), "min_p": float(2.0**-G), "resolution": float(1 / len(eta)), "clusters": G}


def bootstrap(v: pd.DataFrame, weights: Mapping[str, float], reps: int = 2000, seed: int = SEED, keep_nan: bool = False) -> np.ndarray:
    """Bootstrap replicates of L, resampling worlds (within point when no world is shared across points). Replicates
    where a point drew no session are NaN: dropped, unless `keep_nan` (so two statistics on the same worlds and seed
    stay aligned)."""
    worlds, pts, D, a = _matrix(v, weights)
    M = ~np.isnan(D)
    G = len(worlds)
    if G == 0 or not pts:
        return np.array([])
    rng = np.random.default_rng(seed)
    shared = bool((M.sum(1) > 1).any())
    Dz = np.where(M, D, 0.0)
    if shared:
        W = rng.multinomial(G, np.full(G, 1 / G), size=reps).astype(float)  # (reps, G)
        num = W @ Dz
        den = W @ M.astype(float)
    else:
        num = np.zeros((reps, len(pts)))
        den = np.zeros((reps, len(pts)))
        for j in range(len(pts)):
            idx = np.flatnonzero(M[:, j])
            if not len(idx):
                continue
            w = rng.multinomial(len(idx), np.full(len(idx), 1 / len(idx)), size=reps).astype(float)
            num[:, j] = w @ D[idx, j]
            den[:, j] = w.sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        g = num / den
    out = (g * a).sum(1)
    return out if keep_nan else out[np.isfinite(out)]


def _p(st: dict, null: float, alternative: str) -> float | None:
    return t_p_greater(st, null) if alternative == "greater" else t_p_less(st, null)


def contrast(
    v: pd.DataFrame,
    weights: Mapping[str, float],
    *,
    null: float = 0.0,
    alternative: str = "greater",
    alpha: float = ALPHA,
    reps: int = REPS,
    seed: int = SEED,
    flip: bool = True,
    boot: int = 0,
    primary: str = "t",
) -> dict:
    """One-sided test of L against `null` with everything the report shows: estimate, se, df, the two-sided (1 − 2α)
    interval (its relevant end is the one-sided α bound), the t and sign-flip p-values, whether the flip test can reach
    α at all, an optional bootstrap interval, and the decision by the `primary` test (t | flip)."""
    st = lincomb(v, weights)
    lo, hi = t_bounds(st, alpha) if st.get("se") is not None else (None, None)
    p_t = _p(st, null, alternative) if st.get("se") is not None else None
    fl = flip_test(v, weights, null, alternative, reps=reps, seed=seed) if flip and st.get("se") is not None else {"p": None, "min_p": None, "exact": None}
    out = {
        "est": st["est"],
        "se": st["se"],
        "df": st["df"],
        "ci": [lo, hi],
        "null": null,
        "alternative": alternative,
        "alpha": alpha,
        "p_t": p_t,
        "p_flip": fl.get("p"),
        "flip_exact": fl.get("exact"),
        "flip_min_p": fl.get("min_p"),
        "flip_reachable": None if fl.get("min_p") is None else bool(fl["min_p"] <= alpha),
        "points": st["points"],
        "sessions": st["sessions"],
        "shared_sessions": st["shared"],
        "testable": st.get("se") is not None,
        "reason": st.get("reason"),
    }
    if boot and st.get("se") is not None:
        b = bootstrap(v, weights, reps=boot, seed=seed)
        out["boot_ci"] = [float(np.quantile(b, alpha)), float(np.quantile(b, 1 - alpha))] if len(b) else [None, None]
    p = out["p_t"] if primary == "t" else out["p_flip"]
    out["primary"] = primary
    out["p"] = p
    out["reject"] = bool(p is not None and p <= alpha)
    return out


def min_flip_p(sessions: int) -> float:
    """Smallest one-sided p an exact sign-flip over `sessions` independent sessions can give: 2^−n."""
    return 2.0**-int(sessions)


def min_sessions_for(alpha: float) -> int:
    """Fewest sessions whose exact sign-flip can reach one-sided `alpha`."""
    return int(math.ceil(-math.log2(alpha)))


# ---------- ratios ----------


def _psi(v: pd.DataFrame, weights: Mapping[str, float], worlds: list) -> tuple[float, np.ndarray, dict]:
    """Estimate and per-world influence vector (aligned to `worlds`) of a linear combination."""
    wl, _, D, a = _matrix(v, weights)
    M = ~np.isnan(D)
    est, V, g, s2, n, psi = _core(D, M, a)
    full = np.zeros(len(worlds))
    pos = {w: i for i, w in enumerate(worlds)}
    for i, w in enumerate(wl):
        full[pos[w]] = psi[i]
    return float(est), full, {"n": n, "s2": s2, "a": a}


def ratio(num: pd.DataFrame, den: pd.DataFrame, weights: Mapping[str, float] | None = None, *, alpha: float = ALPHA, min_den: float = MIN_DEN, boot: int = 2000, seed: int = SEED) -> dict:
    """R = N / D for two per-session value frames on the same sessions (N = Σ a ḡ_num, D = Σ a ḡ_den; default equal
    weights over the points present). Delta-method t-interval, Fieller interval and bootstrap percentile interval,
    each two-sided (1 − 2α). Not estimated when D ≤ `min_den`; `den_significant` is False when D's one-sided lower
    bound is ≤ 0, and then Fieller's set is unbounded."""
    keys = ["point", "session"]
    if not len(num) or not len(den):
        return {"est": None, "testable": False, "reason": "no paired sessions"}
    m = num.merge(den, on=keys, suffixes=("_n", "_d")).dropna()
    if not len(m):
        return {"est": None, "testable": False, "reason": "no paired sessions"}
    pts = sorted(m["point"].unique())
    weights = dict(weights) if weights is not None else {c: 1.0 / len(pts) for c in pts}
    vn = m[[*keys, "value_n"]].rename(columns={"value_n": "value"})
    vd = m[[*keys, "value_d"]].rename(columns={"value_d": "value"})
    sn, sd = lincomb(vn, weights), lincomb(vd, weights)
    out: dict = {"num": sn["est"], "den": sd["est"], "est": None, "ci_delta": [None, None], "ci_fieller": [None, None], "fieller_bounded": None, "ci_boot": [None, None], "boot_den_small_share": None, "den_significant": None, "testable": False, "reason": None, "sessions": sn["sessions"]}
    if sd["est"] is None or sn["est"] is None:
        return out | {"reason": sd.get("reason") or sn.get("reason")}
    if sd["est"] <= min_den:
        return out | {"reason": f"denominator {sd['est']:+.3f} ≤ {min_den}: no headroom, the ratio is not estimated"}
    R = sn["est"] / sd["est"]
    out["est"] = float(R)
    if sd["se"] is None or sn["se"] is None:
        return out | {"reason": sd.get("reason") or sn.get("reason")}
    worlds = sorted(m["session"].unique())
    _, pn, _ = _psi(vn, weights, worlds)
    _, pdn, _ = _psi(vd, weights, worlds)
    Vn, Vd, C = float(pn @ pn), float(pdn @ pdn), float(pn @ pdn)
    df = sd["df"] or sn["df"]
    q = float(student_t.ppf(1 - alpha, df)) if df else float(norm.ppf(1 - alpha))
    psi_r = (pn - R * pdn) / sd["est"]
    se_r = float(math.sqrt(psi_r @ psi_r))
    out["ci_delta"] = [R - q * se_r, R + q * se_r]
    out["se"] = se_r
    out["df"] = df
    den_lo = sd["est"] - q * math.sqrt(Vd)
    out["den_significant"] = bool(den_lo > 0)
    # Fieller: {r : (N − r D)² ≤ q² (Vn − 2 r C + r² Vd)}
    A = sd["est"] ** 2 - q**2 * Vd
    B = -2 * (sn["est"] * sd["est"] - q**2 * C)
    Cc = sn["est"] ** 2 - q**2 * Vn
    disc = B**2 - 4 * A * Cc
    if A > 0 and disc >= 0:
        r1, r2 = sorted(((-B - math.sqrt(disc)) / (2 * A), (-B + math.sqrt(disc)) / (2 * A)))
        out["ci_fieller"], out["fieller_bounded"] = [r1, r2], True
    else:
        out["fieller_bounded"] = False
    if boot:
        bn = bootstrap(vn, weights, reps=boot, seed=seed, keep_nan=True)
        bd = bootstrap(vd, weights, reps=boot, seed=seed, keep_nan=True)  # same seed and worlds: the same resamples
        fin = np.isfinite(bn) & np.isfinite(bd)
        bn, bd = bn[fin], bd[fin]
        if len(bd):
            small = bd <= min_den
            out["boot_den_small_share"] = float(small.mean())
            r = bn[~small] / bd[~small]
            if len(r):
                out["ci_boot"] = [float(np.quantile(r, alpha)), float(np.quantile(r, 1 - alpha))]
    out["testable"] = True
    if not out["den_significant"]:
        out["reason"] = "denominator not bounded away from 0: Fieller's interval is unbounded; read the ratio with care"
    return out


def ratio_pseudo(num: pd.DataFrame, den: pd.DataFrame, min_den: float = MIN_DEN) -> tuple[pd.DataFrame, dict]:
    """Per-point ratios R_c = n̄_c / d̄_c linearised per session: z_sc = R_c + (n_sc − R_c d_sc) / d̄_c, whose
    per-point mean is R_c and whose spread gives the delta-method variance. Points with d̄_c ≤ `min_den` are dropped
    and named in the second value."""
    keys = ["point", "session"]
    m = num.merge(den, on=keys, suffixes=("_n", "_d")).dropna() if len(num) and len(den) else pd.DataFrame(columns=[*keys, "value_n", "value_d"])
    rows, info = [], {}
    for c, g in m.groupby("point"):
        dbar = float(g["value_d"].mean())
        if dbar <= min_den:
            info[c] = {"R": None, "den": dbar, "reason": f"headroom {dbar:+.3f} ≤ {min_den}"}
            continue
        R = float(g["value_n"].mean()) / dbar
        info[c] = {"R": R, "den": dbar, "n": len(g)}
        rows.append(pd.DataFrame({"point": c, "session": g["session"], "value": R + (g["value_n"] - R * g["value_d"]) / dbar}))
    v = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["point", "session", "value"])
    return v, info


# ---------- slopes on measured capability ----------


def slope_weights(n: Mapping[str, int], x: Mapping[str, float], weighting: str = "sessions") -> dict[str, float]:
    """Weights a_c with Σ a_c ḡ_c = the OLS slope of session values on x (`sessions`: every session counts once) or the
    slope through the per-point means (`points`)."""
    pts = [c for c in n if c in x and n[c] > 0]
    w = {c: float(n[c]) if weighting == "sessions" else 1.0 for c in pts}
    tot = sum(w.values())
    if tot == 0:
        return {}
    xbar = sum(w[c] * x[c] for c in pts) / tot
    sxx = sum(w[c] * (x[c] - xbar) ** 2 for c in pts)
    if sxx <= 0:
        return {}
    return {c: w[c] * (x[c] - xbar) / sxx for c in pts}


def _span(x: Mapping[str, float], pts) -> float:
    xs = [x[c] for c in pts]
    return float(max(xs) - min(xs)) if xs else 0.0


def slope_test(v: pd.DataFrame, capability, *, null: float = 0.0, alternative: str = "less", alpha: float = ALPHA, reps: int = REPS, seed: int = SEED, weighting: str = "sessions", flip: bool = True, primary: str = "t", boot: int = 0) -> dict:
    """The slope of per-session values on measured capability (points without capability, or with fewer than 2
    sessions, are left out and named). Also reported per span: the change along the fitted line from the lowest to
    the highest point. `boot` > 0 adds a session-bootstrap percentile interval (`boot_ci`, `boot_ci_span`)."""
    x = capability_map(capability)
    n = v.groupby("point")["value"].count().to_dict() if len(v) else {}
    dropped = {c: ("no capability" if c not in x else f"{k} session(s)") for c, k in n.items() if c not in x or k < MIN_SESSIONS}
    use = {c: k for c, k in n.items() if c not in dropped}
    base = {"testable": False, "points_used": sorted(use, key=lambda c: x.get(c, 0)), "points_dropped": dropped, "capability": {c: x.get(c) for c in n}}
    if len(use) < 2:
        return base | {"reason": f"a slope needs ≥ 2 capability points with ≥ {MIN_SESSIONS} sessions; have {sorted(use)}" + (f" (dropped {dropped})" if dropped else "")}
    w = slope_weights(use, x, weighting)
    if not w:
        return base | {"reason": "the capability points do not differ"}
    res = contrast(v, w, null=null, alternative=alternative, alpha=alpha, reps=reps, seed=seed, flip=flip, primary=primary, boot=boot)
    span = _span(x, use)
    res |= base | {"testable": res["testable"], "span": span, "weights": w}
    res["est_span"] = None if res["est"] is None else res["est"] * span
    res["ci_span"] = [None if b is None else b * span for b in res["ci"]]
    if "boot_ci" in res:
        res["boot_ci_span"] = [None if b is None else b * span for b in res["boot_ci"]]
    return res


# ---------- G-H1 ----------

REFERENCE_ARMS = {"S-CM*": ("S-CM*", None), "S1": ("S1", None), "S1-pre": ("S1", "S1")}


def gh1(sessions: pd.DataFrame, capability, *, topology: str = "M2", reference: str = "S-CM*", block: str | None = "topo", outcome: str = "item_success", scale: str = "logit", alpha: float = ALPHA, reps: int = REPS, seed: int = SEED, weighting: str = "sessions", flip: bool = True, primary: str = "t", boot: int = 0) -> dict:
    """G-H1: the per-session gap Δ = topology − reference regressed on measured capability; H1: slope < 0 (the gap
    shrinks). `reference`: S-CM*, S1 (as scored: overflow fails the rest of the session) or S1-pre (both arms on the
    items before S1's earliest overflow in the session). Per-point gaps are reported with their intervals."""
    if reference not in REFERENCE_ARMS:
        raise ValueError(f"reference {reference!r}: one of {sorted(REFERENCE_ARMS)}")
    ref, pre = REFERENCE_ARMS[reference]
    wide = session_values(sessions, [topology, ref], block=block, length=None, outcome=outcome, scale=scale, pre_overflow_of=pre)
    v = long_values(wide, {topology: 1.0, ref: -1.0})
    gaps = {}
    for c, g in v.groupby("point"):
        gaps[c] = _brief(contrast(g, {c: 1.0}, alternative="greater", alpha=alpha, reps=reps, seed=seed, flip=flip))
    res = slope_test(v, capability, alternative="less", alpha=alpha, reps=reps, seed=seed, weighting=weighting, flip=flip, primary=primary, boot=boot)
    return {"topology": topology, "reference": reference, "scale": scale, "outcome": outcome, "gaps": gaps} | res


def _brief(c: dict) -> dict:
    keys = ("est", "se", "df", "ci", "p_t", "p_flip", "flip_min_p", "flip_reachable", "sessions", "testable", "reason", "reject")
    out = {k: c.get(k) for k in keys}
    out["n"] = sum(p["n"] for p in c.get("points", {}).values()) if c.get("points") else 0
    return out


# ---------- G-H2 ----------


def gap_iut(sessions: pd.DataFrame, *, block: str | None = "cm", high: str = "O-state", low: str = "CM0", length: str = "long", outcome: str = "item_success", scale: str = "prob", alpha: float = ALPHA, reps: int = REPS, seed: int = SEED, flip: bool = True, primary: str = "t") -> dict:
    """G-H2a: Gap_T = s(high) − s(low) > 0 at every point (intersection-union: each point one-sided at α, no
    adjustment). `all_positive` is the IUT decision; points that cannot be tested make it False."""
    wide = session_values(sessions, [high, low], block=block, length=length, outcome=outcome, scale=scale)
    v = long_values(wide, {high: 1.0, low: -1.0})
    points = {c: _brief(contrast(g, {c: 1.0}, alternative="greater", alpha=alpha, reps=reps, seed=seed, flip=flip, primary=primary)) for c, g in v.groupby("point")}
    tested = [c for c, r in points.items() if r["testable"]]
    flip_ok = all(r["p_flip"] is not None and r["p_flip"] <= alpha for r in points.values()) if points else False
    return {
        "high": high,
        "low": low,
        "points": points,
        "all_positive": bool(points) and len(tested) == len(points) and all(points[c]["reject"] for c in points),
        "all_positive_flip": bool(points) and flip_ok,
        "flip_unreachable": [c for c, r in points.items() if r["flip_reachable"] is False],
        "testable": bool(tested),
        "reason": None if points else "no paired O-state / CM0 sessions",
    }


def headroom(sessions: pd.DataFrame, strategy: str, *, block: str | None = "cm", high: str = "O-state", low: str = "CM0", length: str = "long", outcome: str = "item_success", alpha: float = ALPHA, boot: int = 2000, seed: int = SEED, min_den: float = MIN_DEN) -> dict:
    """R_x = (s_x − s_CM0) / (s_O − s_CM0) per point (and pooled with equal point weights), on the probability scale."""
    wide = session_values(sessions, [strategy, high, low], block=block, length=length, outcome=outcome)
    num, den = long_values(wide, {strategy: 1.0, low: -1.0}), long_values(wide, {high: 1.0, low: -1.0})
    keep = wide.dropna()[["point", "session"]] if len(wide) else wide
    num = num.merge(keep, on=["point", "session"]) if len(num) else num
    den = den.merge(keep, on=["point", "session"]) if len(den) else den
    points = {c: ratio(num[num["point"] == c], den[den["point"] == c], alpha=alpha, boot=boot, seed=seed, min_den=min_den) for c in sorted(set(num["point"]) if len(num) else [])}
    pooled = ratio(num, den, alpha=alpha, boot=boot, seed=seed, min_den=min_den) if len(points) > 1 else None
    return {"strategy": strategy, "points": points, "pooled": pooled, "testable": any(p.get("testable") for p in points.values())}


def tost(sessions: pd.DataFrame, capability, strategy: str, *, estimand: str = TOST_ESTIMAND, margin: float | None = None, block: str | None = "cm", high: str = "O-state", low: str = "CM0", scale: str = "logit", alpha: float = ALPHA, reps: int = REPS, seed: int = SEED, flip: bool = True, primary: str = "t", min_den: float = MIN_DEN) -> dict:
    """G-H2b: equivalence of a strategy's effect across capability. θ = the slope on measured capability × the span
    (the change along the fitted line from the lowest to the highest point), tested against ±margin with two one-sided
    tests at α each (equivalent when both reject; a (1 − 2α) interval inside the margin).

    estimand `R`: headroom recovered per point, linearised per session (default; scale-free, and it normalises CM0's
    overflow, which makes raw gains grow with capability even at a constant R). `gain`: s_x − s_CM0 on `scale`
    (the audit's logit-unit formulation)."""
    margin = TOST_MARGIN[estimand] if margin is None else margin
    x = capability_map(capability)
    if estimand == "R":
        wide = session_values(sessions, [strategy, high, low], block=block, length="long")
        num, den = long_values(wide, {strategy: 1.0, low: -1.0}), long_values(wide, {high: 1.0, low: -1.0})
        v, info = ratio_pseudo(num, den, min_den)
    elif estimand == "gain":
        wide = session_values(sessions, [strategy, low], block=block, length="long", scale=scale)
        v, info = long_values(wide, {strategy: 1.0, low: -1.0}), {}
    else:
        raise ValueError(f"estimand {estimand!r}: R or gain")
    n = v.groupby("point")["value"].count().to_dict() if len(v) else {}
    use = {c: k for c, k in n.items() if c in x and k >= MIN_SESSIONS}
    base = {"strategy": strategy, "estimand": estimand, "margin": margin, "per_point": info, "points_used": sorted(use, key=lambda c: x[c]), "equivalent": False, "testable": False}
    unusable = {c: r["reason"] for c, r in info.items() if r.get("R") is None}
    if len(use) < 2:
        return base | {"reason": f"needs ≥ 2 capability points with ≥ {MIN_SESSIONS} sessions and headroom; have {sorted(use)}" + (f"; no headroom at {unusable}" if unusable else "")}
    w = slope_weights(use, x)
    span = _span(x, use)
    ws = {c: a * span for c, a in w.items()}
    lower = contrast(v, ws, null=-margin, alternative="greater", alpha=alpha, reps=reps, seed=seed, flip=flip, primary=primary)
    upper = contrast(v, ws, null=margin, alternative="less", alpha=alpha, reps=reps, seed=seed + 1, flip=flip, primary=primary)
    pt = None if lower["p_t"] is None or upper["p_t"] is None else max(lower["p_t"], upper["p_t"])
    pf = None if lower["p_flip"] is None or upper["p_flip"] is None else max(lower["p_flip"], upper["p_flip"])
    p = pt if primary == "t" else pf
    return base | {
        "est": lower["est"],
        "se": lower["se"],
        "df": lower["df"],
        "ci": lower["ci"],
        "span": span,
        "p_t": pt,
        "p_flip": pf,
        "p_noninferior_t": lower["p_t"],
        "flip_min_p": lower["flip_min_p"],
        "p": p,
        "equivalent": bool(p is not None and p <= alpha),
        "testable": lower["testable"],
        "reason": lower.get("reason") or (f"no headroom at {unusable} (left out)" if unusable else None),
        "sessions": lower["sessions"],
    }


def gh2(sessions: pd.DataFrame, capability, *, strategies: Sequence[str] = ("CM-sum", "CM-todo"), estimand: str = TOST_ESTIMAND, margin: float | None = None, alpha: float = ALPHA, reps: int = REPS, seed: int = SEED, boot: int = 2000, flip: bool = True, primary: str = "t", block: str | None = "cm") -> dict:
    """G-H2: Gap_T at every point (G-H2a), R_x per strategy present (G-H2c), and the TOST per confirmatory strategy with
    Holm across them (G-H2b)."""
    gaps = gap_iut(sessions, alpha=alpha, reps=reps, seed=seed, flip=flip, primary=primary, block=block)
    arms = sorted(set(_rows(sessions, block, "long", None).get("arm", pd.Series(dtype=str))) - {"O-state", "CM0"})
    heads = {s: headroom(sessions, s, alpha=alpha, boot=boot, seed=seed, block=block) for s in arms}
    tosts = {s: tost(sessions, capability, s, estimand=estimand, margin=margin, alpha=alpha, reps=reps, seed=seed, flip=flip, primary=primary, block=block) for s in strategies}
    pvals = {s: r["p"] if r["testable"] else None for s, r in tosts.items()}
    levels, rejected, tested = holm_test(pvals, alpha) if pvals else ({}, {}, {})
    for s, r in tosts.items():
        r["holm_level"], r["holm_equivalent"], r["holm_tested"] = levels.get(s), bool(rejected.get(s)), bool(tested.get(s))
    return {"gap": gaps, "headroom": heads, "tost": tosts}


def short_control(sessions: pd.DataFrame, *, point: str | None = None, high: str = "O-state", low: str = "CM0", block: str | None = "cm", alpha: float = ALPHA, reps: int = REPS, seed: int = SEED) -> dict:
    """The N ≤ 10 control: Gap at the short length vs at the long length at the same point (different sessions, so a
    Welch contrast); H1 for "gains grow with length": Gap_long − Gap_short > 0 (descriptive)."""
    short = session_values(sessions, [high, low], block=block, length="short")
    if not len(short):
        return {"testable": False, "reason": "no short-session control sessions"}
    point = point or short["point"].iloc[0]
    long_ = session_values(sessions, [high, low], block=block, length="long", points=[point])
    vs = long_values(short[short["point"] == point], {high: 1.0, low: -1.0}).assign(point="short")
    vl = long_values(long_, {high: 1.0, low: -1.0}).assign(point="long")
    vl["session"] = "L:" + vl["session"].astype(str)  # distinct clusters even if a world id repeats
    v = pd.concat([vs, vl], ignore_index=True)
    out = {"point": point, "gap_short": _brief(contrast(vs, {"short": 1.0}, alpha=alpha, reps=reps, seed=seed)) if len(vs) else None, "gap_long": _brief(contrast(vl, {"long": 1.0}, alpha=alpha, reps=reps, seed=seed)) if len(vl) else None}
    if not len(vs) or not len(vl):
        return out | {"testable": False, "reason": "needs both long and short sessions at the point"}
    return out | {"difference": _brief(contrast(v, {"long": 1.0, "short": -1.0}, alpha=alpha, reps=reps, seed=seed)), "testable": True}


def crossing_split(sessions: pd.DataFrame, *, block: str | None = "cm", low: str = "CM0", alpha: float = ALPHA, reps: int = 2000, seed: int = SEED) -> dict:
    """Per point and arm: the gain over `low` on items before vs from the reference W crossing (descriptive), and the
    per-session difference post − pre."""
    rows = _rows(sessions, block, "long", None)
    if not len(rows) or "w_crossing_item" not in rows:
        return {}
    out: dict = {}
    for arm in sorted(set(rows["arm"]) - {low}):
        pre = long_values(session_values(rows, [arm, low], positions="pre"), {arm: 1.0, low: -1.0})
        post = long_values(session_values(rows, [arm, low], positions="post"), {arm: 1.0, low: -1.0})
        if not len(pre) or not len(post):
            continue
        both = post.merge(pre, on=["point", "session"], suffixes=("_post", "_pre"))
        diff = both.assign(value=both["value_post"] - both["value_pre"])[["point", "session", "value"]]
        for c in sorted(both["point"].unique()):
            out.setdefault(c, {})[arm] = {
                "gain_pre": _brief(contrast(pre[pre["point"] == c], {c: 1.0}, alpha=alpha, reps=reps, seed=seed, flip=False)),
                "gain_post": _brief(contrast(post[post["point"] == c], {c: 1.0}, alpha=alpha, reps=reps, seed=seed, flip=False)),
                "post_minus_pre": _brief(contrast(diff[diff["point"] == c], {c: 1.0}, alpha=alpha, reps=reps, seed=seed, flip=False)),
            }
    return out


# ---------- G-H3 ----------


def _cost_pseudo(sessions: pd.DataFrame, a: str, b: str, meter: str, points: Sequence[str], block: str | None) -> tuple[pd.DataFrame, dict, pd.DataFrame | None]:
    """log(CPS_a / CPS_b) per point (CPS = Σ meter / Σ items solved over the point's sessions and epochs), linearised
    per session: z = LR_c + (c_a/c̄_a − k_a/k̄_a) − (c_b/c̄_b − k_b/k̄_b). Also the per-session sums (for the bootstrap)."""
    empty = pd.DataFrame(columns=["point", "session", "value"])
    df = _rows(sessions, block, None, points)
    if not len(df) or meter not in df:
        return empty, {"reason": f"no {meter} recorded"}, None
    df = df[df["arm"].isin([a, b])]
    agg = df.groupby(["point", "session", "arm"]).agg(cost=(meter, "sum"), solved=("items_solved", "sum"), nan=(meter, lambda s: s.isna().any())).reset_index()
    if agg["nan"].any():
        return empty, {"reason": f"{meter} missing for some sessions"}, None
    wide = agg.pivot_table(index=["point", "session"], columns="arm", values=["cost", "solved"]).dropna()
    rows, info = [], {}
    for c in points:
        if c not in wide.index.get_level_values("point"):
            continue
        g = wide.xs(c, level="point")
        cb_a, kb_a, cb_b, kb_b = g[("cost", a)].mean(), g[("solved", a)].mean(), g[("cost", b)].mean(), g[("solved", b)].mean()
        if min(kb_a, kb_b) <= 0 or min(cb_a, cb_b) <= 0:
            info[c] = {"ratio": None, "reason": "no items solved or zero cost"}
            continue
        lr = math.log((cb_a / kb_a) / (cb_b / kb_b))
        z = lr + (g[("cost", a)] / cb_a - g[("solved", a)] / kb_a) - (g[("cost", b)] / cb_b - g[("solved", b)] / kb_b)
        info[c] = {"ratio": math.exp(lr), "cps_" + a: cb_a / kb_a, "cps_" + b: cb_b / kb_b, "n": len(g)}
        rows.append(pd.DataFrame({"point": c, "session": g.index, "value": z.to_numpy()}))
    return (pd.concat(rows, ignore_index=True) if rows else empty), info, wide


def _cost_boot(wide: pd.DataFrame, a: str, b: str, weights: Mapping[str, float], reps: int, seed: int) -> np.ndarray:
    """Session-bootstrap replicates of the pooled log(CPS_a / CPS_b): worlds resampled (globally when a world appears
    at several points, else within point) and Σ cost / Σ solved recomputed per arm and point."""
    pts = list(weights)
    sub = wide[wide.index.get_level_values("point").isin(pts)]
    worlds = sorted(set(sub.index.get_level_values("session")))
    G, C = len(worlds), len(pts)
    X = np.full((4, G, C), np.nan)
    wi, ci = {w: i for i, w in enumerate(worlds)}, {c: j for j, c in enumerate(pts)}
    for (c, w), r in sub.iterrows():
        X[:, wi[w], ci[c]] = [r[("cost", a)], r[("solved", a)], r[("cost", b)], r[("solved", b)]]
    M = ~np.isnan(X[0])
    Xz = np.where(M, X, 0.0)
    rng = np.random.default_rng(seed)
    if (M.sum(1) > 1).any():
        W = rng.multinomial(G, np.full(G, 1 / G), size=reps).astype(float)
        S = np.einsum("rg,kgc->krc", W, Xz)
    else:
        S = np.zeros((4, reps, C))
        for j in range(C):
            idx = np.flatnonzero(M[:, j])
            Wj = rng.multinomial(len(idx), np.full(len(idx), 1 / len(idx)), size=reps).astype(float)
            S[:, :, j] = np.einsum("rg,kg->kr", Wj, Xz[:, idx, j])
    with np.errstate(divide="ignore", invalid="ignore"):
        lr = np.log((S[0] / S[1]) / (S[2] / S[3]))
    out = (lr * np.array([weights[c] for c in pts])).sum(1)
    return out[np.isfinite(out)]


def gh3(sessions: pd.DataFrame, *, block: str | None = "topo", points: Sequence[str] | None = None, outcome: str = "item_success", scale: str = "prob", iso: float = ISOLATION_SHARE, recovery: float = RECOVERY_SHARE, cost_ratio: float = COST_RATIO, meter: str = COST_METER, meters: Sequence[str] = COST_METERS, alpha: float = ALPHA, reps: int = REPS, seed: int = SEED, boot: int = 2000, flip: bool = True, primary: str = "t") -> dict:
    """G-H3 at the points where S1, M1, M2 and S-CM* all ran (equal point weights): the gatekeeper M2 − S1 > 0, H3a
    (isolation share ≥ `iso` as (M1 − S1) − iso (M2 − S1) > 0) and H3b (recovery ≥ `recovery` and cost per solved item
    ratio ≤ `cost_ratio`, an intersection-union), in fixed sequence; the shares with ratio intervals; every cost meter."""
    wide = session_values(sessions, list(TOPO_ARMS), block=block, length=None, outcome=outcome, scale=scale)
    if len(wide):
        ok = wide.dropna()
        counts = ok.groupby("point").size().to_dict()
    else:
        ok, counts = wide, {}
    pts = [c for c in (points or sorted(counts)) if counts.get(c, 0) >= MIN_SESSIONS]
    base = {"points": pts, "sessions": {c: int(counts.get(c, 0)) for c in (points or counts)}}
    if not pts:
        return base | {"testable": False, "reason": f"no point has ≥ {MIN_SESSIONS} sessions with all of {TOPO_ARMS}"}
    ok = ok[ok["point"].isin(pts)]
    w = {c: 1.0 / len(pts) for c in pts}
    kw = {"alpha": alpha, "reps": reps, "seed": seed, "flip": flip, "primary": primary}
    dM2 = long_values(ok, {"M2": 1.0, "S1": -1.0})
    pre = contrast(dM2, w, **kw)
    h3a = contrast(long_values(ok, {"M1": 1.0, "M2": -iso, "S1": iso - 1.0}), w, **kw)  # (M1 − S1) − iso (M2 − S1)
    rec = contrast(long_values(ok, {"S-CM*": 1.0, "M2": -recovery, "S1": recovery - 1.0}), w, **kw)  # (S − S1) − r (M2 − S1)
    iso_share = ratio(long_values(ok, {"M1": 1.0, "S1": -1.0}), dM2, w, alpha=alpha, boot=boot, seed=seed)
    spec_share = ratio(long_values(ok, {"M2": 1.0, "M1": -1.0}), dM2, w, alpha=alpha, boot=boot, seed=seed)
    rec_share = ratio(long_values(ok, {"S-CM*": 1.0, "S1": -1.0}), dM2, w, alpha=alpha, boot=boot, seed=seed)
    costs = {}
    for m in dict.fromkeys([meter, *meters]):
        v, info, sums = _cost_pseudo(sessions, "S-CM*", "M2", m, pts, block)
        if len(v) and v["point"].nunique() == len(pts):
            c = contrast(v, w, null=math.log(cost_ratio), alternative="less", **kw)
            c["ratio"] = None if c["est"] is None else math.exp(c["est"])
            c["ratio_ci"] = [None if b is None else math.exp(b) for b in c["ci"]]
            if boot:
                bs = _cost_boot(sums, "S-CM*", "M2", w, boot, seed)
                c["ratio_boot_ci"] = [float(np.exp(np.quantile(bs, alpha))), float(np.exp(np.quantile(bs, 1 - alpha)))] if len(bs) else [None, None]
            costs[m] = c | {"per_point": info}
        else:
            costs[m] = {"testable": False, "reject": False, "reason": info.get("reason") or "cost not computable at every point", "per_point": info}
    cost = costs.get(meter, {"testable": False, "reject": False})
    seq_pre = pre["reject"]
    seq_a = seq_pre and h3a["reject"]
    h3b_ok = rec["reject"] and cost.get("reject", False)
    return base | {
        "testable": pre["testable"],
        "pre": pre,
        "h3a": h3a | {"tested": seq_pre, "claim": bool(seq_a)},
        "h3b": {"recovery": rec, "cost": cost, "tested": bool(seq_a), "claim": bool(seq_a and h3b_ok), "meter": meter},
        "isolation_share": iso_share,
        "specialization_share": spec_share,
        "recovery_share": rec_share,
        "costs": costs,
        "thresholds": {"isolation": iso, "recovery": recovery, "cost_ratio": cost_ratio},
    }


# ---------- descriptive ----------


def cost_table(sessions: pd.DataFrame, meters: Sequence[str] = COST_METERS, by: Sequence[str] = ("block", "point", "arm")) -> list[dict]:
    """Cost per solved item per meter (Σ meter / Σ items solved), with sessions, items solved and probe tokens apart."""
    if sessions is None or not len(sessions):
        return []
    out = []
    for key, g in sessions.groupby(list(by), dropna=False):
        row = dict(zip(by, key if isinstance(key, tuple) else (key,), strict=True))
        solved = float(g["items_solved"].sum())
        row |= {"session_epochs": len(g), "sessions": g["session"].nunique(), "items": float(g["n_items"].sum()), "solved": solved}
        for m in meters:
            if m in g and g[m].notna().all():
                row[f"{m}_total"] = float(g[m].sum())
                row[f"{m}_per_solved"] = float(g[m].sum() / solved) if solved > 0 else None
            else:
                row[f"{m}_total"] = row[f"{m}_per_solved"] = None
        for extra in ("tokens_probe", "tokens_cm", "cost_usd_probe"):
            if extra in g:
                row[extra] = float(pd.to_numeric(g[extra], errors="coerce").sum())
        out.append(row)
    return out


def outcome_table(sessions: pd.DataFrame, by: Sequence[str] = ("block", "point", "arm", "N")) -> list[dict]:
    """Per group: item success (session-clustered mean and 95% t-interval over epoch-pooled sessions), dependency
    success, report exact, binary session success and pass^k (every epoch of the session succeeded), overflow rate,
    errors."""
    if sessions is None or not len(sessions):
        return []
    out = []
    for key, g in sessions.groupby(list(by), dropna=False):
        row = dict(zip(by, key if isinstance(key, tuple) else (key,), strict=True))
        per = g.groupby("session").agg(k=("items_solved", "sum"), n=("n_items", "sum"), ss=("session_success", "min"))
        rates = per["k"] / per["n"]
        m = float(rates.mean())
        if len(rates) > 1:
            se = float(rates.std(ddof=1) / math.sqrt(len(rates)))
            q = float(student_t.ppf(0.975, len(rates) - 1))
            ci = [m - q * se, m + q * se]
        else:
            ci = [None, None]
        row |= {
            "sessions": len(per),
            "session_epochs": len(g),
            "item_success": m,
            "item_success_ci": ci,
            "dependency_success": _mean(g, "dependency_success"),
            "report_exact": _mean(g, "report_exact"),
            "session_success": _mean(g, "session_success"),
            "pass_k": float(per["ss"].mean()) if per["ss"].notna().all() else None,
            "overflow_rate": _mean(g, "overflow"),
            "errors": int(g["error"].sum()) if "error" in g else 0,
        }
        out.append(row)
    return out


def _mean(g: pd.DataFrame, col: str) -> float | None:
    if col not in g:
        return None
    s = pd.to_numeric(g[col], errors="coerce").dropna()
    return float(s.mean()) if len(s) else None


def probe_table(sessions: pd.DataFrame, by: Sequence[str] = ("block", "point", "arm")) -> list[dict]:
    """Probe F1 per checkpoint: the mean over sessions of the epoch-mean F1 (a checkpoint not taken scores 0), its
    95% session-clustered t-interval, and the share of session-epochs that took it."""
    if sessions is None or not len(sessions) or "probes" not in sessions:
        return []
    rows = []
    for _, r in sessions.iterrows():
        for k, p in (r["probes"] or {}).items():
            rows.append({**{b: r[b] for b in by}, "session": r["session"], "k": int(k), "f1": float(p.get("f1", 0.0) or 0.0), "taken": bool(p.get("taken", False))})
    if not rows:
        return []
    df = pd.DataFrame(rows)
    out = []
    for key, g in df.groupby([*by, "k"], dropna=False):
        per = g.groupby("session")["f1"].mean()
        m = float(per.mean())
        ci = [None, None]
        if len(per) > 1:
            q = float(student_t.ppf(0.975, len(per) - 1))
            se = float(per.std(ddof=1) / math.sqrt(len(per)))
            ci = [m - q * se, m + q * se]
        out.append({**dict(zip([*by, "k"], key, strict=True)), "sessions": len(per), "f1": m, "f1_ci": ci, "coverage": float(g["taken"].mean())})
    return out


def probe_behaviour(sessions: pd.DataFrame, items: pd.DataFrame, by: Sequence[str] = ("block", "point", "arm")) -> list[dict]:
    """Audit §5.3's consistency check: across (session-epoch, checkpoint) pairs, the correlation of probe F1 at k with
    the success of the dependency items after k up to the next checkpoint. A weak correlation means the probes
    measure something the policy does not use."""
    if sessions is None or items is None or not len(sessions) or not len(items) or "probes" not in sessions:
        return []
    if {"dependency", "position", "plan_cell", "epoch"} - set(items.columns):
        return []
    dep = items[items["dependency"].astype(bool)]
    key = ["plan_cell", "arm", "session", "epoch"]
    dep_by = {k: g for k, g in dep.groupby(key)}
    pairs = []
    for _, r in sessions.iterrows():
        pr = r["probes"] or {}
        ks = sorted(int(k) for k in pr)
        g = dep_by.get(tuple(r[c] for c in key))
        if g is None:
            continue
        for i, k in enumerate(ks):
            nxt = ks[i + 1] if i + 1 < len(ks) else int(r["N"])
            sel = g[(g["position"] > k) & (g["position"] <= nxt)]
            if len(sel):
                pairs.append({**{b: r[b] for b in by}, "f1": float(pr[k].get("f1", 0.0) or 0.0) if k in pr else float(pr[str(k)].get("f1", 0.0) or 0.0), "dep": float(sel["success"].mean())})
    if not pairs:
        return []
    df = pd.DataFrame(pairs)
    out = []
    for key_, g in df.groupby(list(by), dropna=False):
        r = None
        if len(g) > 2 and g["f1"].std() > 0 and g["dep"].std() > 0:
            r = float(np.corrcoef(g["f1"], g["dep"])[0, 1])
        out.append({**dict(zip(by, key_ if isinstance(key_, tuple) else (key_,), strict=True)), "pairs": len(g), "r": r})
    return out


TAX_LABELS = ("forgot_constraint", "stale_state", "resurrected_done_item", "dropped_item", "hallucinated_state", "overflow")


def taxonomy_table(sessions: pd.DataFrame, by: Sequence[str] = ("block", "point", "arm")) -> list[dict]:
    """Failure labels (scorers.session.taxonomy) per group: counts, rate per item and share of failed items."""
    if sessions is None or not len(sessions):
        return []
    out = []
    for key, g in sessions.groupby(list(by), dropna=False):
        items = float(g["n_items"].sum())
        failed = float(items - g["items_solved"].sum())
        row = {**dict(zip(by, key if isinstance(key, tuple) else (key,), strict=True)), "items": items, "failed": failed}
        for lab in TAX_LABELS:
            col = f"tax_{lab}"
            c = float(pd.to_numeric(g[col], errors="coerce").sum()) if col in g else None
            row[lab] = c
            row[f"{lab}_per_item"] = None if c is None or not items else c / items
            row[f"{lab}_share_of_failed"] = None if c is None or not failed else c / failed
        out.append(row)
    return out


def degradation(items: pd.DataFrame, *, block: str | None = None, by: Sequence[str] = ("block", "point", "arm"), min_items: int = 30) -> list[dict]:
    """Audit §5.2, descriptive: per group, logistic regressions of item success on centred log(view tokens at the
    decision call), on relative position, and on both with their interaction, with session-clustered (sandwich)
    95% intervals. Items lost to overflow are left out (the harness, not degradation, failed them); items with no
    view tokens recorded are left out and counted."""
    import statsmodels.api as sm

    if items is None or not len(items):
        return []
    if missing := sorted({"success", "session", "overflow", "view_tokens", "rel_position", *by} - set(items.columns)):
        return [{"status": f"item table lacks {missing}"}]
    df = items if block is None or "block" not in items else items[items["block"] == block]
    out = []
    for key, g in df.groupby(list(by), dropna=False):
        row = dict(zip(by, key if isinstance(key, tuple) else (key,), strict=True))
        g = g[~g["overflow"].astype(bool)]
        vt = pd.to_numeric(g["view_tokens"], errors="coerce")
        no_view = int((vt.isna() | (vt <= 0)).sum())
        g = g[vt.notna() & (vt > 0)]
        row |= {"items": len(g), "items_without_view_tokens": no_view, "sessions": g["session"].nunique() if len(g) else 0}
        if len(g) < min_items or g["success"].nunique() < 2 or g["session"].nunique() < 2:
            out.append(row | {"status": "too few items or no variation"})
            continue
        lv = np.log(pd.to_numeric(g["view_tokens"]).to_numpy(dtype=float))
        pos = g["rel_position"].to_numpy(dtype=float)
        y = g["success"].to_numpy(dtype=float)
        groups = pd.factorize(g["session"])[0]
        lvc, posc = lv - lv.mean(), pos - pos.mean()
        fits = {"log_view": np.column_stack([lvc]), "position": np.column_stack([posc]), "joint": np.column_stack([lvc, posc, lvc * posc])}
        names = {"log_view": ["log_view"], "position": ["position"], "joint": ["log_view", "position", "log_view:position"]}
        row["status"] = "ok"
        for model, X in fits.items():
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = sm.GLM(y, sm.add_constant(X, has_constant="add"), family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": groups})
                params, bse = np.asarray(res.params), np.asarray(res.bse)
                if not np.all(np.isfinite(params)) or not np.all(np.isfinite(bse)):
                    raise ValueError("non-finite estimates (separation)")
                row[model] = {n: {"est": float(params[i + 1]), "ci": [float(params[i + 1] - 1.96 * bse[i + 1]), float(params[i + 1] + 1.96 * bse[i + 1])]} for i, n in enumerate(names[model])}
            except Exception as e:  # noqa: BLE001 - descriptive: report, never raise
                row[model] = {"error": f"{type(e).__name__}: {e}"}
        out.append(row)
    return out


def glmm(items: pd.DataFrame, capability, *, block: str | None = "topo", arms: Sequence[str] | None = None, reference: str | None = None, max_rows: int | None = 60_000, seed: int = SEED) -> dict:
    """D-029's descriptive mixed model: success ~ arm × capability (centred) + (1 | session) + (1 | item), by
    statsmodels' BinomialBayesMixedGLM (variational Bayes). Never raises: a failed or non-finite fit is reported."""
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

    x = capability_map(capability)
    if items is None or not len(items):
        return {"status": "no items"}
    if missing := sorted({"success", "session", "arm", "point"} - set(items.columns)):
        return {"status": "skipped", "reason": f"item table lacks {missing}"}
    df = items if block is None or "block" not in items else items[items["block"] == block]
    if arms is not None:
        df = df[df["arm"].isin(list(arms))]
    df = df[df["point"].isin(list(x))]
    if not len(df) or df["arm"].nunique() < 1 or df["point"].nunique() < 2:
        return {"status": "skipped", "reason": "needs ≥ 2 capability points with measured capability"}
    if max_rows and len(df) > max_rows:
        df = df.sample(max_rows, random_state=seed)
    ref = reference if reference in set(df["arm"]) else sorted(df["arm"].unique())[0]
    cap = df["point"].map(x).astype(float)
    item = df["item"] if "item" in df else df["session"].astype(str) + "#" + (df["position"].astype(str) if "position" in df else "")
    data = pd.DataFrame({"success": df["success"].astype(float).to_numpy(), "arm": df["arm"].astype(str).to_numpy(), "cap": (cap - cap.mean()).to_numpy(), "session": df["session"].astype(str).to_numpy(), "item": item.astype(str).to_numpy()})
    formula = f"success ~ C(arm, Treatment(reference='{ref}')) * cap" if data["arm"].nunique() > 1 else "success ~ cap"
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = BinomialBayesMixedGLM.from_formula(formula, {"session": "0 + C(session)", "item": "0 + C(item)"}, data)
            res = model.fit_vb()
        fe, fe_sd, vc, vc_sd = map(np.asarray, (res.fe_mean, res.fe_sd, res.vcp_mean, res.vcp_sd))
        finite = bool(np.all(np.isfinite(fe)) and np.all(np.isfinite(fe_sd)) and np.all(np.isfinite(vc)))
        return {
            "status": "ok" if finite else "non-finite",
            "formula": formula + " + (1 | session) + (1 | item)",
            "reference": ref,
            "rows": len(data),
            "fixed": {n: {"mean": float(m), "sd": float(s)} for n, m, s in zip(model.exog_names, fe, fe_sd, strict=True)},
            # vcp are log standard deviations
            "random_sd": {n: {"mean": float(np.exp(m)), "log_sd_sd": float(s)} for n, m, s in zip(model.vcp_names, vc, vc_sd, strict=True)},
            "warnings": sorted({str(w.message)[:160] for w in caught}),
            "capability_centre": float(cap.mean()),
        }
    except Exception as e:  # noqa: BLE001 - descriptive model: report the failure, never raise
        return {"status": "failed", "formula": formula, "reason": f"{type(e).__name__}: {e}"}
