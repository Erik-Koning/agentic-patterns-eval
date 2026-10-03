"""A synthetic main study as the loader's tidy frame, for the main-analysis tests (no logs, no API).

Every (tier, cell, arm) of the run plan's Studies A, B and F at a reduced size, from `main_power.simulate` (so the
model is the power simulation's), with four distinct cost meters, the plan cells of `planned_sizes`, answer keys for
every arm, and Study C's two extra epochs on the first tasks of each cell.
"""

import numpy as np
import pandas as pd

from ape.analysis import main_power as mp

# Per-sample $ and wall-clock relative to tokens, per arm: the meters disagree, so rankings can flip (H3).
USD_PER_TOKEN = {"S1": 1.0, "S5": 1.4, "S3s": 1.2, "S7": 1.0, "S9": 1.0, "M1": 0.7, "M1s": 0.7, "M1k": 0.9, "M2": 0.9, "M7": 0.8, "S8k3": 1.0}
WALL_PER_TOKEN = {"S1": 1.0, "S5": 1.1, "S3s": 1.1, "S7": 1.0, "S9": 1.2, "M1": 0.45, "M1s": 1.0, "M1k": 0.5, "M2": 0.5, "M7": 0.5, "S8k3": 0.4}


def synthetic_frame(seed: int = 0, n_tasks: int = 27, worlds: int = 9, effects: dict | None = None, c_tasks: int = 6, drop_arms: tuple[str, ...] = ()) -> pd.DataFrame:
    """The tidy frame of one synthetic main study. `effects` {arm: success offset over S1} (default: modest gains)."""
    sizes = mp.planned_sizes()
    eff = {"S1": 0.0, "S9": 0.03, "M1": 0.10, "M1s": 0.10, "M7": 0.06, "S3s": 0.04, "S5": 0.12, "S7": -0.05, "M1k": 0.14, "M2": 0.14} | (effects or {})
    settings = mp.Settings(worlds=worlds)
    spec, n, ep = {}, {}, {}
    for (tier, cell, arm), sz in sizes.items():
        if arm in drop_arms:
            continue
        base = mp._base(tier, cell, mp.BASELINES, settings)
        spec.setdefault((tier, cell), {})[arm] = float(np.clip(base + eff.get(arm, 0.0), 0.05, 0.95))
        n[(tier, cell)] = n_tasks
        ep[(tier, cell, arm)] = sz["epochs"]
    rng = np.random.default_rng(seed)
    draw = mp.simulate(spec, n, ep, mp.Sigmas(), settings, rng)
    df = draw.frame()
    df["plan_cell"] = [sizes[(t, c, a)]["plan_cell"] for t, c, a in zip(df["tier"], df["cell"], df["arm"], strict=True)]
    df = _meters(df, rng)
    # answer keys for non-S1 arms: right, or a wrong answer of the run's own
    miss = df["answer_key"].isna() & (df["arm"] != "S1")
    df.loc[miss, "answer_key"] = np.where(df.loc[miss, "success"] > 0, df.loc[miss, "task"] + ":0", df.loc[miss, "task"] + ":w" + df.loc[miss, "epoch"].astype(str))
    df = pd.concat([df, _study_c(df, c_tasks, rng, drop_arms)], ignore_index=True)
    return df.assign(family=df["cell"].str.split("-").str[0], level=df["cell"].str.split("-").str[1])


def _meters(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    tok = df["tokens"].to_numpy() * 10_000
    usd = tok * df["arm"].map(USD_PER_TOKEN).fillna(1.0).to_numpy() * 1e-6
    wall = tok * df["arm"].map(WALL_PER_TOKEN).fillna(1.0).to_numpy() * 1e-3 * rng.lognormal(0, 0.1, len(df))
    calls = np.maximum(1, np.round(tok / 2500))
    return df.assign(tokens=tok, usd=usd, cost_usd=usd, wall=wall, working_time=wall, total_time=wall * 1.2, calls=calls)


def _study_c(df: pd.DataFrame, c_tasks: int, rng: np.random.Generator, drop_arms) -> pd.DataFrame:
    """Two extra epochs on the first `c_tasks` tasks of each Luna Study A/B cell for Study C's arms, and the live S8k3."""
    out = []
    plans = {"main.C.a": (("S1", "S8k3", "M1", "M7"), ("F1-2", "F1-32", "F2-2", "F2-10")), "main.C.b": (("S1", "S5", "S8k3", "M1", "M2"), ("F3-5", "F3-60", "F7-10", "F7-1000"))}
    luna = df[df["tier"] == "luna"]
    for plan, (arms, cells) in plans.items():
        for cell in cells:
            tasks = sorted(luna.loc[luna["cell"] == cell, "task"].unique())[:c_tasks]
            for arm in arms:
                if arm in drop_arms:
                    continue
                src = luna[(luna["cell"] == cell) & (luna["arm"] == ("S1" if arm == "S8k3" else arm)) & luna["task"].isin(tasks) & (luna["epoch"] <= 2)]
                for _, r in src.iterrows():
                    rr = r.copy()
                    rr["plan_cell"], rr["arm"] = plan, arm
                    p = 0.85 if r["success"] > 0 else 0.15
                    rr["success"] = float(rng.random() < p)
                    rr["answer_key"] = f"{r['task']}:0" if rr["success"] else f"{r['task']}:c{r['epoch']}"
                    out.append(rr)
    return pd.DataFrame(out)
