"""Helpers for the descriptive part of the notebook."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..data.choice_sets import ChoiceData
from ..data.loaders import GRAPH_DIR, TEXT_DIR


def file_inventory(raw_dir: str | Path) -> pd.DataFrame:
    """Size and number of data lines of every raw file."""
    raw_dir = Path(raw_dir)
    rows = []
    for f in sorted((raw_dir / GRAPH_DIR).iterdir()):
        with open(f, encoding="utf-8") as fh:
            n = sum(1 for ln in fh if ln.strip() and not ln.startswith("#"))
        rows.append({"file": f.name, "data lines": n, "size (MB)": f.stat().st_size / 1e6})
    texts = list((raw_dir / TEXT_DIR).glob("*.txt"))
    rows.append({"file": f"{TEXT_DIR}/ (one .txt per article)", "data lines": len(texts),
                 "size (MB)": sum(t.stat().st_size for t in texts) / 1e6})
    return pd.DataFrame(rows).set_index("file")


def click_through(data: ChoiceData, values: np.ndarray, bins: int | np.ndarray = 10,
                  row_mask: np.ndarray | None = None) -> pd.DataFrame:
    """Model-free click-through rate of candidate links, by bins of a feature.

    The click-through rate of a bin is (links of the bin that were clicked)
    / (times a link of the bin was offered). Bins are quantiles of the
    feature when ``bins`` is an integer.
    """
    chosen = data.chosen_mask()
    mask = np.ones(data.n_rows, dtype=bool) if row_mask is None else row_mask
    v, c = values[mask], chosen[mask]
    if np.isscalar(bins):
        edges = np.unique(np.quantile(v, np.linspace(0, 1, int(bins) + 1)))
    else:
        edges = np.asarray(bins)
    idx = np.clip(np.searchsorted(edges, v, side="right") - 1, 0, len(edges) - 2)
    df = pd.DataFrame({"bin": idx, "value": v, "clicked": c})
    out = df.groupby("bin").agg(value=("value", "mean"), offered=("clicked", "size"),
                                clicks=("clicked", "sum"))
    out["ctr"] = out["clicks"] / out["offered"]
    out["ctr_se"] = np.sqrt(out["ctr"] * (1 - out["ctr"]) / out["offered"])
    return out


def click_quality(clicks: pd.DataFrame, by: str | None = None) -> pd.DataFrame:
    """Share of clicks that bring the player closer to, as far from, or farther
    from the target (shortest-path distance), optionally broken down by ``by``."""
    labels = pd.cut(clicks["delta"], [-1.5, -0.5, 0.5, np.inf],
                    labels=["closer", "same distance", "farther"])
    if by is None:
        return labels.value_counts(normalize=True).sort_index().to_frame("share")
    return pd.crosstab(clicks[by], labels, normalize="index")


def group_mean_ci(df: pd.DataFrame, group: str, value: str, n_boot: int = 1000,
                  min_size: int = 1, seed: int = 0) -> pd.DataFrame:
    """Mean of ``value`` per group with a 95% bootstrap confidence interval."""
    rng = np.random.default_rng(seed)
    rows = []
    for name, g in df.groupby(group):
        v = g[value].to_numpy()
        if len(v) < min_size:
            continue
        boot = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
        rows.append({group: name, "n": len(v), "mean": v.mean(),
                     "ci_low": np.percentile(boot, 2.5), "ci_high": np.percentile(boot, 97.5)})
    return pd.DataFrame(rows).set_index(group).sort_values("mean")


def sample_rows(data: ChoiceData, n: int = 500_000, seed: int = 0) -> pd.DataFrame:
    """Random sample of candidate rows as a DataFrame (for correlations, plots)."""
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(data.n_rows, size=min(n, data.n_rows), replace=False))
    df = pd.DataFrame(np.asarray(data.X[idx]), columns=data.feature_names)
    df["clicked"] = data.chosen_mask()[idx]
    return df
