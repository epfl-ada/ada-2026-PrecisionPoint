"""Shared figure style and small plotting helpers.

One quiet style for every figure of the project: light surface, hairline
solid gridlines, thin marks, a fixed categorical colour order (never
cycled) and a blue/red diverging pair for "less / more tempting than
expected", with a neutral grey at "as expected".
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

#: Categorical slots, always used in this order.
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
               "#e87ba4", "#008300", "#6250d6", "#e34948"]
BLUE, ORANGE, AQUA = CATEGORICAL[:3]

#: Diverging poles: avoided (below expectation) vs tempting (above).
AVOIDED = "#2a78d6"
TEMPTING = "#e34948"
NEUTRAL = "#c3c2b7"


def set_style() -> None:
    """Apply the project style to matplotlib (call once per notebook)."""
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "figure.dpi": 110,
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.labelsize": 10,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
        "axes.axisbelow": True, "axes.prop_cycle": mpl.cycler(color=CATEGORICAL),
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2, "text.color": INK, "lines.linewidth": 2,
        "lines.markersize": 5, "legend.frameon": False, "legend.fontsize": 9,
        "patch.linewidth": 0,
    })


def ccdf(ax: plt.Axes, values: np.ndarray, label: str | None = None, **kw) -> None:
    """Complementary cumulative distribution P(X >= x) on log-log axes."""
    x = np.sort(np.asarray(values)[np.asarray(values) > 0])
    y = 1.0 - np.arange(len(x)) / len(x)
    ax.loglog(x, y, label=label, **kw)
    ax.set_ylabel("share of items with value ≥ x")


def bar_with_ci(ax: plt.Axes, labels, est, low, high, color: str = BLUE,
                horizontal: bool = True) -> None:
    """Point estimates with confidence intervals as thin bars plus whiskers."""
    pos = np.arange(len(labels))
    err = np.vstack([np.asarray(est) - np.asarray(low), np.asarray(high) - np.asarray(est)])
    if horizontal:
        ax.barh(pos, est, height=0.6, color=color)
        ax.errorbar(est, pos, xerr=err, fmt="none", ecolor=INK_2, elinewidth=1, capsize=2)
        ax.set_yticks(pos, labels)
        ax.grid(axis="y", visible=False)
    else:
        ax.bar(pos, est, width=0.6, color=color)
        ax.errorbar(pos, est, yerr=err, fmt="none", ecolor=INK_2, elinewidth=1, capsize=2)
        ax.set_xticks(pos, labels)
        ax.grid(axis="x", visible=False)


def diverging_colors(log_values: np.ndarray) -> list[str]:
    """Red for values above 0 (more clicks than expected), blue below."""
    return [TEMPTING if v > 0 else AVOIDED for v in np.asarray(log_values)]


def note(ax: plt.Axes, text: str) -> None:
    """Small source / caveat line under a plot."""
    ax.annotate(text, xy=(0, -0.2), xycoords="axes fraction", fontsize=8,
                color=MUTED, ha="left", va="top")
