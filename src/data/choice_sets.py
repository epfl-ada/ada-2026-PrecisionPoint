"""Long-format choice data for the conditional logit model.

Every forward click is treated as a discrete choice: a player standing on
page ``u`` with target ``t`` picks one of the outgoing links of ``u``. The
data therefore has one row per (click, candidate link) pair, about 23
million rows for the 366k clicks. Rows of the same click are contiguous and
``offsets[k]:offsets[k + 1]`` delimits the choice set of click ``k``, the
same layout as a CSR sparse matrix. Everything is stored in numpy arrays
(``float32`` features) so the full data fits in about 1 GB of memory.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .loaders import UNREACHABLE
from .paths import visited_before

#: Features of a candidate link ``j`` for a click from ``u`` towards target ``t``.
FEATURE_DESCRIPTIONS = {
    "closer": "j is one click closer to the target than u (shortest-path distance)",
    "farther": "j is farther from the target than u (including when t cannot be reached from j)",
    "is_target": "j is the target itself",
    "dead_end": "the target cannot be reached from j (descriptive only, part of farther)",
    "similarity": "TF-IDF cosine similarity between j and the target (z-score)",
    "position": "relative position of the first mention of j on page u (0 = top)",
    "position_missing": "the link anchor of j was not found in the text of u",
    "log_mentions": "log(1 + number of mentions of j on page u)",
    "log_indegree": "log(1 + number of articles linking to j) (z-score)",
    "visited": "j was already visited earlier in the same game",
}


@dataclass
class ChoiceData:
    """Choice sets of many clicks stored in CSR layout.

    Attributes
    ----------
    X : (n_rows, n_features) float32 feature matrix.
    feature_names : names of the columns of ``X``.
    offsets : (n_clicks + 1,) row offsets, choice set ``k`` is
        ``offsets[k]:offsets[k + 1]``.
    alt : (n_rows,) article id of each candidate link.
    chosen : (n_clicks,) row index of the link that was actually clicked.
    clicks : click-level metadata, row ``k`` describes choice set ``k``.
    scaling : mean and standard deviation used to standardise features.
    """

    X: np.ndarray
    feature_names: list[str]
    offsets: np.ndarray
    alt: np.ndarray
    chosen: np.ndarray
    clicks: pd.DataFrame
    scaling: dict = field(default_factory=dict)

    @property
    def n_clicks(self) -> int:
        return len(self.offsets) - 1

    @property
    def n_rows(self) -> int:
        return len(self.alt)

    @property
    def sizes(self) -> np.ndarray:
        return np.diff(self.offsets)

    def row_click(self) -> np.ndarray:
        """Index of the click (choice set) each row belongs to."""
        return np.repeat(np.arange(self.n_clicks), self.sizes)

    def columns(self, names: list[str]) -> np.ndarray:
        """Column indices of the requested features."""
        return np.array([self.feature_names.index(n) for n in names])

    def chosen_mask(self) -> np.ndarray:
        """Boolean vector over rows, True for clicked links."""
        mask = np.zeros(self.n_rows, dtype=bool)
        mask[self.chosen] = True
        return mask

    def subset(self, click_mask: np.ndarray) -> "ChoiceData":
        """Keep only the clicks selected by a boolean mask (e.g. a train split)."""
        keep = np.flatnonzero(click_mask)
        sizes = self.sizes[keep]
        offsets = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
        start = np.repeat(self.offsets[keep], sizes)
        rows = start + np.arange(offsets[-1]) - np.repeat(offsets[:-1], sizes)
        chosen = offsets[:-1] + (self.chosen[keep] - self.offsets[keep])
        return ChoiceData(self.X[rows], list(self.feature_names), offsets, self.alt[rows],
                          chosen, self.clicks.iloc[keep].reset_index(drop=True),
                          dict(self.scaling))


def _adjacency(links: pd.DataFrame, n_articles: int) -> tuple[np.ndarray, pd.DataFrame]:
    """CSR adjacency of the link graph without self-links, sorted by (source, target)."""
    lk = (links[links["source_id"] != links["target_id"]]
          .sort_values(["source_id", "target_id"]).reset_index(drop=True))
    counts = np.bincount(lk["source_id"], minlength=n_articles)
    ptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    return ptr, lk


def _zscore(x: np.ndarray) -> tuple[np.ndarray, float, float]:
    mu, sd = float(x.mean()), float(x.std())
    return ((x - mu) / sd).astype(np.float32), mu, sd


def build_choice_data(moves: pd.DataFrame, games: pd.DataFrame, articles: pd.DataFrame,
                      links: pd.DataFrame, dist: np.ndarray, similarity: np.ndarray
                      ) -> tuple[ChoiceData, pd.Series]:
    """Assemble the choice data of all modelable forward clicks.

    Parameters
    ----------
    moves : output of :func:`src.data.paths.extract_moves`.
    games : output of :func:`src.data.loaders.load_games`.
    articles : output of :func:`src.data.loaders.load_articles`.
    links : links with integer ``source_id``/``target_id`` and the position
        columns of :func:`src.data.features.link_positions`.
    dist : shortest-path matrix, ``dist[i, j]`` from ``i`` to ``j``.
    similarity : article-by-article similarity matrix.

    Returns
    -------
    data : the :class:`ChoiceData`.
    dropped : number of clicks removed at each cleaning step, for reporting.
    """
    n = len(articles)
    ids = pd.Series(articles.index.to_numpy(), index=articles["article"])
    ptr, adj = _adjacency(links, n)
    adj_target = adj["target_id"].to_numpy(np.int32)
    adj_key = adj["source_id"].to_numpy(np.int64) * n + adj["target_id"].to_numpy(np.int64)

    # ---- 1. clicks that can be modelled -------------------------------------------
    fw = moves.loc[~moves["is_back"]].copy()
    dropped = {"forward clicks": len(fw)}
    fw["u"], fw["v"], fw["t"] = (fw["from_page"].map(ids), fw["to_page"].map(ids),
                                 fw["target"].map(ids))
    ok = fw[["u", "v", "t"]].notna().all(axis=1)
    dropped["target not in article list"] = int((~ok).sum())
    fw = fw[ok].astype({"u": np.int64, "v": np.int64, "t": np.int64})

    key = fw["u"].to_numpy() * n + fw["v"].to_numpy()
    pos = np.searchsorted(adj_key, key)
    in_graph = (pos < len(adj_key)) & (adj_key[np.minimum(pos, len(adj_key) - 1)] == key)
    dropped["clicked link missing from links.tsv"] = int((~in_graph).sum())
    fw, pos = fw[in_graph], pos[in_graph]

    du = dist[fw["u"].to_numpy(), fw["t"].to_numpy()]
    reachable = du != UNREACHABLE
    dropped["target unreachable from page"] = int((~reachable).sum())
    fw, pos, du = fw[reachable], pos[reachable], du[reachable]
    dropped["modelled clicks"] = len(fw)

    fw = fw.join(games.set_index("game_id")["ip"], on="game_id")
    u, t = fw["u"].to_numpy(), fw["t"].to_numpy()
    clicks = pd.DataFrame({
        "game_id": fw["game_id"].to_numpy(), "ip": fw["ip"].to_numpy(),
        "step": fw["step"].to_numpy(), "click_no": fw["click_no"].to_numpy(),
        "after_back": fw["after_back"].to_numpy(), "finished": fw["finished"].to_numpy(),
        "u": u, "v": fw["v"].to_numpy(), "t": t, "du": du.astype(np.int16),
        "dv": dist[fw["v"].to_numpy(), t].astype(np.int16),
    })
    clicks["delta"] = np.where(clicks["dv"] == UNREACHABLE, np.nan, clicks["dv"] - clicks["du"])

    # ---- 2. expand every click into its choice set ---------------------------------
    sizes = ptr[u + 1] - ptr[u]
    offsets = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
    row_click = np.repeat(np.arange(len(clicks)), sizes)
    adj_row = np.repeat(ptr[u], sizes) + (np.arange(offsets[-1]) - np.repeat(offsets[:-1], sizes))
    alt = adj_target[adj_row]
    chosen = offsets[:-1] + (pos - ptr[u])
    del sizes

    # ---- 3. features -----------------------------------------------------------------
    t_rows = t[row_click]
    d_alt = dist[alt, t_rows].astype(np.int16)
    du_rows = clicks["du"].to_numpy()[row_click]
    feats = {
        "closer": d_alt == du_rows - 1,
        "farther": d_alt > du_rows,  # UNREACHABLE (255) counts as farther
        "is_target": alt == t_rows,
        "dead_end": d_alt == UNREACHABLE,
    }
    del d_alt, du_rows
    # The target is perfectly similar to itself (cosine 1, about 20 standard
    # deviations above the mean), which would let `similarity` absorb the
    # pull of the target link. Its similarity is therefore set to the mean
    # and `is_target` alone captures that pull.
    sim = similarity[alt, t_rows]
    sim[feats["is_target"]] = np.nan
    scaling = {"similarity": (float(np.nanmean(sim)), float(np.nanstd(sim)))}
    mu, sd = scaling["similarity"]
    feats["similarity"] = np.nan_to_num((sim - mu) / sd, nan=0.0).astype(np.float32)
    del t_rows, sim

    rel_pos = adj["rel_pos"].to_numpy(np.float32)[adj_row]
    feats["position_missing"] = np.isnan(rel_pos)
    feats["position"] = np.nan_to_num(rel_pos, nan=0.0)
    feats["log_mentions"] = np.log1p(adj["n_mentions"].to_numpy(np.float32)[adj_row])
    del rel_pos, adj_row

    indegree = np.bincount(adj["target_id"], minlength=n)
    feats["log_indegree"], mu, sd = _zscore(np.log1p(indegree)[alt])
    scaling["log_indegree"] = (mu, sd)

    first = visited_before(moves, ids)
    vkey = first["game_id"].to_numpy(np.int64) * n + first["page_id"].to_numpy(np.int64)
    order = np.argsort(vkey)
    vkey, vstep = vkey[order], first["first_step"].to_numpy()[order]
    qkey = clicks["game_id"].to_numpy(np.int64)[row_click] * n + alt
    hit = np.minimum(np.searchsorted(vkey, qkey), len(vkey) - 1)
    feats["visited"] = (vkey[hit] == qkey) & (vstep[hit] < clicks["step"].to_numpy()[row_click])
    del qkey, hit, row_click

    names = list(FEATURE_DESCRIPTIONS)
    X = np.empty((len(alt), len(names)), dtype=np.float32)
    for j, name in enumerate(names):
        X[:, j] = feats.pop(name)
    data = ChoiceData(X, names, offsets, alt, chosen.astype(np.int64), clicks, scaling)
    return data, pd.Series(dropped, name="clicks")
