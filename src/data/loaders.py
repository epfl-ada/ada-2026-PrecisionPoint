"""Readers for the raw Wikispeedia files (SNAP release).

All functions take ``raw_dir``, the folder that contains the extracted
``wikispeedia_paths-and-graph/`` and ``plaintext_articles/`` directories
(see ``src/scripts/download_data.sh``), and return tidy pandas / numpy
objects.

Conventions used everywhere in the project
-------------------------------------------
* Article names stay URL-encoded (``%C3%85land``) because that is how they
  appear in every raw file. They are the join keys. A decoded, human
  readable ``title`` column is added for display only.
* ``articles.tsv`` defines an integer id per article (its row number).
  The rows and columns of the shortest-path matrix follow the same order,
  so ids index the matrix directly.
"""
from __future__ import annotations

from pathlib import Path
from urllib.parse import unquote

import numpy as np
import pandas as pd

GRAPH_DIR = "wikispeedia_paths-and-graph"
TEXT_DIR = "plaintext_articles"

#: Sentinel used in the ``uint8`` distance matrix for "target unreachable".
UNREACHABLE = 255


def _read_tsv(path: Path, names: list[str]) -> pd.DataFrame:
    """Read one SNAP TSV file.

    The files start with a block of ``#`` comment lines, have no header row
    and must be read as raw strings: some article names (``NaN``-like words)
    would otherwise be parsed as missing values.
    """
    return pd.read_csv(
        path,
        sep="\t",
        comment="#",
        header=None,
        names=names,
        dtype=str,
        na_filter=False,
        encoding="utf-8",
    )


def decode_title(article: str) -> str:
    """``'%C3%85land'`` -> ``'Åland'``, ``'Pacific_Ocean'`` -> ``'Pacific Ocean'``."""
    return unquote(article).replace("_", " ")


def load_articles(raw_dir: str | Path) -> pd.DataFrame:
    """One row per article, indexed by the integer article id."""
    df = _read_tsv(Path(raw_dir) / GRAPH_DIR / "articles.tsv", ["article"])
    df["title"] = df["article"].map(decode_title)
    df.index.name = "article_id"
    return df


def load_categories(raw_dir: str | Path) -> pd.DataFrame:
    """Article -> category assignments, with the hierarchy split into levels.

    Categories look like ``subject.Science.Biology.Mammals``. An article can
    have several categories. ``level1`` is the broad topic (``Science``),
    ``level2`` the sub-topic (``Science.Biology``) and ``level3`` the finest
    available label (``Science.Biology.Mammals``, or ``level2`` when the
    category has only two levels).
    """
    df = _read_tsv(Path(raw_dir) / GRAPH_DIR / "categories.tsv", ["article", "category"])
    parts = df["category"].str.split(".")
    df["depth"] = parts.str.len() - 1  # the leading "subject" token is not a level
    df["level1"] = parts.str[1]
    df["level2"] = parts.str[1:3].str.join(".")
    df["level3"] = parts.str[1:4].str.join(".")
    return df


def load_links(raw_dir: str | Path) -> pd.DataFrame:
    """Directed hyperlinks ``source -> target`` between articles."""
    return _read_tsv(Path(raw_dir) / GRAPH_DIR / "links.tsv", ["source", "target"])


def load_distance_matrix(raw_dir: str | Path, n_articles: int) -> np.ndarray:
    """Shortest-path lengths (in clicks) between all pairs of articles.

    The raw file stores one row of single digits per source article, with
    ``_`` for unreachable targets. Returns an ``(n, n)`` ``uint8`` array in
    which ``dist[i, j]`` is the distance from article ``i`` to ``j`` and
    unreachable pairs hold :data:`UNREACHABLE`. ``uint8`` keeps the full
    4604 x 4604 matrix at about 21 MB.
    """
    path = Path(raw_dir) / GRAPH_DIR / "shortest-path-distance-matrix.txt"
    with open(path, encoding="ascii") as fh:
        rows = [ln.strip() for ln in fh if ln.strip() and not ln.startswith("#")]
    if len(rows) != n_articles or any(len(r) != n_articles for r in rows):
        raise ValueError("distance matrix does not match the article list")
    raw = np.frombuffer("".join(rows).encode("ascii"), dtype=np.uint8)
    dist = (raw - ord("0")).astype(np.uint8).reshape(n_articles, n_articles)
    dist[raw.reshape(n_articles, n_articles) == ord("_")] = UNREACHABLE
    return dist


def load_games(raw_dir: str | Path) -> pd.DataFrame:
    """Finished and unfinished games in one table, one row per game.

    Columns
    -------
    game_id, ip (hashed IP address), timestamp (UTC), duration (seconds),
    path (raw ``;``-separated string, ``<`` = back click), source, target,
    finished (bool), rating (1-5, NaN when not given),
    outcome (``finished``, ``timeout`` or ``restart``).
    """
    graph_dir = Path(raw_dir) / GRAPH_DIR
    fin = _read_tsv(graph_dir / "paths_finished.tsv",
                    ["ip", "timestamp", "duration", "path", "rating"])
    unf = _read_tsv(graph_dir / "paths_unfinished.tsv",
                    ["ip", "timestamp", "duration", "path", "target", "outcome"])

    fin["target"] = fin["path"].str.split(";").str[-1]
    fin["outcome"] = "finished"
    unf["rating"] = "NULL"

    games = pd.concat([fin, unf], ignore_index=True)
    games["finished"] = games["outcome"].eq("finished")
    games["source"] = games["path"].str.split(";").str[0]
    games["timestamp"] = pd.to_datetime(games["timestamp"].astype("int64"), unit="s", utc=True)
    games["duration"] = games["duration"].astype("int64")
    games["rating"] = pd.to_numeric(games["rating"].replace("NULL", np.nan))
    games.index.name = "game_id"
    cols = ["ip", "timestamp", "duration", "path", "source", "target",
            "finished", "outcome", "rating"]
    return games[cols].reset_index()


def load_plaintext(raw_dir: str | Path, articles: pd.Series) -> pd.Series:
    """Plain text of every article, indexed like ``articles``.

    The raw files start with a ``#copyright`` line, the title and a
    "2007 Schools Wikipedia Selection. Related subjects: ..." banner. The
    banner is kept because players see it too.
    """
    text_dir = Path(raw_dir) / TEXT_DIR
    texts = [(text_dir / f"{a}.txt").read_text(encoding="utf-8") for a in articles]
    return pd.Series(texts, index=articles.index, name="text")
