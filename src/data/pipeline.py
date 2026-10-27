"""End-to-end preprocessing: raw SNAP files -> cached analysis-ready data.

``build_processed`` runs every step once (about 3 minutes on a laptop, the
link-position search dominates) and writes the results to
``data/processed/``. ``load_processed`` reads them back in a few seconds.
The notebook calls ``load_processed(build_if_missing=True)``.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import choice_sets, features, loaders, paths


@dataclass
class Processed:
    """All analysis-ready objects of the project."""

    articles: pd.DataFrame      # one row per article: title, topic, degrees, text stats
    categories: pd.DataFrame    # article -> category (several per article possible)
    links: pd.DataFrame         # one row per link, with its position on the page
    games: pd.DataFrame         # one row per game, with summary statistics
    moves: pd.DataFrame         # one row per click or back click
    dist: np.ndarray            # shortest-path matrix (uint8, 255 = unreachable)
    similarity: np.ndarray      # TF-IDF cosine similarity matrix (float32)
    choices: choice_sets.ChoiceData
    dropped: pd.Series          # clicks removed while building the choice data


def _log(msg: str, t0: float) -> None:
    print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)


def build_processed(raw_dir: str | Path, out_dir: str | Path) -> None:
    """Run the full preprocessing pipeline and cache the results."""
    t0 = time.time()
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    articles = loaders.load_articles(raw_dir)
    categories = loaders.load_categories(raw_dir)
    links = loaders.load_links(raw_dir)
    dist = loaders.load_distance_matrix(raw_dir, len(articles))
    games = loaders.load_games(raw_dir)
    _log("raw files loaded", t0)

    ids = pd.Series(articles.index.to_numpy(), index=articles["article"])
    links["source_id"] = links["source"].map(ids)
    links["target_id"] = links["target"].map(ids)

    bodies = loaders.load_plaintext(raw_dir, articles["article"]).map(features.article_body)
    similarity = features.tfidf_similarity(bodies)
    _log("TF-IDF similarity computed", t0)
    links = features.link_positions(links, bodies, articles["title"])
    _log("link positions computed", t0)

    articles = articles.join(features.article_text_stats(bodies, articles["title"]))
    articles["disease"] = features.has_medical_infobox(bodies)
    nonself = links[links["source_id"] != links["target_id"]]
    articles["out_degree"] = np.bincount(nonself["source_id"], minlength=len(articles))
    articles["in_degree"] = np.bincount(nonself["target_id"], minlength=len(articles))
    main_topic = (categories.drop_duplicates("article").set_index("article")["level1"])
    articles["topic"] = articles["article"].map(main_topic).fillna("None")
    articles["n_categories"] = articles["article"].map(categories.groupby("article").size()).fillna(0).astype(int)

    moves = paths.extract_moves(games)
    games = paths.game_summary(games, moves, dist, ids, loaders.UNREACHABLE)
    _log("games and moves extracted", t0)

    choices, dropped = choice_sets.build_choice_data(moves, games, articles, links, dist, similarity)
    _log(f"choice data built: {choices.n_clicks:,} clicks, {choices.n_rows:,} rows", t0)

    articles.to_parquet(out_dir / "articles.parquet")
    categories.to_parquet(out_dir / "categories.parquet")
    links.to_parquet(out_dir / "links.parquet")
    games.to_parquet(out_dir / "games.parquet")
    moves.to_parquet(out_dir / "moves.parquet")
    np.save(out_dir / "distance.npy", dist)
    np.save(out_dir / "similarity.npy", similarity)
    np.save(out_dir / "choice_X.npy", choices.X)
    np.savez(out_dir / "choice_index.npz", offsets=choices.offsets, alt=choices.alt,
             chosen=choices.chosen)
    choices.clicks.to_parquet(out_dir / "choice_clicks.parquet")
    meta = {"feature_names": choices.feature_names, "scaling": choices.scaling,
            "dropped": dropped.to_dict()}
    (out_dir / "choice_meta.json").write_text(json.dumps(meta, indent=2))
    _log(f"saved to {out_dir}", t0)


def load_processed(raw_dir: str | Path = "data/raw", out_dir: str | Path = "data/processed",
                   build_if_missing: bool = True, mmap: bool = False) -> Processed:
    """Load the cached data, building it first if needed.

    ``mmap=True`` memory-maps the 0.9 GB feature matrix instead of reading it.
    """
    out_dir = Path(out_dir)
    if not (out_dir / "choice_meta.json").exists():
        if not build_if_missing:
            raise FileNotFoundError(f"no processed data in {out_dir}")
        build_processed(raw_dir, out_dir)
    meta = json.loads((out_dir / "choice_meta.json").read_text())
    index = np.load(out_dir / "choice_index.npz")
    choices = choice_sets.ChoiceData(
        X=np.load(out_dir / "choice_X.npy", mmap_mode="r" if mmap else None),
        feature_names=meta["feature_names"], offsets=index["offsets"], alt=index["alt"],
        chosen=index["chosen"], clicks=pd.read_parquet(out_dir / "choice_clicks.parquet"),
        scaling={k: tuple(v) for k, v in meta["scaling"].items()},
    )
    return Processed(
        articles=pd.read_parquet(out_dir / "articles.parquet"),
        categories=pd.read_parquet(out_dir / "categories.parquet"),
        links=pd.read_parquet(out_dir / "links.parquet"),
        games=pd.read_parquet(out_dir / "games.parquet"),
        moves=pd.read_parquet(out_dir / "moves.parquet"),
        dist=np.load(out_dir / "distance.npy"),
        similarity=np.load(out_dir / "similarity.npy"),
        choices=choices,
        dropped=pd.Series(meta["dropped"], name="clicks"),
    )
