"""Turn raw game paths into navigation events.

A raw path is a ``;``-separated list of article names in which ``<`` stands
for a click on the browser's back button, e.g.::

    14th_century;Time;Science;Nature;Weather;Sunlight;<;Sun;Rainbow

Back clicks behave like a stack: ``A;B;C;<;<;D`` means the player went back
from C to B, then to A, and clicked D *from A*. Resolving the stack matters
for the choice model, because the page a player chooses from is the page
they are actually looking at, not the previous token in the string.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BACK = "<"


def resolve_path(path: str) -> list[tuple[str, str, bool]]:
    """Return the moves of one game as ``(from_page, to_page, is_back)``.

    Malformed paths with more back clicks than pages (they do not occur in
    the released data, but are cheap to guard against) stop at the first
    impossible back click.
    """
    tokens = path.split(";")
    stack = [tokens[0]]
    moves: list[tuple[str, str, bool]] = []
    for tok in tokens[1:]:
        if tok == BACK:
            if len(stack) < 2:
                break
            current = stack.pop()
            moves.append((current, stack[-1], True))
        else:
            moves.append((stack[-1], tok, False))
            stack.append(tok)
    return moves


def extract_moves(games: pd.DataFrame) -> pd.DataFrame:
    """One row per move (forward click or back click) of every game.

    Columns
    -------
    game_id, step (0-based index of the move in its game), from_page,
    to_page, is_back, after_back (previous move was a back click),
    click_no (0-based index among the forward clicks of the game, -1 for
    back clicks), target, finished.
    """
    rows = []
    for gid, path in zip(games["game_id"].to_numpy(), games["path"].to_numpy()):
        prev_back = False
        click_no = 0
        for step, (src, dst, is_back) in enumerate(resolve_path(path)):
            rows.append((gid, step, src, dst, is_back, prev_back, -1 if is_back else click_no))
            if not is_back:
                click_no += 1
            prev_back = is_back
    moves = pd.DataFrame(rows, columns=["game_id", "step", "from_page", "to_page",
                                        "is_back", "after_back", "click_no"])
    meta = games.set_index("game_id")[["target", "finished"]]
    return moves.join(meta, on="game_id")


def visited_before(moves: pd.DataFrame, article_ids: pd.Series) -> pd.DataFrame:
    """First step at which each article was seen in each game.

    Used to flag, for every click, the candidate links that point back to a
    page the player has already visited. The source page counts as visited
    at step -1.
    """
    first = (moves.loc[~moves["is_back"], ["game_id", "step", "to_page"]]
             .rename(columns={"to_page": "page"}))
    src = moves.groupby("game_id", sort=False)["from_page"].first().reset_index()
    src = src.rename(columns={"from_page": "page"}).assign(step=-1)
    seen = pd.concat([src, first], ignore_index=True)
    seen["page_id"] = seen["page"].map(article_ids)
    seen = seen.dropna(subset=["page_id"]).astype({"page_id": "int64"})
    return (seen.groupby(["game_id", "page_id"], sort=False)["step"].min()
            .rename("first_step").reset_index())


def featured_pairs(games: pd.DataFrame, min_games: int = 1000) -> pd.Series:
    """Source-target pairs played at least ``min_games`` times.

    Wikispeedia proposed a few fixed puzzles to many players (e.g. *Brain ->
    Telephone*). They are huge repeated experiments on the same decisions,
    so we keep them apart from the rest of the games.
    """
    counts = games.groupby(["source", "target"]).size().sort_values(ascending=False)
    return counts[counts >= min_games].rename("games")


def game_summary(games: pd.DataFrame, moves: pd.DataFrame, dist: np.ndarray,
                 article_ids: pd.Series, unreachable: int) -> pd.DataFrame:
    """Per-game descriptive variables used in the descriptive analysis.

    Adds the number of forward clicks and back clicks, the shortest possible
    number of clicks between source and target (``optimal``), and for
    finished games the number of extra clicks over that optimum.
    """
    per_game = moves.groupby("game_id")["is_back"]
    agg = pd.DataFrame({"n_back": per_game.sum()})
    agg["n_clicks"] = per_game.size() - agg["n_back"]
    out = games.join(agg, on="game_id")
    out[["n_clicks", "n_back"]] = out[["n_clicks", "n_back"]].fillna(0).astype(int)
    s = out["source"].map(article_ids)
    t = out["target"].map(article_ids)
    ok = s.notna() & t.notna()
    optimal = pd.Series(np.nan, index=out.index)
    d = dist[s[ok].astype(int).to_numpy(), t[ok].astype(int).to_numpy()].astype(float)
    d[d == unreachable] = np.nan
    optimal[ok] = d
    out["optimal"] = optimal
    out["extra_clicks"] = np.where(out["finished"], out["n_clicks"] - out["optimal"], np.nan)
    return out
