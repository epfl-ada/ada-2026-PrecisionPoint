"""Article- and link-level features computed from the plaintext articles.

* :func:`tfidf_similarity` - semantic similarity between every pair of
  articles (proxy for how related a link *looks* to the target).
* :func:`link_positions` - where on the source page each outgoing link
  appears (proxy for its visibility).
* :func:`article_text_stats` - length and lexicon valence of every article
  (used to explain temptation in RQ2).

The plaintext files are the text of the HTML pages players saw, in the same
order (infobox first, then body, then navigation boxes), so character
offsets in the plaintext approximate the vertical position of a link on the
rendered page. Exact positions require the 755 MB HTML release, which we
plan to parse for Milestone P3.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

BANNER = "2007 Schools Wikipedia Selection"
_DISAMBIGUATION = re.compile(r"\s*\([^)]*\)\s*$")
_WORD = r"(?u)\b[a-zA-Z][a-zA-Z]+\b"


def article_body(text: str) -> str:
    """Strip the copyright line, title and "Related subjects" banner.

    What remains starts with the infobox / first paragraph, i.e. the top of
    the page as displayed to players.
    """
    start = text.find(BANNER)
    if start == -1:
        return text.split("\n", 1)[-1]
    end = text.find("\n\n", start)
    return text[end + 2:] if end != -1 else ""


def tfidf_similarity(bodies: pd.Series) -> np.ndarray:
    """Cosine similarity of TF-IDF vectors for all pairs of articles.

    ``sublinear_tf`` dampens very long articles, and terms appearing in more
    than half of the articles are dropped as uninformative. Rows of the
    TF-IDF matrix are L2-normalised, so ``X @ X.T`` is the cosine matrix.
    The dense ``float32`` result takes ~85 MB and makes every lookup O(1).
    """
    vec = TfidfVectorizer(stop_words="english", sublinear_tf=True, min_df=2,
                          max_df=0.5, token_pattern=_WORD, dtype=np.float32)
    X = vec.fit_transform(bodies.to_numpy())
    return (X @ X.T).toarray().astype(np.float32)


def _mention_pattern(title: str) -> re.Pattern:
    """Regex matching a mention of an article title in running text.

    Anchor texts often differ slightly from the linked title, so the match
    is deliberately loose:

    * the disambiguation suffix is dropped (``Mercury (planet)`` -> ``Mercury``);
    * ``X language`` also matches ``X`` (``French language`` -> ``French``);
    * titles of six letters or more lose their last letter and every match
      may carry up to three extra letters, so ``Iran`` matches ``Iranian``
      and ``Chordate`` matches ``Chordata``.
    """
    base = _DISAMBIGUATION.sub("", title)
    variants = {base}
    if base.endswith(" language"):
        variants.add(base[: -len(" language")])
    stems = {v[:-1] if len(v) >= 6 else v for v in variants}
    alternation = "|".join(re.escape(v) for v in sorted(stems, key=len, reverse=True))
    return re.compile(r"(?<!\w)(?:" + alternation + r")[a-z]{0,3}(?!\w)", re.IGNORECASE)


def link_positions(links: pd.DataFrame, bodies: pd.Series, titles: pd.Series) -> pd.DataFrame:
    """Relative position and number of mentions of every link on its page.

    Parameters
    ----------
    links : DataFrame with integer columns ``source_id`` and ``target_id``.
    bodies : article bodies (see :func:`article_body`) indexed by article id.
    titles : decoded titles indexed by article id.

    Returns
    -------
    ``links`` with three new columns: ``rel_pos`` (offset of the first
    mention divided by the body length, 0 = top of the page, NaN when the
    title is not found), ``n_mentions`` and ``found``.
    """
    patterns = {i: _mention_pattern(t) for i, t in titles.items()}
    body = bodies.to_dict()
    lengths = bodies.str.len().to_dict()
    rel_pos = np.full(len(links), np.nan)
    n_mentions = np.zeros(len(links), dtype=np.int32)
    for k, (s, t) in enumerate(zip(links["source_id"].to_numpy(), links["target_id"].to_numpy())):
        hits = [m.start() for m in patterns[t].finditer(body[s])]
        if hits:
            rel_pos[k] = hits[0] / max(lengths[s], 1)
            n_mentions[k] = len(hits)
    return links.assign(rel_pos=rel_pos, n_mentions=n_mentions, found=~np.isnan(rel_pos))


def has_medical_infobox(bodies: pd.Series) -> pd.Series:
    """True for articles about a disease or disorder.

    Disease articles carry an infobox with classification codes (ICD-9,
    ICD-10, DiseasesDB, MedlinePlus, eMedicine). This objective rule finds 54
    articles. It is conservative: a few diseases without the infobox (e.g.
    *Beriberi*) are missed, but nothing that is not a disease is included,
    unlike the broader *Health and medicine* category, which also holds
    *Humour*, *Edible salt* or *Intelligence*.
    """
    return bodies.str.contains(r"ICD-\s?(?:9|10)\b|DiseasesDB|MedlinePlus|eMedicine", regex=True)


def article_text_stats(bodies: pd.Series, titles: pd.Series) -> pd.DataFrame:
    """Length and emotional tone of each article.

    Tone uses the VADER lexicon (about 7,500 words rated from -4 to +4 by
    human annotators). ``valence`` is the mean rating of the lexicon words
    found in the article, ``neg_share`` / ``pos_share`` the fraction of all
    words that are negative / positive. We use the lexicon directly instead
    of VADER's sentence scorer, whose rules (emoticons, capitalisation,
    exclamation marks) target social-media text rather than encyclopedic
    prose. Words of the article's own title are ignored, otherwise names
    such as *Great Britain* or *Dead Sea* would dominate the score.
    """
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    lexicon = {w: v for w, v in SentimentIntensityAnalyzer().lexicon.items()
               if re.fullmatch(r"[a-z]+", w)}
    vocab = sorted(lexicon)
    vec = CountVectorizer(vocabulary=vocab, lowercase=True, token_pattern=r"(?u)\b[a-z]+\b")
    counts = vec.transform(bodies.to_numpy()).tocsr()
    in_title = vec.transform(titles.reindex(bodies.index).to_numpy()) > 0
    counts = (counts - counts.multiply(in_title)).tocsr()
    counts.eliminate_zeros()
    scores = np.array([lexicon[w] for w in vocab])
    hits = np.asarray(counts.sum(axis=1)).ravel()
    word = re.compile(_WORD)
    n_words = np.array([len(word.findall(b)) for b in bodies.to_numpy()])
    neg = np.asarray(counts[:, scores < 0].sum(axis=1)).ravel()
    pos = np.asarray(counts[:, scores > 0].sum(axis=1)).ravel()
    with np.errstate(invalid="ignore", divide="ignore"):
        valence = np.where(hits > 0, counts @ scores / np.maximum(hits, 1), np.nan)
    return pd.DataFrame({"n_words": n_words, "valence": valence,
                         "neg_share": neg / np.maximum(n_words, 1),
                         "pos_share": pos / np.maximum(n_words, 1)}, index=bodies.index)
