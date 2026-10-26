r"""Conditional (McFadden) logit for choosing one link among those of a page.

Model
-----
Click :math:`k` happens on a page whose outgoing links form the choice set
:math:`C_k`. Candidate :math:`j \in C_k` has features :math:`x_{kj}` and
utility :math:`U_{kj} = \beta^\top x_{kj} + \varepsilon_{kj}` with i.i.d.
Gumbel noise, which gives

.. math::  P(y_k = j) = \frac{\exp(\beta^\top x_{kj})}{\sum_{l \in C_k} \exp(\beta^\top x_{kl})}.

The log-likelihood :math:`\ell(\beta) = \sum_k [\beta^\top x_{k y_k} - \log
\sum_{l} \exp(\beta^\top x_{kl})]` is concave, with gradient
:math:`\sum_k (x_{k y_k} - \bar x_k)` where :math:`\bar x_k = \sum_l p_{kl}
x_{kl}`, and negative Hessian (Fisher information)
:math:`I(\beta) = \sum_k \sum_l p_{kl} (x_{kl} - \bar x_k)(x_{kl} - \bar x_k)^\top`.
We maximise it with Newton's method (a handful of iterations), streaming
over chunks of rows so that the 23-million-row design never has to be
copied in float64.

Clicks of the same player are not independent, so besides the classical
standard errors :math:`\sqrt{\mathrm{diag}\, I^{-1}}` we report
cluster-robust (sandwich) errors :math:`I^{-1} M I^{-1}` with
:math:`M = \frac{G}{G-1} \sum_g s_g s_g^\top`, where :math:`s_g` is the sum
of the per-click scores of player :math:`g`.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from ..data.choice_sets import ChoiceData

_CHUNK_ROWS = 2_000_000


def _chunks(offsets: np.ndarray, max_rows: int = _CHUNK_ROWS):
    """Yield (first_click, last_click_exclusive) ranges of about ``max_rows`` rows."""
    n = len(offsets) - 1
    start = 0
    while start < n:
        end = int(np.searchsorted(offsets, offsets[start] + max_rows, side="right")) - 1
        end = min(max(end, start + 1), n)
        yield start, end
        start = end


def _segment_softmax(v: np.ndarray, starts: np.ndarray, sizes: np.ndarray):
    """Log-normaliser and probabilities of each contiguous segment of ``v``."""
    vmax = np.maximum.reduceat(v, starts)
    e = np.exp(v - np.repeat(vmax, sizes))
    denom = np.add.reduceat(e, starts)
    return vmax + np.log(denom), e / np.repeat(denom, sizes)


@dataclass
class _Pass:
    loglik: float
    grad: np.ndarray
    info: np.ndarray | None
    scores: np.ndarray | None


def _utility(X: np.ndarray, cols: np.ndarray, beta: np.ndarray, r0: int, r1: int,
             offset: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    Xc = X[r0:r1, cols].astype(np.float64)
    v = Xc @ beta
    if offset is not None:
        v += offset[r0:r1]
    return Xc, v


def evaluate(data: ChoiceData, cols: np.ndarray, beta: np.ndarray, *, hessian: bool = True,
             scores: bool = False, offset: np.ndarray | None = None) -> _Pass:
    """One streaming pass: log-likelihood, gradient, information and per-click scores.

    ``offset`` is an optional fixed term added to every utility (used to plug
    in article effects).
    """
    p = len(cols)
    ll, grad = 0.0, np.zeros(p)
    info = np.zeros((p, p)) if hessian else None
    score_rows = np.empty((data.n_clicks, p)) if scores else None
    for a, b in _chunks(data.offsets):
        r0, r1 = int(data.offsets[a]), int(data.offsets[b])
        Xc, v = _utility(data.X, cols, beta, r0, r1, offset)
        off = data.offsets[a:b + 1] - r0
        starts, sizes = off[:-1], np.diff(off)
        log_norm, prob = _segment_softmax(v, starts, sizes)
        ch = data.chosen[a:b] - r0
        ll += float((v[ch] - log_norm).sum())
        xbar = np.add.reduceat(prob[:, None] * Xc, starts)
        s = Xc[ch] - xbar
        grad += s.sum(axis=0)
        if hessian:
            info += (Xc * prob[:, None]).T @ Xc - xbar.T @ xbar
        if scores:
            score_rows[a:b] = s
    return _Pass(ll, grad, info, score_rows)


def predict_proba(data: ChoiceData, cols: np.ndarray, beta: np.ndarray,
                  offset: np.ndarray | None = None) -> np.ndarray:
    """Choice probability of every row (float32, sums to one per click)."""
    out = np.empty(data.n_rows, dtype=np.float32)
    for a, b in _chunks(data.offsets):
        r0, r1 = int(data.offsets[a]), int(data.offsets[b])
        _, v = _utility(data.X, cols, beta, r0, r1, offset)
        off = data.offsets[a:b + 1] - r0
        out[r0:r1] = _segment_softmax(v, off[:-1], np.diff(off))[1]
    return out


class ConditionalLogit:
    """Conditional logit fitted by Newton-Raphson on a :class:`ChoiceData`.

    Parameters
    ----------
    features : names of the columns of ``data.X`` entering the utility.
    """

    def __init__(self, features: list[str]):
        self.features = list(features)
        self.coef_: np.ndarray | None = None

    # -- fitting -------------------------------------------------------------------------
    def fit(self, data: ChoiceData, cluster: str | None = "ip", max_iter: int = 50,
            tol: float = 1e-9, max_step: float = 2.0, verbose: bool = False) -> "ConditionalLogit":
        """Damped Newton-Raphson.

        Far from the optimum a pure Newton step can overshoot badly: once the
        target link gets a probability close to one, the curvature along its
        coefficient vanishes and the next step explodes. Each step is
        therefore capped at ``max_step`` in every coordinate and accepted
        only if it satisfies the Armijo sufficient-increase condition
        (backtracking otherwise). Close to the optimum full steps are taken
        and convergence is quadratic.
        """
        cols = data.columns(self.features)
        beta = np.zeros(len(cols))
        prev = evaluate(data, cols, beta)
        self.loglik_null_ = -float(np.log(data.sizes).sum())
        n_steps, converged = 0, False
        for _ in range(max_iter):
            step = np.linalg.solve(prev.info, prev.grad)
            decrement = float(prev.grad @ step)  # Newton decrement squared
            if decrement < tol * max(1.0, abs(prev.loglik)):
                converged = True
                break
            step *= min(1.0, max_step / np.abs(step).max())
            slope, t, cur = float(prev.grad @ step), 1.0, None
            while t >= 1e-8:
                trial = evaluate(data, cols, beta + t * step)
                if trial.loglik >= prev.loglik + 1e-4 * t * slope:
                    cur = trial
                    break
                t /= 2
            if cur is None:  # no ascent possible any more: numerically at the optimum
                warnings.warn("line search failed; stopping at the current estimate")
                break
            beta = beta + t * step
            n_steps += 1
            if verbose:
                print(f"step {n_steps}: loglik={cur.loglik:,.2f} t={t:.3g} decrement={decrement:.2e}")
            prev = cur
        else:
            warnings.warn(f"no convergence after {max_iter} Newton steps")
        final = evaluate(data, cols, beta, scores=cluster is not None)
        self.coef_ = beta
        self.loglik_ = final.loglik
        self.n_clicks_ = data.n_clicks
        self.n_iter_ = n_steps
        self.converged_ = converged
        cov = np.linalg.inv(final.info)
        self.cov_ = cov
        if cluster is not None:
            groups = pd.factorize(data.clicks[cluster])[0]
            n_groups = groups.max() + 1
            summed = np.zeros((n_groups, len(cols)))
            np.add.at(summed, groups, final.scores)
            meat = summed.T @ summed * n_groups / (n_groups - 1)
            self.cov_robust_ = cov @ meat @ cov
            self.n_clusters_ = int(n_groups)
        else:
            self.cov_robust_ = None
        return self

    # -- results -------------------------------------------------------------------------
    @property
    def pseudo_r2_(self) -> float:
        """McFadden's pseudo R^2 relative to random clicking (all links equally likely)."""
        return 1.0 - self.loglik_ / self.loglik_null_

    def summary(self) -> pd.DataFrame:
        """Coefficients, odds ratios, classical and cluster-robust inference."""
        se = np.sqrt(np.diag(self.cov_))
        out = pd.DataFrame({"coef": self.coef_, "odds_ratio": np.exp(self.coef_), "se": se},
                           index=pd.Index(self.features, name="feature"))
        if self.cov_robust_ is not None:
            rse = np.sqrt(np.diag(self.cov_robust_))
            out["se_robust"] = rse
            z = self.coef_ / rse
        else:
            z = self.coef_ / se
        out["z"] = z
        out["p_value"] = 2 * stats.norm.sf(np.abs(z))
        return out

    def predict_proba(self, data: ChoiceData) -> np.ndarray:
        return predict_proba(data, data.columns(self.features), self.coef_)

    def score(self, data: ChoiceData) -> dict:
        """Out-of-sample metrics on another :class:`ChoiceData`.

        Returns the mean log-likelihood per click, McFadden's pseudo R^2
        against random clicking, top-1 accuracy (the clicked link has the
        highest utility) and the mean percentile rank of the clicked link
        within its page (0 = best ranked).
        """
        cols = data.columns(self.features)
        res = evaluate(data, cols, self.coef_, hessian=False)
        null = -float(np.log(data.sizes).sum())
        hits, pct = 0, 0.0
        for a, b in _chunks(data.offsets):
            r0, r1 = int(data.offsets[a]), int(data.offsets[b])
            _, v = _utility(data.X, cols, self.coef_, r0, r1, None)
            off = data.offsets[a:b + 1] - r0
            sizes = np.diff(off)
            vc = v[data.chosen[a:b] - r0]
            hits += int((np.maximum.reduceat(v, off[:-1]) <= vc).sum())
            better = np.add.reduceat((v > np.repeat(vc, sizes)).astype(np.float64), off[:-1])
            pct += float((better / sizes).sum())
        return {"loglik_per_click": res.loglik / data.n_clicks,
                "pseudo_r2": 1.0 - res.loglik / null,
                "top1_accuracy": hits / data.n_clicks,
                "mean_rank_pct": pct / data.n_clicks}
