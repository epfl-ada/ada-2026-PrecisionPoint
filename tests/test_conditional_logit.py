"""Checks of the conditional logit implementation on simulated data.

1. Coefficients used to simulate choices are recovered.
2. Estimates and standard errors match statsmodels' ConditionalLogit.

Run with ``python -m pytest tests``.
"""
import numpy as np
import pandas as pd
import pytest

from src.data.choice_sets import ChoiceData
from src.models.conditional_logit import ConditionalLogit

TRUE_BETA = np.array([1.0, -0.5, 2.0])


def simulate(n_clicks: int = 4000, seed: int = 0) -> ChoiceData:
    rng = np.random.default_rng(seed)
    sizes = rng.integers(2, 30, size=n_clicks)
    offsets = np.concatenate([[0], np.cumsum(sizes)])
    n_rows = offsets[-1]
    X = np.column_stack([rng.normal(size=n_rows), rng.uniform(size=n_rows),
                         rng.binomial(1, 0.1, size=n_rows)]).astype(np.float32)
    util = X @ TRUE_BETA + rng.gumbel(size=n_rows)
    chosen = np.array([offsets[k] + np.argmax(util[offsets[k]:offsets[k + 1]])
                       for k in range(n_clicks)])
    clicks = pd.DataFrame({"ip": rng.integers(0, 500, size=n_clicks),
                           "game_id": np.arange(n_clicks)})
    return ChoiceData(X, ["a", "b", "c"], offsets, np.zeros(n_rows, dtype=np.int32),
                      chosen, clicks)


def test_recovers_true_coefficients():
    data = simulate(n_clicks=20000)
    model = ConditionalLogit(["a", "b", "c"]).fit(data)
    se = np.sqrt(np.diag(model.cov_))
    assert np.all(np.abs(model.coef_ - TRUE_BETA) < 4 * se)


def test_matches_statsmodels():
    sm_cl = pytest.importorskip("statsmodels.discrete.conditional_models")
    data = simulate(n_clicks=1500, seed=1)
    ours = ConditionalLogit(["a", "b", "c"]).fit(data, cluster=None)
    y = data.chosen_mask().astype(float)
    groups = np.repeat(np.arange(data.n_clicks), data.sizes)
    # Newton: the default BFGS of statsmodels stops slightly before the optimum
    ref = sm_cl.ConditionalLogit(y, data.X.astype(float), groups=groups).fit(
        method="newton", maxiter=100, disp=False)
    np.testing.assert_allclose(ours.coef_, ref.params, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.sqrt(np.diag(ours.cov_)), ref.bse, rtol=1e-3)


def test_subset_keeps_choice_sets_consistent():
    data = simulate(n_clicks=50, seed=2)
    keep = np.arange(50) % 3 == 0
    sub = data.subset(keep)
    assert sub.n_clicks == keep.sum()
    for k, orig in enumerate(np.flatnonzero(keep)):
        a, b = data.offsets[orig], data.offsets[orig + 1]
        np.testing.assert_array_equal(sub.X[sub.offsets[k]:sub.offsets[k + 1]], data.X[a:b])
        assert sub.chosen[k] - sub.offsets[k] == data.chosen[orig] - a
