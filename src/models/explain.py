r"""RQ2: relating article temptation to article characteristics.

The natural model is the count model behind the temptation score itself.
Article :math:`a` receives :math:`O_a` clicks out of :math:`E_a` expected,
and its temptation depends on its characteristics :math:`z_a`:

.. math::  O_a \sim \mathrm{NegBin}\big(\mu_a = E_a\, e^{z_a^\top \gamma},\ \mathrm{Var} = \mu_a + \kappa \mu_a^2\big).

This is the Gamma-Poisson model of :mod:`src.models.temptation` with a prior
mean that depends on :math:`z_a` instead of being shared by all articles:
:math:`\log E_a` enters as an offset (exposure) and :math:`e^{\gamma_j}` is
the multiplier on clicks associated with characteristic :math:`j`. Working
with the counts avoids regressing already-shrunk estimates, which would
attenuate the coefficients, and weights every article by how much
information it carries.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def nb_regression(table: pd.DataFrame, rhs: str, min_expected: float = 1.0):
    """Negative binomial (NB2) regression of observed on expected clicks.

    ``table`` needs ``observed`` and ``expected`` columns plus the variables
    used in ``rhs`` (the right-hand side of a patsy formula). Articles with
    fewer than ``min_expected`` expected clicks carry almost no information
    and are left out. Standard errors are heteroskedasticity-robust.
    """
    d = table[table["expected"] >= min_expected]
    model = smf.negativebinomial(f"observed ~ {rhs}", data=d, exposure=d["expected"])
    return model.fit(disp=0, maxiter=1000, cov_type="HC1")


def tidy(results, exp_label: str = "multiplier on clicks") -> pd.DataFrame:
    """Coefficient table with 95% confidence intervals and exponentiated effects.

    The dispersion parameter of the negative binomial (``alpha`` in
    statsmodels) is dropped from the table.
    """
    ci = results.conf_int()
    out = pd.DataFrame({"coef": results.params, "se": results.bse, "p_value": results.pvalues,
                        "ci_low": ci[0], "ci_high": ci[1]})
    out = out.drop(index=[i for i in out.index if i == "alpha"])
    out[exp_label] = np.exp(out["coef"])
    out[f"{exp_label} (95% CI)"] = [f"{np.exp(l):.2f} to {np.exp(h):.2f}"
                                    for l, h in zip(out["ci_low"], out["ci_high"])]
    return out
