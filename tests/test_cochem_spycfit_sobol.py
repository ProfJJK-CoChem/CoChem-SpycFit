"""Sobol sensitivity estimator checked against functions with analytic indices.

Requirement: SF-FIT-005 (sensitivity measures identify their estimator and limitations),
amendment §4 and §5 (the total-effect estimator varied the complement of the input).

Evidence class: analytical correctness. The expected indices are computed below from
closed-form expressions, not from the code under test.

* Linear additive model Y = sum(c_i X_i), X_i ~ U(0, 1): V_i = c_i**2 / 12 and, with no
  interactions, S_i = S_Ti = V_i / V.
* Ishigami function Y = sin X1 + a sin^2 X2 + b X3^4 sin X1, X_i ~ U(-pi, pi)
  (Ishigami & Homma 1990, doi:10.1109/ISUMA.1990.151285; indices as in Saltelli et al.
  2010, doi:10.1016/j.cpc.2009.09.018).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from cochem_spycfit.convolver import compute_sobol_sensitivity

N_SAMPLES = 20000
# Sampling tolerance for N_SAMPLES base samples; the 7e42507 estimator was off by more
# than 0.3 on the indices checked below.
TOL = 0.04


def _linear(coefficients):
    c = np.asarray(coefficients, dtype=float)

    def model(p):
        return np.atleast_1d(np.dot(c, np.asarray(p, dtype=float)))

    return model


def test_linear_additive_indices() -> None:
    coefficients = [1.0, 0.0, 0.5]
    variances = [c * c / 12.0 for c in coefficients]
    total = sum(variances)
    expected = [v / total for v in variances]

    result = compute_sobol_sensitivity(
        model_fn=_linear(coefficients),
        param_bounds=np.array([[0.0, 1.0]] * 3),
        n_samples=N_SAMPLES,
        random_seed=7,
        param_names=["x1", "x2", "x3"],
    )

    for name, value in zip(["x1", "x2", "x3"], expected):
        assert result["first_order_S"][name] == pytest.approx(value, abs=TOL)
        assert result["total_order_ST"][name] == pytest.approx(value, abs=TOL)


def test_ishigami_indices() -> None:
    a, b = 7.0, 0.1
    pi = math.pi
    v1 = 0.5 * (1.0 + b * pi ** 4 / 5.0) ** 2
    v2 = a * a / 8.0
    v13 = 8.0 * b * b * pi ** 8 / 225.0
    var = v1 + v2 + v13
    expected_s = {"x1": v1 / var, "x2": v2 / var, "x3": 0.0}
    expected_st = {"x1": (v1 + v13) / var, "x2": v2 / var, "x3": v13 / var}

    def ishigami(p):
        x1, x2, x3 = (float(v) for v in p)
        return np.atleast_1d(math.sin(x1) + a * math.sin(x2) ** 2 + b * x3 ** 4 * math.sin(x1))

    result = compute_sobol_sensitivity(
        model_fn=ishigami,
        param_bounds=np.array([[-pi, pi]] * 3),
        n_samples=N_SAMPLES,
        random_seed=11,
        param_names=["x1", "x2", "x3"],
    )

    for name in ("x1", "x2", "x3"):
        assert result["first_order_S"][name] == pytest.approx(expected_s[name], abs=TOL)
        assert result["total_order_ST"][name] == pytest.approx(expected_st[name], abs=TOL)


def test_zero_output_variance_is_an_error() -> None:
    """Regression: a zero variance was replaced by 1.0, giving indices of 0."""
    with pytest.raises(ValueError, match="variance is zero"):
        compute_sobol_sensitivity(
            model_fn=lambda p: np.atleast_1d(3.0),
            param_bounds=np.array([[0.0, 1.0]] * 2),
            n_samples=50,
            param_names=["x1", "x2"],
        )


def test_names_must_match_parameters() -> None:
    """Regression: any bounds were labelled with the eight Watson names."""
    with pytest.raises(ValueError, match="param_names is required"):
        compute_sobol_sensitivity(
            model_fn=_linear([1.0, 1.0]),
            param_bounds=np.array([[0.0, 1.0]] * 2),
            n_samples=50,
        )
