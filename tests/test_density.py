"""Validation against known answers.

The key check: quotes generated from Black-Scholes must give back the analytic lognormal density.
Every scenario is priced from a known model, so the "truth" never depends on the code under test.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import trapezoid

from rnd import black_price, density_from_smile, estimate_density, implied_vol, lognormal_pdf
from rnd.quotes import estimate_forward, smile_points

S0, R, DIV, T, SIGMA = 100.0, 0.03, 0.01, 0.5, 0.25
F = S0 * np.exp((R - DIV) * T)
STRIKES = np.arange(60.0, 160.1, 2.5)


def make_quotes(vol_fn, strikes=STRIKES, iv_noise=0.0, half_spread=0.0, seed=0):
    """Call and put quotes on forward F. ``vol_fn(k)`` gives the true vol, k = ln(K/F)."""
    rng = np.random.default_rng(seed)
    rows = []
    for K in strikes:
        vol = float(vol_fn(np.log(K / F)))
        for kind in ("call", "put"):
            v = vol + rng.normal(0, iv_noise) if iv_noise else vol
            mid = float(black_price(F, K, T, v, R, kind == "call"))
            rows.append((K, kind, mid - half_spread, mid + half_spread))
    df = pd.DataFrame(rows, columns=["strike", "type", "bid", "ask"])
    df["mid"] = (df["bid"] + df["ask"]) / 2
    return df


def flat(k):
    return np.full_like(np.asarray(k, float), SIGMA)


def peak_error(d, truth_pdf):
    return np.abs(d.pdf - truth_pdf).max() / truth_pdf.max()


def lognormal_std(sigma=SIGMA):
    return F * np.sqrt(np.exp(sigma**2 * T) - 1)


def test_implied_vol_roundtrip():
    for kind in (True, False):
        p = float(black_price(F, 110.0, T, 0.31, R, kind))
        assert implied_vol(p, F, 110.0, T, R, kind) == pytest.approx(0.31, abs=1e-9)


def test_implied_vol_rejects_arbitrage_prices():
    assert np.isnan(implied_vol(0.0, F, 130.0, T, R))
    assert np.isnan(implied_vol(1e6, F, 130.0, T, R))


def test_lognormal_pdf_has_forward_mean_and_unit_mass():
    K = np.linspace(1, 600, 200_000)
    q = lognormal_pdf(K, F, SIGMA, T)
    assert trapezoid(q, K) == pytest.approx(1.0, abs=1e-6)
    assert trapezoid(K * q, K) == pytest.approx(F, rel=1e-5)


def test_recovers_black_scholes_density():
    d = estimate_density(make_quotes(flat), T, R, spot=S0)
    truth = lognormal_pdf(d.strikes, F, SIGMA, T)
    assert d.forward == pytest.approx(F, rel=1e-9)
    # the e^{rT} factor matters: without it the raw integral would be off by ~1.5%
    assert d.raw_integral == pytest.approx(1.0, abs=1e-3)
    assert d.mean == pytest.approx(F, rel=1e-3)
    assert d.std == pytest.approx(lognormal_std(), rel=5e-3)
    assert peak_error(d, truth) < 2e-3
    assert d.negative_mass > -1e-6


def test_probabilities_match_lognormal_cdf():
    from scipy.stats import norm

    d = estimate_density(make_quotes(flat), T, R, spot=S0)
    sd = SIGMA * np.sqrt(T)
    for x in (0.9 * S0, S0, 1.1 * S0):
        expected = norm.cdf((np.log(x / F) + 0.5 * sd**2) / sd)
        assert d.prob_below(x) == pytest.approx(expected, abs=2e-3)


@pytest.mark.parametrize("method", ["poly", "spline"])
def test_noisy_quotes_are_smoothed(method):
    quotes = make_quotes(flat, iv_noise=0.002, half_spread=0.01, seed=3)
    d = estimate_density(quotes, T, R, spot=S0, method=method)
    truth = lognormal_pdf(d.strikes, F, SIGMA, T)
    assert d.raw_integral == pytest.approx(1.0, abs=0.01)
    assert d.mean == pytest.approx(F, rel=5e-3)
    assert d.std == pytest.approx(lognormal_std(), rel=0.03)
    assert peak_error(d, truth) < 0.05


def test_naive_finite_differences_fail_on_the_same_noisy_quotes():
    """Motivation for smoothing: differentiate the raw noisy call prices directly."""
    quotes = make_quotes(flat, iv_noise=0.002, half_spread=0.01, seed=3)
    calls = quotes[quotes["type"] == "call"]
    K, C = calls["strike"].to_numpy(), calls["mid"].to_numpy()
    h = K[1] - K[0]
    naive = np.exp(R * T) * (C[2:] - 2 * C[1:-1] + C[:-2]) / h**2
    truth = lognormal_pdf(K[1:-1], F, SIGMA, T)
    naive_err = np.abs(naive - truth).max() / truth.max()

    d = estimate_density(quotes, T, R, spot=S0)
    smooth_err = peak_error(d, lognormal_pdf(d.strikes, F, SIGMA, T))
    assert naive_err > 5 * smooth_err


def test_skewed_smile_matches_reference_density():
    """Non-flat smile: compare to Breeden-Litzenberger applied to the exact smile."""

    def skew(k):
        k = np.asarray(k, float)
        w = 0.0011 + 0.05 * (-0.7 * (k - 0.01) + np.sqrt((k - 0.01) ** 2 + 0.2**2))
        return np.sqrt(w / T)

    d = estimate_density(make_quotes(skew, iv_noise=0.001, seed=5), T, R, spot=S0)
    K, ref = density_from_smile(skew, F, T, R)
    ref = np.interp(d.strikes, K, ref, left=0.0, right=0.0)
    ref /= trapezoid(ref, d.strikes)
    assert peak_error(d, ref) < 0.05
    assert d.mean == pytest.approx(F, rel=5e-3)
    assert d.skew < -0.3  # equity-style negative skew


def test_forward_from_put_call_parity():
    assert estimate_forward(make_quotes(flat), T, R) == pytest.approx(F, rel=1e-9)


def test_smile_uses_otm_options_only():
    pts = smile_points(make_quotes(flat), T, R)
    assert np.allclose(pts.iv, SIGMA, atol=1e-6)
    assert len(pts.strikes) == len(STRIKES)  # exactly one option per strike


def test_dropping_the_discount_factor_would_be_caught():
    """Guard for the e^{rT} factor: the raw integral must be 1, not e^{-rT}."""
    d = estimate_density(make_quotes(flat), T, R, spot=S0)
    assert abs(d.raw_integral - np.exp(-R * T)) > 5e-3
