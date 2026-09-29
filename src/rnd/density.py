"""Breeden-Litzenberger: risk-neutral density from a smoothed implied-vol smile.

    q(K) = e^{rT} * d^2 C / dK^2

Pipeline: OTM quotes -> implied vols -> smooth smile in log-moneyness -> rebuild Black-76 call
prices on a fine uniform strike grid -> second difference -> clip negatives -> renormalise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.polynomial import Polynomial
from scipy.integrate import cumulative_trapezoid, trapezoid
from scipy.interpolate import UnivariateSpline

from .bs import black_price, lognormal_pdf
from .quotes import SmilePoints, smile_points


class Smile:
    """Smooth implied-vol curve sigma(k), k = ln(K/F), fitted in total variance w = sigma^2 T.

    Inside the quoted range w is a polynomial (``method="poly"``) or a smoothing spline
    (``method="spline"``); total variance is much closer to polynomial than vol is, and its
    wings are close to linear. Outside, w continues from the edge (d = distance from the edge)
    with curvature ``w''(d) = (s2 + (s3 + s2/tau) d) e^{-d/tau}``, where s1, s2, s3 are the
    edge's first three derivatives. The extension is C^3, so the implied density is continuous
    and has no kink at the edge, and its curvature dies out, so w ends up linear in log-strike,
    the behaviour Lee's moment formula requires of the wings.
    """

    def __init__(self, k, iv, T, method="poly", degree=5, smoothing=None, tau=0.1):
        k, w = np.asarray(k, float), np.asarray(iv, float) ** 2 * T
        if method == "poly":
            p = Polynomial.fit(k, w, degree)
            self._core, self._d, self._d2, self._d3 = p, p.deriv(), p.deriv(2), p.deriv(3)
        elif method == "spline":
            # smoothing budget: quote noise of ~0.3 vol points at the ATM variance level
            if smoothing is None:
                smoothing = len(k) * (2 * 0.003 * np.sqrt(w.mean() * T)) ** 2
            s = smoothing
            sp = UnivariateSpline(k, w, k=3, s=s)
            self._core, self._d, self._d2 = sp, sp.derivative(), sp.derivative(2)
            self._d3 = sp.derivative(3)
        else:
            raise ValueError("method must be 'poly' or 'spline'")
        self.k0, self.k1, self.tau, self.T, self.method = k.min(), k.max(), tau, T, method

    def _extension(self, edge, sign, d):
        """Outward continuation of total variance from ``edge``; sign +1 right, -1 left."""
        w0, s1, s2 = float(self._core(edge)), sign * float(self._d(edge)), float(self._d2(edge))
        s3 = sign * float(self._d3(edge))
        t = self.tau
        e = np.exp(-d / t)
        f = t * t * (d / t - 1 + e)  # double integral of e^{-d/t}
        g = t * t * d - 2 * t**3 + t * t * d * e + 2 * t**3 * e  # double integral of d e^{-d/t}
        return w0 + s1 * d + s2 * f + (s3 + s2 / t) * g

    def __call__(self, k):
        k = np.asarray(k, float)
        w = np.asarray(self._core(np.clip(k, self.k0, self.k1)), float)
        w = np.where(k > self.k1, self._extension(self.k1, 1, np.abs(k - self.k1)), w)
        w = np.where(k < self.k0, self._extension(self.k0, -1, np.abs(self.k0 - k)), w)
        return np.sqrt(np.maximum(w, 1e-6 * self.T) / self.T)


@dataclass
class Density:
    strikes: np.ndarray
    pdf: np.ndarray  # renormalised to integrate to 1
    raw_integral: float  # integral of the clipped density before renormalising (diagnostic)
    negative_mass: float  # integral of the discarded negative part
    forward: float
    spot: float
    T: float
    rate: float
    atm_vol: float
    smile: Smile
    points: SmilePoints

    def _int(self, y):
        return float(trapezoid(y, self.strikes))

    @property
    def mean(self):
        return self._int(self.strikes * self.pdf)

    @property
    def std(self):
        return float(np.sqrt(self._int((self.strikes - self.mean) ** 2 * self.pdf)))

    @property
    def skew(self):
        return self._int(((self.strikes - self.mean) / self.std) ** 3 * self.pdf)

    @property
    def excess_kurtosis(self):
        return self._int(((self.strikes - self.mean) / self.std) ** 4 * self.pdf) - 3.0

    def cdf(self, x):
        c = cumulative_trapezoid(self.pdf, self.strikes, initial=0.0)
        return np.interp(x, self.strikes, c)

    def prob_below(self, x):
        return float(self.cdf(x))

    def prob_above(self, x):
        return 1.0 - self.prob_below(x)

    def lognormal(self):
        """Benchmark: lognormal with the same forward and ATM vol."""
        return lognormal_pdf(self.strikes, self.forward, self.atm_vol, self.T)


def density_from_smile(smile, forward, T, rate, n_grid=4001, width=8.0):
    """Breeden-Litzenberger on a callable smile. Returns (strikes, raw_pdf) with the raw pdf
    *not* clipped or normalised, on the interior grid points."""
    s = float(smile(0.0)) * np.sqrt(T)
    K = np.linspace(max(forward * np.exp(-width * s), 1e-6), forward * np.exp(width * s), n_grid)
    C = black_price(forward, K, T, smile(np.log(K / forward)), rate, is_call=True)
    h = K[1] - K[0]
    d2C = (C[2:] - 2 * C[1:-1] + C[:-2]) / h**2
    return K[1:-1], np.exp(rate * T) * d2C


def estimate_density(
    quotes: pd.DataFrame,
    T: float,
    rate: float,
    spot: float | None = None,
    div_yield: float = 0.0,
    forward: float | None = None,
    method: str = "poly",
    degree: int = 5,
    n_grid: int = 4001,
) -> Density:
    """Estimate the risk-neutral density of S_T from one expiry of option quotes."""
    pts = smile_points(quotes, T, rate, forward)
    smile = Smile(pts.k, pts.iv, T, method=method, degree=degree)
    K, raw = density_from_smile(smile, pts.forward, T, rate, n_grid)
    clipped = np.clip(raw, 0.0, None)
    raw_integral = float(trapezoid(clipped, K))
    neg = float(trapezoid(np.clip(raw, None, 0.0), K))
    spot = spot if spot is not None else pts.forward * np.exp(-(rate - div_yield) * T)
    return Density(
        strikes=K,
        pdf=clipped / raw_integral,
        raw_integral=raw_integral,
        negative_mass=neg,
        forward=pts.forward,
        spot=float(spot),
        T=T,
        rate=rate,
        atm_vol=float(smile(0.0)),
        smile=smile,
        points=pts,
    )
