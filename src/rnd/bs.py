"""Black-76 pricing (Black-Scholes on the forward), implied vol and the lognormal density."""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr


def black_price(F, K, T, sigma, r, is_call=True):
    """Discounted Black-76 price of a European option on forward ``F``."""
    F, K, sigma = np.broadcast_arrays(*(np.asarray(x, dtype=float) for x in (F, K, sigma)))
    sd = sigma * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sd**2) / sd
    d2 = d1 - sd
    df = np.exp(-r * T)
    call = df * (F * ndtr(d1) - K * ndtr(d2))
    return call if is_call else call - df * (F - K)


def implied_vol(price, F, K, T, r, is_call=True):
    """Invert Black-76 by bracketing. Returns NaN if the price is outside no-arbitrage bounds."""
    df = np.exp(-r * T)
    intrinsic = df * max(F - K, 0.0) if is_call else df * max(K - F, 0.0)
    upper = df * F if is_call else df * K
    if not intrinsic < price < upper:
        return np.nan

    def f(s):
        return float(black_price(F, K, T, s, r, is_call)) - price

    lo, hi = 1e-4, 5.0
    if f(lo) > 0 or f(hi) < 0:
        return np.nan
    return brentq(f, lo, hi, xtol=1e-12)


def lognormal_pdf(K, F, sigma, T):
    """Density of S_T when log S_T is normal with mean ln F - sigma^2 T / 2 (so E[S_T] = F)."""
    K = np.asarray(K, dtype=float)
    sd = sigma * np.sqrt(T)
    z = (np.log(K / F) + 0.5 * sd**2) / sd
    return np.exp(-0.5 * z**2) / (K * sd * np.sqrt(2 * np.pi))
