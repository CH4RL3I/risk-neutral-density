"""Loading option quotes and turning them into a clean implied-vol smile.

Handling of puts: for every strike we use the *out-of-the-money* option (puts below the forward,
calls above it). OTM options carry almost no intrinsic value, so their prices are the most
informative about volatility and the least contaminated by early-exercise premium and stale
in-the-money quotes. The forward itself is backed out of put-call parity,
``F = K + e^{rT} (C - P)``, using the strikes where call and put prices are closest, which
absorbs dividends and borrow costs without needing a dividend assumption.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .bs import implied_vol

_TYPES = {"c": "call", "call": "call", "p": "put", "put": "put"}


def load_quotes(path) -> pd.DataFrame:
    """Read a CSV with columns strike, type, expiry and either bid+ask or mid.

    An optional ``spot`` column is kept if present. Rows with a zero bid or a non-positive
    mid are dropped (no real market).
    """
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    missing = {"strike", "type", "expiry"} - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {sorted(missing)}")
    df["type"] = df["type"].astype(str).str.strip().str.lower().map(_TYPES)
    if df["type"].isna().any():
        raise ValueError("column 'type' must contain call/put (or c/p)")
    df["expiry"] = pd.to_datetime(df["expiry"]).dt.normalize()
    if "mid" not in df.columns:
        if not {"bid", "ask"} <= set(df.columns):
            raise ValueError("CSV needs either a 'mid' column or both 'bid' and 'ask'")
        df["mid"] = (df["bid"] + df["ask"]) / 2
    keep = df["mid"] > 0
    if "bid" in df.columns:
        keep &= df["bid"] > 0
    return df[keep].reset_index(drop=True)


def estimate_forward(quotes: pd.DataFrame, T: float, r: float) -> float:
    """Forward from put-call parity: median over the three strikes with the smallest |C - P|."""
    calls = quotes[quotes["type"] == "call"].groupby("strike")["mid"].mean()
    puts = quotes[quotes["type"] == "put"].groupby("strike")["mid"].mean()
    both = calls.index.intersection(puts.index)
    if len(both) == 0:
        raise ValueError("cannot infer the forward: no strike has both a call and a put quote")
    diff = calls[both] - puts[both]
    near = diff.abs().nsmallest(3).index
    return float(np.median(near.to_numpy() + np.exp(r * T) * diff[near].to_numpy()))


@dataclass
class SmilePoints:
    forward: float
    k: np.ndarray  # log-moneyness ln(K / F)
    iv: np.ndarray
    strikes: np.ndarray


def smile_points(quotes: pd.DataFrame, T: float, r: float, forward: float | None = None):
    """OTM implied vols in log-moneyness, sorted by strike. Unsolvable quotes are dropped."""
    F = forward if forward is not None else estimate_forward(quotes, T, r)
    otm = quotes[
        ((quotes["type"] == "put") & (quotes["strike"] < F))
        | ((quotes["type"] == "call") & (quotes["strike"] >= F))
    ]
    K, iv = [], []
    for strike, mid, kind in zip(otm["strike"], otm["mid"], otm["type"], strict=True):
        v = implied_vol(mid, F, strike, T, r, is_call=kind == "call")
        if np.isfinite(v):
            K.append(strike)
            iv.append(v)
    if len(K) < 6:
        raise ValueError(f"only {len(K)} usable OTM quotes; need at least 6")
    order = np.argsort(K)
    K, iv = np.asarray(K)[order], np.asarray(iv)[order]
    return SmilePoints(forward=F, k=np.log(K / F), iv=iv, strikes=K)


def load_yfinance(ticker: str, expiry: str | None = None) -> pd.DataFrame:
    """Fetch one expiry of a live option chain (needs ``pip install rnd[data]`` and a network).

    Returns a frame in the same layout as the CSV loader, with an extra ``spot`` column.
    Never used in the tests.
    """
    try:
        import yfinance as yf
    except ImportError as e:  # pragma: no cover
        raise ImportError("install the data extra: pip install 'rnd[data]'") from e
    t = yf.Ticker(ticker)
    expiry = expiry or t.options[0]
    chain = t.option_chain(expiry)
    spot = float(t.history(period="1d")["Close"].iloc[-1])
    frames = []
    for kind, df in (("call", chain.calls), ("put", chain.puts)):
        f = df[["strike", "bid", "ask"]].copy()
        f["type"], f["expiry"], f["spot"] = kind, expiry, spot
        frames.append(f)
    return pd.concat(frames, ignore_index=True)
