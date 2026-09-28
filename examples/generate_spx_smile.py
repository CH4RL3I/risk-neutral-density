"""Generate a synthetic, SPX-like option chain (no market data, no licensing issues).

Total implied variance follows a raw-SVI smile with negative skew; call and put quotes are priced
with Black-76, given a bid/ask spread that widens in the wings, and a little quote noise.
Writes examples/spx_like.csv. Run: ``uv run python examples/generate_spx_smile.py``
"""

from pathlib import Path

import numpy as np
import pandas as pd

from rnd.bs import black_price

SPOT, RATE, DIV = 5500.0, 0.04, 0.013
ASOF, EXPIRY = pd.Timestamp("2026-09-28"), pd.Timestamp("2026-12-28")
T = (EXPIRY - ASOF).days / 365.0
# raw SVI: w(k) = a + b (rho (k - m) + sqrt((k - m)^2 + s^2)), k = ln(K / F)
A, B, RHO, M, S = 0.0007, 0.05, -0.7, 0.01, 0.12


def svi_vol(k):
    w = A + B * (RHO * (k - M) + np.sqrt((k - M) ** 2 + S**2))
    return np.sqrt(w / T)


def main(seed: int = 7) -> None:
    rng = np.random.default_rng(seed)
    F = SPOT * np.exp((RATE - DIV) * T)
    rows = []
    for K in np.arange(4200, 7025, 25.0):
        vol = svi_vol(np.log(K / F))
        for kind in ("call", "put"):
            mid = float(black_price(F, K, T, vol, RATE, kind == "call"))
            mid *= 1 + rng.normal(0, 0.002)  # quote noise
            half = max(0.05, 0.004 * mid + 0.25 * abs(np.log(K / F)))  # wider spread in the wings
            bid, ask = round(mid - half, 1), round(mid + half, 1)
            if bid > 0:
                rows.append((EXPIRY.date(), K, kind, bid, ask))
    out = Path(__file__).with_name("spx_like.csv")
    pd.DataFrame(rows, columns=["expiry", "strike", "type", "bid", "ask"]).to_csv(out, index=False)
    print(f"wrote {len(rows)} quotes to {out} (spot {SPOT}, asof {ASOF.date()}, T={T:.4f})")


if __name__ == "__main__":
    main()
