"""Command line interface: ``rnd fit`` and ``rnd fetch``."""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from .density import estimate_density
from .plot import plot_density
from .quotes import load_quotes, load_yfinance


def _summary(d, T_days: float) -> str:
    S0 = d.spot
    rows = [
        ("Expiry (days)", f"{T_days:.0f}"),
        ("Spot", f"{S0:,.2f}"),
        ("Forward", f"{d.forward:,.2f}"),
        ("ATM implied vol", f"{100 * d.atm_vol:.2f}%"),
        ("Integral before renorm", f"{d.raw_integral:.4f}"),
        ("Negative mass clipped", f"{abs(d.negative_mass):.2e}"),
        ("Mean", f"{d.mean:,.2f}"),
        ("Std dev", f"{d.std:,.2f}"),
        ("Skewness", f"{d.skew:.3f}"),
        ("Excess kurtosis", f"{d.excess_kurtosis:.3f}"),
    ]
    for f in (0.8, 0.9, 0.95):
        rows.append((f"P(S_T < {f:.2f} * S0)", f"{100 * d.prob_below(f * S0):.2f}%"))
    for f in (1.05, 1.10):
        rows.append((f"P(S_T > {f:.2f} * S0)", f"{100 * d.prob_above(f * S0):.2f}%"))
    w = max(len(k) for k, _ in rows)
    return "\n".join(f"{k:<{w}}  {v}" for k, v in rows)


def _fit(args) -> int:
    quotes = load_quotes(args.csv)
    expiries = sorted(quotes["expiry"].unique())
    if args.expiry:
        expiry = pd.Timestamp(args.expiry)
    elif len(expiries) == 1:
        expiry = expiries[0]
    else:
        print(
            "CSV has several expiries, pick one with --expiry: "
            + ", ".join(str(e.date()) for e in expiries),
            file=sys.stderr,
        )
        return 2
    quotes = quotes[quotes["expiry"] == expiry]
    asof = pd.Timestamp(args.asof) if args.asof else pd.Timestamp.today().normalize()
    days = (expiry - asof).days
    if days <= 0:
        print(
            f"expiry {expiry.date()} is not after the valuation date {asof.date()}", file=sys.stderr
        )
        return 2
    spot = args.spot
    if spot is None and "spot" in quotes.columns:
        spot = float(quotes["spot"].iloc[0])
    d = estimate_density(
        quotes,
        days / 365.0,
        args.rate,
        spot=spot,
        div_yield=args.div_yield,
        method=args.method,
        degree=args.degree,
    )
    print(_summary(d, days))
    if args.plot:
        plot_density(d, args.plot, title=f"Risk-neutral density, expiry {expiry.date()}")
        print(f"\nplot saved to {args.plot}")
    return 0


def _fetch(args) -> int:
    df = load_yfinance(args.ticker, args.expiry)
    df.to_csv(args.out, index=False)
    print(f"wrote {len(df)} quotes to {args.out}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="rnd", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fit", help="estimate the density from a CSV of option quotes")
    f.add_argument("csv")
    f.add_argument("--rate", type=float, required=True, help="continuously compounded rate")
    f.add_argument("--asof", help="valuation date YYYY-MM-DD (default: today)")
    f.add_argument("--expiry", help="expiry YYYY-MM-DD if the CSV holds several")
    f.add_argument("--spot", type=float, help="spot; default: from CSV 'spot' column or forward")
    f.add_argument(
        "--div-yield",
        type=float,
        default=0.0,
        help="only used to derive spot from the forward when --spot is absent",
    )
    f.add_argument("--method", choices=["poly", "spline"], default="poly")
    f.add_argument("--degree", type=int, default=5, help="polynomial degree (method=poly)")
    f.add_argument("--plot", help="save the density plot to this path")
    f.set_defaults(func=_fit)

    g = sub.add_parser("fetch", help="download one expiry from yfinance (needs rnd[data])")
    g.add_argument("ticker")
    g.add_argument("--expiry", help="YYYY-MM-DD, default: nearest")
    g.add_argument("--out", default="quotes.csv")
    g.set_defaults(func=_fetch)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
