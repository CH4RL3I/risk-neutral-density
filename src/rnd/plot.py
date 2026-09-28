"""Density-vs-lognormal figure plus the fitted smile."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .density import Density  # noqa: E402


def plot_density(d: Density, path, title: str | None = None) -> None:
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.5, 1]})
    blue, grey, red = "#1f5fa8", "#7a7a7a", "#c0392b"

    ln = d.lognormal()
    visible = d.pdf > 0.002 * d.pdf.max()
    lo, hi = d.strikes[visible].min(), d.strikes[visible].max()
    m = (d.strikes >= lo) & (d.strikes <= hi)
    ax.plot(d.strikes[m], d.pdf[m], color=blue, lw=2, label="Implied (Breeden-Litzenberger)")
    ax.plot(d.strikes[m], ln[m], color=grey, lw=1.6, ls="--", label="Lognormal, same ATM vol")
    ax.axvline(d.forward, color=red, lw=1, alpha=0.7, label=f"Forward {d.forward:,.0f}")
    ax.fill_between(d.strikes[m], d.pdf[m], ln[m], where=d.pdf[m] > ln[m], color=blue, alpha=0.12)
    ax.set_xlabel("$S_T$")
    ax.set_ylabel("risk-neutral density")
    ax.set_title(title or f"Risk-neutral density, T = {d.T * 365:.0f} days")
    ax.legend(frameon=False, fontsize=9)

    k = np.linspace(d.points.k.min() - 0.05, d.points.k.max() + 0.05, 300)
    ax2.plot(d.forward * np.exp(k), 100 * d.smile(k), color=blue, lw=2, label="Fitted smile")
    ax2.scatter(d.points.strikes, 100 * d.points.iv, s=12, color=grey, label="OTM quotes")
    ax2.set_xlabel("strike")
    ax2.set_ylabel("implied vol (%)")
    ax2.set_title("Implied-vol smile")
    ax2.legend(frameon=False, fontsize=9)

    for a in (ax, ax2):
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
