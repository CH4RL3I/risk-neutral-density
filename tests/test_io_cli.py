from pathlib import Path

import pytest

from rnd.cli import main
from rnd.quotes import load_quotes

EXAMPLE = Path(__file__).parent.parent / "examples" / "spx_like.csv"


def test_loader_handles_mid_short_types_and_dead_quotes(tmp_path):
    p = tmp_path / "q.csv"
    p.write_text(
        "Strike,Type,Expiry,Bid,Ask\n"
        "100,C,2026-12-18,4.0,4.2\n"
        "100,p,2026-12-18,3.0,3.4\n"
        "150,call,2026-12-18,0.0,0.1\n"  # zero bid: dropped
    )
    q = load_quotes(p)
    assert list(q["type"]) == ["call", "put"]
    assert list(q["mid"]) == pytest.approx([4.1, 3.2])


def test_loader_requires_price_columns(tmp_path):
    p = tmp_path / "q.csv"
    p.write_text("strike,type,expiry\n100,call,2026-12-18\n")
    with pytest.raises(ValueError, match="mid"):
        load_quotes(p)


def test_cli_on_example_file(tmp_path, capsys):
    png = tmp_path / "out.png"
    code = main(
        [
            "fit",
            str(EXAMPLE),
            "--rate",
            "0.04",
            "--asof",
            "2026-09-28",
            "--spot",
            "5500",
            "--plot",
            str(png),
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "Skewness" in out and "P(S_T < 0.90 * S0)" in out
    assert png.stat().st_size > 10_000
