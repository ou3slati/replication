"""Run with: python tests/test_bch.py"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bch import double_selection, ols, rlasso, single_selection

rng = np.random.default_rng(0)


def test_rlasso_selects_signal_not_noise():
    X = rng.standard_normal((300, 100))
    y = X[:, [0, 5, 9]] @ [1.0, -0.8, 0.6] + rng.standard_normal(300)
    sel = set(rlasso(X, y))
    assert {0, 5, 9} <= sel and len(sel) <= 6
    assert len(rlasso(X, rng.standard_normal(300))) <= 2


def test_ols_hc3():
    X = np.column_stack([rng.standard_normal(2000), np.ones(2000)])
    b, s = ols(X @ [2.0, 1.0] + rng.standard_normal(2000), X)
    assert abs(b[0] - 2) < 0.1 and abs(s[0] - 1 / np.sqrt(2000)) < 0.003


def test_double_beats_single_with_hidden_confounder():
    err_s, err_d = [], []
    for _ in range(30):
        X = rng.standard_normal((200, 100))
        d = 1.5 * X[:, 0] + rng.standard_normal(200)
        y = 0.5 * d + 0.25 * X[:, 0] + rng.standard_normal(200)
        err_s.append(single_selection(y, d, X)["alpha"] - 0.5)
        err_d.append(double_selection(y, d, X)["alpha"] - 0.5)
    assert abs(np.mean(err_d)) < 0.05 < abs(np.mean(err_s))


def test_abortion_script_on_fake_data():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        pd.DataFrame({"state": np.repeat(np.arange(48), 12)}).to_csv(tmp / "levitt_state.csv", index=False)
        cols = {}
        for tag, (y_, d_) in {"viol": ("v", "V"), "prop": ("p", "P"), "murd": ("m", "M")}.items():
            X = rng.standard_normal((576, 284))
            d = X[:, 0] + rng.standard_normal(576)
            y = -0.1 * d + 0.3 * X[:, 1] + rng.standard_normal(576)
            cols.update({f"Dy{y_}": y, f"Dx{y_}": d, f"Dy{d_}": y, f"Dx{d_}": d})
            pd.DataFrame(X, columns=[f"c{tag}{j}" for j in range(284)]).to_csv(
                tmp / f"levitt_controls_{tag}.csv", index=False)
        pd.DataFrame(cols).to_csv(tmp / "levitt_linear.csv", index=False)
        pd.DataFrame(cols).to_csv(tmp / "levitt_partialled.csv", index=False)
        subprocess.run([sys.executable, ROOT / "scripts/replicate_abortion.py", "--out", tmp, "--force", "2$"],
                       check=True, capture_output=True, env={**os.environ, "BCH_DATA_DIR": str(tmp)})
        t = pd.read_csv(tmp / "table2.csv")
        assert len(t) == 15
        assert (t[t.method == "Double selection"].estimate + 0.1).abs().max() < 0.15


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print("ok", name)
