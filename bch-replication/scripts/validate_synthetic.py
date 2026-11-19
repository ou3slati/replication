"""Known-truth check on data shaped like the abortion application (not a Table 2 result)."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bch import all_controls, double_selection, no_controls, ols, single_selection

STATES, YEARS, P, ALPHA = 48, 12, 284, -0.1
N = STATES * YEARS
J = np.arange(P)
SIGMA = 0.5 ** np.abs(J[:, None] - J[None, :])
BETA = 1 / (J + 1.0) ** 2
B = BETA @ SIGMA @ BETA
R2_D, R2_Y = 0.9, 0.05  # how well x predicts d and y
C_D = np.sqrt(R2_D / ((1 - R2_D) * B))
M, G = C_D * BETA, (np.sqrt(R2_Y * (1 + ALPHA**2) / ((1 - R2_Y) * B)) - ALPHA * C_D) * BETA
CHOL, STATE = np.linalg.cholesky(SIGMA), np.repeat(np.arange(STATES), YEARS)

ap = argparse.ArgumentParser()
ap.add_argument("--reps", type=int, default=500)
ap.add_argument("--seed", type=int, default=5200)
ap.add_argument("--out", default=str(ROOT / "results"))
a = ap.parse_args()
rng = np.random.default_rng(a.seed)
se = {"se_type": "cluster", "groups": STATE}
res = {}

for _ in range(a.reps):
    x = rng.standard_normal((N, P)) @ CHOL.T
    shock = lambda: np.sqrt(0.5) * (rng.standard_normal(STATES)[STATE] + rng.standard_normal(N))
    v, z = shock(), shock()  # half of each error is a state-level shock
    d = x @ M + v
    y = ALPHA * d + x @ G + z
    b, s = ols(y - x @ G, np.column_stack([d, np.ones(N)]), **se)
    fits = {"No controls": no_controls(y, d, **se), "All controls": all_controls(y, d, x, **se),
            "Single selection": single_selection(y, d, x, **se),
            "Double selection": double_selection(y, d, x, **se),
            "Oracle (knows g)": {"alpha": b[0], "se": s[0], "selected": []}}
    for k, f in fits.items():
        res.setdefault(k, []).append((f["alpha"], f["se"], len(f["selected"])))

rows = []
for k, v in res.items():
    est, s, n = np.array(v).T
    rows.append({"estimator": k, "true_alpha": ALPHA, "mean_estimate": est.mean(),
                 "bias": est.mean() - ALPHA, "rmse": np.sqrt(np.mean((est - ALPHA) ** 2)),
                 "coverage_95": np.mean(np.abs(est - ALPHA) <= 1.96 * s),
                 "mean_controls": n.mean(), "reps": a.reps})
df = pd.DataFrame(rows)
Path(a.out).mkdir(exist_ok=True)
df.to_csv(f"{a.out}/synthetic_validation.csv", index=False)
print(df[["estimator", "bias", "rmse", "coverage_95", "mean_controls"]].round(3).to_string(index=False))
