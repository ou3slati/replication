"""BCH (2014) Table 1 Monte Carlo. Usage: python scripts/simulate_table1.py --design 1 --reps 1000"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bch import double_selection, ols, single_selection

N, P, ALPHA = 100, 200, 0.5
CELLS = [(0.0, 0.2), (0.8, 0.2), (0.0, 0.8), (0.8, 0.8)]  # (R2 of y on x, R2 of d on x)
J = np.arange(1, P + 1)
SIGMA = 0.5 ** np.abs(J[:, None] - J[None, :])
CHOL = np.linalg.cholesky(SIGMA)


def scales(r2_y, r2_d, beta):
    """Constants that give the target population R^2's (unit-variance noise)."""
    B = beta @ SIGMA @ beta
    c_d = np.sqrt(r2_d / ((1 - r2_d) * B))
    c_y = np.sqrt(r2_y * (1 + ALPHA**2) / ((1 - r2_y) * B)) - ALPHA * c_d
    return c_y, c_d


def draw(design, r2_y, r2_d, rng):
    beta = 1 / J**2
    if design == 3:  # five decaying coefficients plus small random ones
        c_y, c_d = scales(r2_y, r2_d, np.where(J <= 5, beta, 0))
        rnd = rng.normal(0, np.sqrt(1 / P), size=(2, P))
        g, m = np.where(J <= 5, c_y * beta, rnd[0]), np.where(J <= 5, c_d * beta, rnd[1])
    else:
        c_y, c_d = scales(r2_y, r2_d, beta)
        g, m = c_y * beta, c_d * beta
    x = rng.standard_normal((N, P)) @ CHOL.T
    v, z = rng.standard_normal(N), rng.standard_normal(N)
    if design == 2:  # heteroskedastic errors
        s_d = np.abs(1 + x @ beta)
        d = x @ m + s_d / np.sqrt(np.mean(s_d**2)) * v
        s_y = np.abs(1 + ALPHA * d + x @ beta)
        y = ALPHA * d + x @ g + s_y / np.sqrt(np.mean(s_y**2)) * z
    else:
        d = x @ m + v
        y = ALPHA * d + x @ g + z
    return y, d, x, g, m


def oracle(y, d):
    b, s = ols(y, np.column_stack([d, np.ones(N)]))
    return {"alpha": b[0], "se": s[0]}


def estimates(y, d, x, g, m):
    return {
        "Oracle": oracle(y - x @ g, d),
        "Double-Selection Oracle": oracle(y - x @ g, d - x @ m),
        "Post-Lasso (single selection)": single_selection(y, d, x),
        "Double-Selection": double_selection(y, d, x),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", type=int, default=1, choices=[1, 2, 3])
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=5200)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    rows = []
    for r2_y, r2_d in CELLS:
        res = {}
        for _ in range(a.reps):
            for name, f in estimates(*draw(a.design, r2_y, r2_d, rng)).items():
                res.setdefault(name, []).append((f["alpha"], f["se"]))
        for name, v in res.items():
            est, se = np.array(v).T
            rows.append({"design": a.design, "R2_structure_y": r2_y, "R2_first_stage_d": r2_d,
                         "estimator": name, "rmse": np.sqrt(np.mean((est - ALPHA) ** 2)),
                         "rejection_rate": np.mean(np.abs(est - ALPHA) / se > 1.959964),
                         "reps": a.reps})
        print(f"done R2_y={r2_y}, R2_d={r2_d}", flush=True)

    df = pd.DataFrame(rows)
    Path(a.out).mkdir(exist_ok=True)
    df.to_csv(f"{a.out}/simulation_design{a.design}_table1.csv", index=False)
    print(df.pivot_table(index="estimator", columns=["R2_first_stage_d", "R2_structure_y"],
                         values="rejection_rate").round(3))


if __name__ == "__main__":
    main()
