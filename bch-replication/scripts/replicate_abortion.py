"""BCH (2014) Table 2: effect of abortion on crime. Usage: python scripts/replicate_abortion.py [--force "regex,..."]"""

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bch import all_controls, double_selection, no_controls, single_selection

DATA = Path(os.environ.get("BCH_DATA_DIR", ROOT / "data" / "raw"))
OUTCOMES = {  # raw y, raw d, y and d with year effects partialled out, controls file
    "Violent crime": ("Dyv", "Dxv", "DyV", "DxV", "viol"),
    "Property crime": ("Dyp", "Dxp", "DyP", "DxP", "prop"),
    "Murder": ("Dym", "Dxm", "DyM", "DxM", "murd"),
}

ap = argparse.ArgumentParser()
ap.add_argument("--force", default="", help="regexes for controls to force in (PDS+)")
ap.add_argument("--out", default=str(ROOT / "results"))
a = ap.parse_args()

state = pd.read_csv(DATA / "levitt_state.csv")["state"].to_numpy()
raw, part = pd.read_csv(DATA / "levitt_linear.csv"), pd.read_csv(DATA / "levitt_partialled.csv")
se = {"se_type": "cluster", "groups": state, "intercept": False}
rows, picks = [], []

for label, (yr, dr, yp, dp, tag) in OUTCOMES.items():
    C = pd.read_csv(DATA / f"levitt_controls_{tag}.csv")
    names, X, y, d = C.columns, C.to_numpy(float), part[yp].to_numpy(), part[dp].to_numpy()
    fits = {
        "First difference": no_controls(raw[yr].to_numpy(), raw[dr].to_numpy(), **se),
        "All controls": all_controls(y, d, X, **se),
        "Single selection": single_selection(y, d, X, **se),
        "Double selection": double_selection(y, d, X, **se),
    }
    if a.force:
        force = [i for i, n in enumerate(names) if any(re.search(p, n) for p in a.force.split(","))]
        fits["Double selection+"] = double_selection(y, d, X, force=force, **se)

    for method, f in fits.items():
        rows.append({"outcome": label, "method": method, "estimate": f["alpha"], "se": f["se"],
                     "ci_low": f["alpha"] - 1.96 * f["se"], "ci_high": f["alpha"] + 1.96 * f["se"],
                     "n_controls": len(f["selected"]),
                     "n_I_y": len(f["I_y"]) if "I_y" in f else None,
                     "n_I_d": len(f["I_d"]) if "I_d" in f else None})
    ds = fits["Double selection"]
    for step, idx in [("I_y (predicts crime)", ds["I_y"]), ("I_d (predicts abortion)", ds["I_d"]),
                      ("single selection", fits["Single selection"]["selected"])]:
        picks += [{"outcome": label, "step": step, "control": names[i]} for i in idx]

Path(a.out).mkdir(exist_ok=True)
table = pd.DataFrame(rows)
table.to_csv(f"{a.out}/table2.csv", index=False)
pd.DataFrame(picks).to_csv(f"{a.out}/selected_controls.csv", index=False)
print(table[["outcome", "method", "estimate", "se", "n_controls"]].round(3).to_string(index=False))
