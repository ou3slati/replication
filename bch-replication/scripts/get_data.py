"""Download the abortion-crime panel (CSV extracts of the BCH replication archive, via Carlos Mendez)."""

import urllib.request
from pathlib import Path

import pandas as pd

URL = "https://raw.githubusercontent.com/cmg777/starter-academic-v501/master/content/tutorials/r_double_lasso/data/"
FILES = ["levitt_state.csv", "levitt_linear.csv", "levitt_partialled.csv",
         "levitt_controls_viol.csv", "levitt_controls_prop.csv", "levitt_controls_murd.csv"]
DATA = Path(__file__).resolve().parents[1] / "data" / "raw"

DATA.mkdir(parents=True, exist_ok=True)
for f in FILES:
    if not (DATA / f).exists():
        urllib.request.urlretrieve(URL + f, DATA / f)
    print(f, pd.read_csv(DATA / f).shape)
