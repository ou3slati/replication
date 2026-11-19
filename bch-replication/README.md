# Replicating Belloni, Chernozhukov & Hansen (2014)

STAT 5200 paper replication of

> Belloni, A., Chernozhukov, V. and Hansen, C. (2014). "Inference on Treatment
> Effects after Selection among High-Dimensional Controls." *Review of Economic
> Studies* 81(2): 608–650.

The repo implements post-double-selection from scratch in Python, including
the paper’s rigorous Lasso, and uses it to replicate:

1. **The Monte Carlo study** (Section 6, Table 1): single selection versus
   double selection versus infeasible oracles.
2. **The abortion and crime application** (Table 2): Donohue and Levitt (2001)
   re-estimated with data-driven selection among 284 candidate controls.

## Quick start

```bash
pip install -r requirements.txt

python tests/test_bch.py                          # sanity tests, a few seconds

python scripts/simulate_table1.py --design 1      # Table 1, Design 1 (~1 min)
python scripts/simulate_table1.py --design 2
python scripts/simulate_table1.py --design 3

python scripts/get_data.py                        # download the panel (once)
python scripts/replicate_abortion.py              # Table 2

python scripts/validate_synthetic.py              # known-truth check (~1 min)
```

On macOS use `python3`, or work inside a virtual environment:
`python3 -m venv .venv && source .venv/bin/activate`.

Results land in `results/`.

## Layout

```
bch/
  rlasso.py        rigorous Lasso: BCH penalty level + iterated loadings
  ols.py           OLS with state-clustered (CR1) and HC3 standard errors
  estimators.py    no controls, all controls, single selection, double selection
scripts/
  simulate_table1.py     Monte Carlo, Designs 1–3
  get_data.py            downloads the abortion–crime panel, checks shapes
  replicate_abortion.py  Table 2 estimates + which controls were selected
tests/test_bch.py
results/               output CSVs
```

## Implementation

**Selection step (`bch/rlasso.py`).** Each Lasso minimizes
E_n[(y − x′β)²] + (λ/n) Σ ψ_j |β_j| with
λ = 2c√n Φ⁻¹(1 − γ/2p), c = 1.1, γ = 0.05 (the paper’s recommendation), and
heteroskedasticity-robust loadings ψ_j = √E_n[x_j² e²] updated from post-Lasso
residuals (Appendix A). Lasso only picks variables; estimates always come from
OLS.

**Single selection (`single_selection`).** Lasso of y on (d, x) with d left
unpenalized, as in the paper’s simulations. By Frisch–Waugh this equals a Lasso
of M_d y on M_d x, which is how it is computed.

**Double selection (`double_selection`).** Lasso of y on x, Lasso of d on x,
then OLS of y on d and the union. Optional `force` indices add extra
controls in (the paper’s “Post-Double-Selection+”).

**Standard errors.** Clustered by state (CR1) for the application, HC3
(MacKinnon–White jackknife) for the simulations, matching the paper.

## Simulation results (1,000 replications, seed 5200)

Rejection rate of a nominal 5% test of the true α (should be ≈ 0.05):

| Design 1 | d R² .2, y R² 0 | d R² .2, y R² .8 | d R² .8, y R² 0 | d R² .8, y R² .8 |
|---|---|---|---|---|
| Single selection, ours | 0.187 | 0.074 | 0.618 | 0.468 |
| Single selection, paper | 0.205 | 0.064 | 0.987 | 0.974 |
| Double selection, ours | 0.072 | 0.061 | 0.059 | 0.047 |
| Double selection, paper | 0.063 | 0.058 | 0.074 | 0.062 |
| Oracle, ours | 0.057 | 0.059 | 0.046 | 0.047 |

Full output, including RMSE and Designs 2–3, is in
`results/simulation_design*_table1.csv`.

## Abortion and crime results (284 controls, state-clustered SEs)

| Outcome | Paper draft (PDS) | Ours: double selection | Ours: single selection | Controls in final OLS |
|---|---|---|---|---|
| Violent crime | −0.087 (0.181) | −0.188 (0.120) | −0.155 (0.033) | 14 |
| Property crime | −0.094 (0.051) | −0.041 (0.069) | −0.040 (0.024) | 16 |
| Murder | 0.006 (0.280) | −0.116 (0.146) | −0.206 (0.051) | 8 |

- With double selection every 95% CI includes zero, as in the paper.
- The initial abortion rate is selected for every outcome (`xV0`, `xP0`,
  `xM0` and `xM0*t`), the paper’s explanation for the imprecision.
- Selection is driven by the abortion equation (13, 13, 8 controls) far more
  than the crime equation (1, 3, 0): the regime where double selection matters.
- Single selection keeps no controls for violent crime or murder and so
  reproduces Donohue and Levitt’s significant effects. The selection method
  alone changes the conclusion.

The paper column is the 2012 working-paper version (251 controls). The
published Table 2 uses 284 controls; compare against it once available. Full output: `results/table2.csv`,
`results/selected_controls.csv`.

## Synthetic validation (known truth)

`scripts/validate_synthetic.py` keeps the application’s shape (48 states × 12
years, 284 controls, state-level shocks, R² of 0.9 for the treatment and 0.05
for the outcome) but plants a known effect α = −0.1. 500 replications:

| Estimator | Bias | 95% CI coverage | Controls kept |
|---|---|---|---|
| Double selection | 0.007 | 0.928 | 5.4 |
| Single selection | 0.126 | 0.216 | 0.3 |
| All controls | 0.006 | 0.888 | 284 |
| No controls | 0.160 | 0.000 | 0 |
| Oracle | 0.000 | 0.952 | — |

In data shaped like the application, single selection behaves as it did on
the real data (it keeps almost no controls) and its intervals miss the truth
almost 80% of the time; double selection is close to nominal. These are code
checks, not Table 2 results.

## Issues found while replicating

1. **Starting the penalty loadings.** Following Appendix A literally (initial
   residuals from an intercept-only fit) made the Lasso of d on x select
   nothing two-thirds of the time when the first stage is weak. That dropped the
   main confounder, and double selection rejected 12.6% instead of ~6%.
   Starting from the 5 regressors most correlated with the outcome, as the
   authors’ `hdm` R package does, fixes it. Appendix A allows any small initial
   set, so this is still within the paper’s algorithm.
2. **How badly single selection fails depends on implementation details.** In
   our runs it over-rejects 47–62% of the time with a strong first stage, against
   97–99% in the paper. The conclusion is the same (it is badly invalid exactly
   where theory says), but the magnitude is sensitive to tuning.
3. **Design 3 is under-specified.** The text says the 195 remaining
   coefficients are random; the Table 1 note says 95. We follow the text. Our
   double selection rejects 13.8% in one cell versus 5.5% in the paper, which
   may reflect that difference.
4. **Draft versus published paper.** The arXiv draft (v3, 2012) uses 251
   candidate controls in the application; the published version and its
   replication data use 284, and Table 2 differs. Compare against the
   **published** Table 2.
5. **The control set is not listed in the paper.** We use CSV extracts of the
   replication archive (see `scripts/get_data.py` for the source). If possible,
   verify them against the original archive.
6. **Selection counts differ from an `hdm`-based replication.** Our Lasso of
   the abortion rate keeps 13, 13 and 8 controls; Fitzgerald et al. (2026),
   using the authors’ `hdm` package, report 8, 9 and 9. The estimates differ in
   size but every double-selection interval still includes zero.
7. **Post-Double-Selection+ not yet run.** It needs the column names of
   Donohue and Levitt’s 8 original controls; pass them with `--force`.

## Data

`scripts/get_data.py` downloads six CSVs extracted from the BCH replication
archive by Carlos Mendez (posted with his Double LASSO tutorials). The panel is
48 states × 12 years of first differences (n = 576) with 284 controls per
crime outcome; year effects are already partialled out. Original sources: the
ReStud supplementary material, and the JEP 2014 replication package on
openICPSR (project 113924).

## References

- Belloni, Chernozhukov & Hansen (2014), *Review of Economic Studies* 81(2).
- Belloni, Chen, Chernozhukov & Hansen (2012), *Econometrica* 80(6) (rigorous
  Lasso and penalty loadings).
- Donohue & Levitt (2001), *Quarterly Journal of Economics* 116(2).
- Chernozhukov, Hansen & Spindler, `hdm` R package (reference implementation).
- Fitzgerald, Lattimore, Robinson & Zhu (2026), "Double LASSO: Replication and
  Practical Insights," *Journal of Applied Econometrics*.
