# Detection Limits of Power-Domain NOMA: Information-Theoretic Bounds and Spatial Diversity

Reproducibility code for the paper

> **Liang Dong and Robert W. Heath Jr.**, "On the Detection Limits of
> Power-Domain NOMA: Information-Theoretic Bounds and Spatial Diversity,"
> *IEEE Transactions on Communications*, 2026 (under review).

The paper develops an information-theoretic framework for the detection limits
of power-domain NOMA under finite-alphabet modulation. It identifies the two
roles of power allocation (setting the SIC decoding order and shaping the
composite-constellation geometry), derives the constellation-constrained
mutual-information hierarchy
$I_{\mathrm{TIN}} \le I_{\mathrm{MAP}} \le I_{\mathrm{oracle}} \le \log_2 M$
with the structural-gain and oracle-deficit metrics, establishes Fano-type
lower and pairwise-error-probability upper bounds on the symbol error rate,
and extends the analysis to the MIMO-NOMA uplink, where a regime transition at
$N_r = K$ governs whether linear MMSE-SIC approaches the MAP bound. This
repository contains the simulation code that produces every numerical result
and figure in the manuscript.

## Contents

| File | Role |
|---|---|
| `sim_learned_mud.py` | Base SISO downlink NOMA library: constellations, Rayleigh channel, marginal-MAP / oracle / TIN detectors, mutual-information kernels, IEEE plotting style |
| `sim_mimo_noma.py` | Base MIMO uplink NOMA library: spatial channel, MMSE / MMSE-SIC / MAP / oracle detectors |
| `sim_it_bounds.py` | SISO information-theoretic bounds: CCMI hierarchy, structural gain and oracle deficit, Fano/PEP SER bounds, composite minimum-distance geometry, power-differentiation sweep |
| `sim_mimo_it_bounds.py` | MIMO information-theoretic bounds: MI hierarchy, SER, MAP-oracle gap vs. spatial diversity, the $N_r = K$ regime transition |
| `sim_revision1.py` | Revision additions: imperfect-CSI GMI, per-user fairness, and the power-allocation universality sweep across $K$ and modulation |
| `replot_it_bounds.py` | Regenerate the SISO figures from cached results, without re-running the simulation |
| `replot_mimo.py` | Regenerate the MIMO figures from cached results |
| `replot_revision1.py` | Regenerate the revision figures from cached results |
| `sim_it_bounds_results.pkl` | Cached SISO Monte Carlo results (MI/SER vs. SNR, power sweep) |
| `sim_mimo_results.pkl` | Cached MIMO Monte Carlo results (MI/SER vs. SNR for $N_r \in \{1,2,3,4\}$) |
| `revision1_results.json` | Cached imperfect-CSI, fairness, and power-universality results |

`sim_it_bounds.py` and `sim_revision1.py` import `sim_learned_mud`;
`sim_mimo_it_bounds.py` imports `sim_mimo_noma`. All four base/analysis modules
are bundled here, so the repository is self-contained.

## Requirements

- Python 3.9 or later
- NumPy, SciPy, Matplotlib, PyTorch (CUDA optional but strongly recommended)

```bash
pip install -r requirements.txt
```

## Quick start

```bash
# SISO bounds: CCMI hierarchy, information gaps, Fano/PEP SER bounds, power sweep
python sim_it_bounds.py

# MIMO bounds: MI hierarchy, SER, MAP-oracle gap, regime transition at N_r = K
python sim_mimo_it_bounds.py

# Revision additions: imperfect CSI, multi-user fairness, power-allocation universality
python sim_revision1.py
```

The cached `.pkl` / `.json` results are included, so the figures can be
regenerated in seconds **without a GPU**:

```bash
python replot_it_bounds.py
python replot_mimo.py
python replot_revision1.py
```

## Reproducibility notes

- All experiments use a fixed seed (`SEED = 42`), three users with the default
  power allocation `P = [0.2, 0.3, 0.5]`, i.i.d. Rayleigh block fading with unit
  variance, and a reference SNR of 16 dB for cross-comparisons.
- The SISO and MIMO Monte Carlo runs use 4,000,000 samples per SNR point; the
  revision runs use 2,000,000 (MI/SER) and 500,000 (power sweep) samples.
- The power-differentiation sweep of `sim_it_bounds.py` (Fig. "power_mi") and the
  $K=3$ curve of the universality sweep in `sim_revision1.py` share the identical
  allocation path through the default `[0.2, 0.3, 0.5]`, so they coincide to
  within Monte Carlo error.
- Figures are written as EPS and PDF with Type-42 (TrueType) fonts for IEEE
  production. They are not tracked in the repository (see `.gitignore`); run the
  `replot_*.py` scripts to regenerate them.
- The 16-QAM, $K=4$ marginal-MAP case enumerates $16^4 = 65536$ composite
  hypotheses and is skipped on an 8 GB GPU.

## License

Released under the MIT License. See [LICENSE](LICENSE).
