"""
Re-plot the Paper 1 SISO figures from saved Monte Carlo results, WITHOUT
re-running the simulation.

Usage (from paper1_it_bounds_r1/):
    PYTHONPATH=.. NOMA_OUT_DIR=. python replot_it_bounds.py

Workflow for future figure-format changes:
    1. Edit the plotting functions (fonts, sizes, colors, ...) in sim_it_bounds.py.
    2. Run this script. It loads sim_it_bounds_results.pkl (created by the last
       full run of sim_it_bounds.py) and regenerates every figure instantly.

Only the formatting changes; the underlying numbers are exactly those produced
by the rigorous simulation, so the figures stay faithful to the results.
"""
import os
import pickle

# Importing sim_it_bounds pulls in the plotting functions and the module-level
# matplotlib/Type-42 configuration; it does NOT run the simulation (guarded by
# __main__). Needs PYTHONPATH=.. so the shared base sim resolves.
from sim_it_bounds import (
    OUT_DIR,
    plot_mi_comparison,
    plot_mi_gaps,
    plot_error_bounds,
    plot_fa_capacity_gap,
    plot_power_mi,
    plot_constellation_distance,
)

CACHE = os.path.join(OUT_DIR, 'sim_it_bounds_results.pkl')


def main():
    if not os.path.exists(CACHE):
        raise SystemExit(
            f"Results cache not found: {CACHE}\n"
            "Run a full simulation first:  "
            "PYTHONPATH=.. NOMA_OUT_DIR=. python sim_it_bounds.py")

    with open(CACHE, 'rb') as f:
        data = pickle.load(f)
    all_snr = data['all_snr']
    all_results = data['all_results']
    power_results = data['power_results']
    print(f"Loaded results from {CACHE} "
          f"(n_samples={data.get('n_samples')}, P={data.get('P')}, "
          f"SEED={data.get('SEED')})")

    # Figure 1: MI comparison
    plot_mi_comparison(all_snr, all_results,
                       os.path.join(OUT_DIR, 'fig_it_mi_comparison.eps'))
    # Figure 2: MI gaps
    plot_mi_gaps(all_snr, all_results,
                 os.path.join(OUT_DIR, 'fig_it_mi_gaps.eps'))
    # Figure 3: Error bounds
    plot_error_bounds(all_snr, all_results,
                      os.path.join(OUT_DIR, 'fig_it_error_bounds.eps'))
    # Figure 4: Finite-alphabet capacity gap
    plot_fa_capacity_gap(all_snr, all_results,
                         os.path.join(OUT_DIR, 'fig_it_fa_capacity_gap.eps'))
    # Figure 5: Power-MI analysis
    plot_power_mi(power_results,
                  os.path.join(OUT_DIR, 'fig_it_power_mi.eps'))
    # Figure 6: Constellation distance analysis (analytical, recomputed)
    plot_constellation_distance(
        os.path.join(OUT_DIR, 'fig_it_constellation_distance.eps'))

    print("Re-plotted all figures from cache (no simulation run).")


if __name__ == '__main__':
    main()
