"""
Re-plot the Paper 1 MIMO figures from saved Monte Carlo results, WITHOUT
re-running the simulation.

Usage (from paper1_it_bounds_r1/):
    PYTHONPATH=.. NOMA_OUT_DIR=. python replot_mimo.py

Workflow for figure-format changes:
    1. Edit the plot_mimo_* functions (fonts, sizes, legends, ...) in
       sim_mimo_it_bounds.py.
    2. Run this script. It loads sim_mimo_results.pkl (written by the last full
       run of sim_mimo_it_bounds.py) and regenerates the figures instantly.

Only the formatting changes; the numbers are exactly those produced by the
rigorous simulation.
"""
import os
import pickle

from sim_mimo_it_bounds import (
    OUT_DIR,
    plot_mimo_mi_comparison,
    plot_mimo_ser_comparison,
    plot_mimo_diversity_gain,
    plot_mimo_nr_mi_tradeoff,
)

CACHE = os.path.join(OUT_DIR, 'sim_mimo_results.pkl')


def main():
    if not os.path.exists(CACHE):
        raise SystemExit(
            f"Results cache not found: {CACHE}\n"
            "Run a full simulation first:  "
            "PYTHONPATH=.. NOMA_OUT_DIR=. python sim_mimo_it_bounds.py")

    with open(CACHE, 'rb') as f:
        d = pickle.load(f)
    all_snr = d['all_snr']
    all_results = d['all_results']
    print(f"Loaded results from {CACHE} "
          f"(n_samples={d.get('n_samples')}, mod={d.get('mod_type')})")

    plot_mimo_mi_comparison(
        all_snr[2], all_results[2], all_snr[4], all_results[4],
        os.path.join(OUT_DIR, 'fig_mimo_mi_comparison.eps'))
    plot_mimo_ser_comparison(
        all_snr[2], all_results[2], all_snr[4], all_results[4],
        os.path.join(OUT_DIR, 'fig_mimo_ser_comparison.eps'))
    plot_mimo_diversity_gain(
        {2: all_results[2], 4: all_results[4]},
        {2: all_snr[2], 4: all_snr[4]},
        os.path.join(OUT_DIR, 'fig_mimo_diversity_gain.eps'))
    plot_mimo_nr_mi_tradeoff(
        all_results, all_snr,
        os.path.join(OUT_DIR, 'fig_mimo_nr_mi_tradeoff.eps'))

    print("Re-plotted MIMO figures from cache (no simulation run).")


if __name__ == '__main__':
    main()
