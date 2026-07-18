"""
Re-plot the Paper 1 second-round revision (R2) figures from saved results,
WITHOUT re-running the GPU simulation.

Usage (from the repository root):
    NOMA_OUT_DIR=. python replot_revision2.py

Loads:
    sim_revision2_corr.pkl                    -> fig_mimo_correlation   (Sec. VII-F)
    sim_revision2_ep.pkl + sim_mimo_results.pkl -> fig_mimo_ser_comparison (Sec. VI, EP curve)

Also prints the deterministic nearest-neighbor multiplicity table (Part A of
sim_revision2.py), which needs no simulation and no GPU.

Only the figures are produced here; the numbers are exactly those from the
rigorous run of sim_revision2.py.
"""
import os
import pickle

from sim_revision2 import (
    OUT_DIR, run_part_A, plot_correlation, plot_ser_with_ep,
)

CORR = os.path.join(OUT_DIR, 'sim_revision2_corr.pkl')
EP = os.path.join(OUT_DIR, 'sim_revision2_ep.pkl')
MIMO = os.path.join(OUT_DIR, 'sim_mimo_results.pkl')


def main():
    # Part A: deterministic nearest-neighbor multiplicity factor (no GPU).
    run_part_A()

    # Part B: correlated-channel MI figure from cache.
    if os.path.exists(CORR):
        with open(CORR, 'rb') as f:
            d = pickle.load(f)
        plot_correlation(d['snr'], d['out'], d['r_vals'],
                         os.path.join(OUT_DIR, 'fig_mimo_correlation.eps'))
    else:
        print(f"[skip] {CORR} not found; run sim_revision2.py to generate it.")

    # Part C: EP SER figure from cache, with the cached base detector curves.
    if os.path.exists(EP) and os.path.exists(MIMO):
        with open(EP, 'rb') as f:
            ep = pickle.load(f)
        with open(MIMO, 'rb') as f:
            base = pickle.load(f)
        plot_ser_with_ep(base['all_snr'], base['all_results'], ep['ep_ser'],
                         os.path.join(OUT_DIR, 'fig_mimo_ser_comparison.eps'))
    else:
        print(f"[skip] {EP} or {MIMO} not found; run sim_revision2.py.")

    print("Re-plotted R2 figures from cache (no simulation run).")


if __name__ == '__main__':
    main()
