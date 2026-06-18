"""
Re-plot the Paper 1 R1-addition figures (imperfect CSI, multi-user fairness,
power-allocation universality) from the saved results in revision1_results.json,
WITHOUT re-running the simulation.

Usage (from paper1_it_bounds_r1/):
    PYTHONPATH=.. NOMA_OUT_DIR=. python replot_revision1.py

Workflow for figure-format changes:
    1. Edit the plot_* functions (fonts, sizes, legends, ...) in sim_revision1.py.
    2. Run this script. It loads revision1_results.json (written by the last full
       run of sim_revision1.py) and regenerates the figures instantly.

The numbers are exactly those produced by the rigorous simulation; only the
formatting changes.
"""
import os
import json

from sim_revision1 import (
    OUT,
    plot_imperfect_csi,
    plot_fairness,
    plot_power_universality,
)

CACHE = os.path.join(OUT, 'revision1_results.json')


def main():
    if not os.path.exists(CACHE):
        raise SystemExit(
            f"Results cache not found: {CACHE}\n"
            "Run a full simulation first:  "
            "PYTHONPATH=.. NOMA_OUT_DIR=. python sim_revision1.py")

    with open(CACHE) as f:
        d = json.load(f)
    print(f"Loaded results from {CACHE} (keys: {list(d.keys())})")

    # Each is guarded so a problem with one figure does not block the others.
    jobs = [
        ('power_universality', lambda: plot_power_universality(
            d['power_universality'],
            os.path.join(OUT, 'fig_it_power_universality.eps'))),
        ('fairness', lambda: plot_fairness(
            d['fairness'], 'QPSK', os.path.join(OUT, 'fig_it_fairness.eps'))),
        ('imperfect_csi', lambda: plot_imperfect_csi(
            d['csi'], 'QPSK', os.path.join(OUT, 'fig_it_csi.eps'))),
    ]
    for name, fn in jobs:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            print(f"  WARNING: could not replot {name}: {exc}")

    print("Re-plotted revision-1 figures from cache (no simulation run).")


if __name__ == '__main__':
    main()
