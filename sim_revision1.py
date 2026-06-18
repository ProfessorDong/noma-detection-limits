#!/usr/bin/env python
"""
Paper 1 (IT Bounds) -- Revision 1 additional simulations.

Generates the three new figures requested by the reviewers, reusing the
exact compute kernels of sim_it_bounds.py so that all numbers are consistent
with the original submission.

  fig_it_csi.eps            -- R4#7 / Editor#1: impact of imperfect CSI on
                               I_MAP and SER (User 1).
  fig_it_fairness.eps       -- R4#8: per-user (1,2,3) I_MAP and SER vs SNR.
  fig_it_power_universality.eps -- R4#5 / Editor#2: I_MAP vs power
                               differentiation for K in {2,3,4} and
                               BPSK / QPSK / 16-QAM (non-monotonic peak).

Run (GPU):
  NOMA_OUT_DIR=. PYTHONUNBUFFERED=1 python sim_revision1.py

The finite-alphabet capacity-gap figure (R4#6) is produced by the existing
plot_fa_capacity_gap() path in sim_it_bounds.py and is not duplicated here.
"""
import os
import sys
import json
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # base sims (sim_learned_mud, sim_mimo_noma) are bundled alongside this file

# NOTE: sim_learned_mud and sim_it_bounds each reopen sys.stdout at import.
# Importing sim_it_bounds first runs that chain exactly once; we then grab the
# already-loaded sim_learned_mud module from sys.modules (no re-execution) so
# that we can override its global K for the variable-user-count sweep.
from sim_it_bounds import (
    get_constellation, generate_noma_data, setup_ieee_style,
    compute_log_marginal_posteriors, compute_ccmi, compute_oracle_mi,
    compute_tin_mi, detect_exact_map, detect_oracle_sic, detect_conv_sic,
    run_mi_simulation, P_DEFAULT, SEED,
)
base = sys.modules['sim_learned_mud']

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = os.environ.get('NOMA_OUT_DIR', '.')
COLORS = {'BPSK': '#1f77b4', 'QPSK': '#ff7f0e', '16QAM': '#2ca02c'}
USER_COLORS = ['#1f77b4', '#d62728', '#2ca02c']

# Rigorous Monte Carlo sizes (overridable by env for quick checks).
N_MI = int(os.environ.get('NOMA_N_MI', 2_000_000))        # MI / SER per point
N_POW = int(os.environ.get('NOMA_N_POW', 500_000))        # power-sweep per point (matches Fig power_mi for K=3 consistency)


def _ieee_style():
    """IEEE style with Type-42 (TrueType) fonts to avoid production rejection."""
    setup_ieee_style()
    plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42})


def _save(fig, filename):
    """Save EPS + PDF companion (LaTeX picks PDF if no extension is given)."""
    fig.savefig(filename, format='eps')
    fig.savefig(filename.replace('.eps', '.pdf'), format='pdf')
    fig.savefig(filename.replace('.eps', '.png'), format='png', dpi=300)
    plt.close(fig)
    print('  Saved:', filename, '(+ .pdf, .png)')


def _map_ser_from_data(y, h_est, user_k, S, sigma2, P_arr, x_k_idx):
    pred = detect_exact_map(y, h_est, user_k, S, sigma2, P_arr)
    return float(np.mean(pred != x_k_idx))


def _gmi_from_post(post, x_k_idx, M):
    """Generalized MI of a (possibly mismatched) marginal-MAP decoder:
        GMI = log2(M) + (1/n) sum_i log2 q(x_true_i | y_i).
    Under matched CSI this equals the CCMI log2(M) - H; under CSI mismatch it
    is the correct achievable rate and never exceeds the perfect-CSI MI.
    """
    q_true = np.clip(post[np.arange(len(x_k_idx)), x_k_idx], 1e-30, 1.0)
    return float(np.log2(M) + np.mean(np.log2(q_true)))


# ----------------------------------------------------------------------
# R4#7 / Editor#1 : Imperfect CSI
# ----------------------------------------------------------------------
def sim_imperfect_csi(mod_type='QPSK', user_k=0, P_arr=None,
                      n_samples=500000):
    """I_MAP and MAP-SER vs SNR for several CSI error variances.

    The receiver detects with the *estimated* channel h_est = h_true + e,
    e ~ CN(0, csi_var), modelling MMSE channel estimation error. The MI uses
    the mismatched posterior built from h_est, lower-bounding the true MI.
    """
    if P_arr is None:
        P_arr = list(P_DEFAULT)
    P_arr = np.array(P_arr)
    S = get_constellation(mod_type)
    M = len(S)
    snr_range = np.arange(0, 31, 2)
    csi_vars = [0.0, 0.01, 0.05, 0.10]   # estimation error variance

    res = {'snr': snr_range.tolist(), 'csi_vars': csi_vars,
           'I_MAP': {}, 'MAP_SER': {}}
    print('\n=== Imperfect CSI (mod=%s, User %d) ===' % (mod_type, user_k + 1))
    for cv in csi_vars:
        I_list, ser_list = [], []
        for snr in snr_range:
            np.random.seed(SEED + int(snr * 100))
            y, h_est, h_true, x_k, x_all, x_k_idx, sigma2 = generate_noma_data(
                n_samples, int(snr), mod_type, user_k, P_arr, csi_error_var=cv)
            _, post = compute_log_marginal_posteriors(
                y, h_est, user_k, S, sigma2, P_arr)
            # GMI of the mismatched MAP decoder (uses estimated channel h_est in
            # its metric but is scored at the true symbol). Valid achievable
            # rate under CSI error; reduces to I_MAP when the CSI is perfect.
            I_map = _gmi_from_post(post, x_k_idx, M)
            ser = _map_ser_from_data(y, h_est, user_k, S, sigma2, P_arr, x_k_idx)
            I_list.append(float(I_map))
            ser_list.append(max(ser, 1e-6))
            print('  csi_var=%.2f SNR=%2d : I_MAP=%.4f  SER=%.3e'
                  % (cv, snr, I_map, ser))
        res['I_MAP'][str(cv)] = I_list
        res['MAP_SER'][str(cv)] = ser_list
    return res


def plot_imperfect_csi(res, mod_type, filename):
    _ieee_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    snr = np.array(res['snr'])
    for cv in res['csi_vars']:
        lbl = r'$\sigma_e^2=%.2f$' % cv if cv > 0 else 'perfect CSI'
        axes[0].plot(snr, res['I_MAP'][str(cv)], 'o-', markersize=3,
                     linewidth=1.2, label=lbl)
        axes[1].semilogy(snr, res['MAP_SER'][str(cv)], 'o-', markersize=3,
                         linewidth=1.2, label=lbl)
    axes[0].set_xlabel('SNR (dB)', fontsize=12)
    axes[0].set_ylabel(r'GMI (bits)', fontsize=12)
    axes[0].set_title('Mismatched-decoder GMI', fontsize=12)
    axes[0].axhline(0.0, color='gray', linestyle=':', linewidth=0.8)
    axes[0].set_ylim(-2.0, 2.1)   # GMI diverges below this for large error
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=10, framealpha=0.5, loc='lower left')
    axes[1].set_xlabel('SNR (dB)', fontsize=12)
    axes[1].set_ylabel(r'MAP SER', fontsize=12)
    axes[1].set_title('Symbol Error Rate', fontsize=12)
    axes[1].grid(True, alpha=0.3, which='both')
    axes[1].legend(fontsize=10, framealpha=0.5)
    plt.tight_layout()
    _save(fig, filename)


# ----------------------------------------------------------------------
# R4#8 : Multi-user fairness (Users 1,2,3)
# ----------------------------------------------------------------------
def sim_fairness(mod_type='QPSK', P_arr=None, n_samples=500000):
    """I_MAP and MAP-SER vs SNR for every user under the default allocation."""
    if P_arr is None:
        P_arr = list(P_DEFAULT)
    res = {'P': list(P_arr), 'users': {}}
    print('\n=== Multi-user fairness (mod=%s, P=%s) ===' % (mod_type, P_arr))
    for k in range(len(P_arr)):
        snr_range, r = run_mi_simulation(mod_type, user_k=k,
                                         P_arr=np.array(P_arr),
                                         n_samples=n_samples)
        res['snr'] = snr_range.tolist()
        res['users'][str(k)] = {
            'I_MAP': r['I_MAP'].tolist(),
            'MAP_SER': r['MAP_SER'].tolist(),
            'I_oracle': r['I_oracle'].tolist(),
        }
    return res


def plot_fairness(res, mod_type, filename):
    _ieee_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    snr = np.array(res['snr'])
    P = res['P']
    for k in sorted(res['users'].keys(), key=int):
        ki = int(k)
        lbl = r'User %d ($P_%d=%.1f$)' % (ki + 1, ki + 1, P[ki])
        c = USER_COLORS[ki % len(USER_COLORS)]
        axes[0].plot(snr, res['users'][k]['I_MAP'], 'o-', color=c,
                     markersize=3, linewidth=1.2, label=lbl)
        axes[1].semilogy(snr, res['users'][k]['MAP_SER'], 'o-', color=c,
                         markersize=3, linewidth=1.2, label=lbl)
    axes[0].set_xlabel('SNR (dB)', fontsize=12)
    axes[0].set_ylabel(r'$I_{\mathrm{MAP}}$ (bits)', fontsize=12)
    axes[0].set_title('Per-User Mutual Information', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=10, framealpha=0.5)
    axes[1].set_xlabel('SNR (dB)', fontsize=12)
    axes[1].set_ylabel('MAP SER', fontsize=12)
    axes[1].set_title('Per-User Symbol Error Rate', fontsize=12)
    axes[1].grid(True, alpha=0.3, which='both')
    axes[1].legend(fontsize=10, framealpha=0.5)
    plt.tight_layout()
    _save(fig, filename)


# ----------------------------------------------------------------------
# R4#5 / Editor#2 : Non-monotonic power allocation across K and modulation
# ----------------------------------------------------------------------
def _power_path(Kval, n_points=15, t_max=2.4):
    """One-parameter allocation sweep from equal power to a reference allocation.

    P(t) = normalize(P_equal + t (P_ref - P_equal)) for t in [0, t_max], so the
    differentiation std(P(t)) grows monotonically with t while the weakest user
    (index 0) is progressively starved. The reference allocation P_ref is the
    paper's default [0.2, 0.3, 0.5] for K=3, with the SAME n_points, t_max and
    SEED as run_power_mi_simulation, so the K=3 curve traces the identical
    allocation path as the Fig. power_mi sweep and agrees with it to within
    Monte Carlo error (peak I_MAP 1.275 here vs 1.274 there; that sweep uses
    more samples). For K != 3, P_ref is the linearly increasing allocation
    [1, 2, ..., K] / sum(1..K).
    """
    P_equal = np.full(Kval, 1.0 / Kval)
    if Kval == 3:
        anchor = np.array([0.2, 0.3, 0.5])    # paper default; matches Fig power_mi
    else:
        anchor = np.arange(1, Kval + 1, dtype=float)
        anchor = anchor / anchor.sum()
    direction = anchor - P_equal
    configs = []
    for t in np.linspace(0, t_max, n_points):
        P = P_equal + t * direction
        P = np.maximum(P, 1e-3)
        P /= P.sum()
        configs.append(P)
    return configs


def sim_power_universality(snr_db=15, mods=('BPSK', 'QPSK', '16QAM'),
                           Ks=(2, 3, 4), n_samples=300000):
    """I_MAP (weak user) vs std(P) for each (K, modulation)."""
    res = {'snr_db': snr_db, 'data': {}}
    print('\n=== Power-allocation universality (SNR=%d dB) ===' % snr_db)
    orig_K = base.K
    try:
        for Kval in Ks:
            base.K = Kval                      # generate_noma_data uses base.K
            configs = _power_path(Kval)
            for mod in mods:
                S = get_constellation(mod)
                M = len(S)
                # Skip cases whose marginal-MAP enumeration M^{K-1} is too
                # large for the GPU memory budget (e.g. 16-QAM, K=4 -> 4096).
                if M ** (Kval - 1) > 1024:
                    print('  SKIP %s K=%d (M^{K-1}=%d exceeds memory budget)'
                          % (mod, Kval, M ** (Kval - 1)))
                    continue
                stds, imap = [], []
                for P_arr in configs:
                    np.random.seed(SEED)
                    y, h_est, h_true, x_k, x_all, x_k_idx, sigma2 = \
                        generate_noma_data(n_samples, snr_db, mod, 0, P_arr)
                    _, post = compute_log_marginal_posteriors(
                        y, h_est, 0, S, sigma2, P_arr)
                    stds.append(float(np.std(P_arr)))
                    imap.append(float(compute_ccmi(post, M)))
                key = '%s_K%d' % (mod, Kval)
                res['data'][key] = {'std': stds, 'I_MAP': imap,
                                    'peak_std': stds[int(np.argmax(imap))],
                                    'peak_I': float(np.max(imap))}
                print('  %s K=%d : peak I_MAP=%.3f at std=%.3f'
                      % (mod, Kval, np.max(imap), stds[int(np.argmax(imap))]))
    finally:
        base.K = orig_K
    return res


def plot_power_universality(res, filename, mods=('BPSK', 'QPSK', '16QAM'),
                            Ks=(2, 3, 4)):
    _ieee_style()
    fig, axes = plt.subplots(1, len(mods), figsize=(3.2 * len(mods), 3.0),
                             sharey=False)
    if len(mods) == 1:
        axes = [axes]
    styles = {2: 's--', 3: 'o-', 4: '^:'}
    M_of = {'BPSK': 2, 'QPSK': 4, '16QAM': 16}
    for ax, mod in zip(axes, mods):
        for Kval in Ks:
            # Skip cases beyond the GPU memory budget (e.g. 16-QAM, K=4),
            # matching the simulation, so a stale cached entry is not
            # re-introduced into the figure.
            if M_of[mod] ** (Kval - 1) > 1024:
                continue
            key = '%s_K%d' % (mod, Kval)
            if key not in res['data']:
                continue
            d = res['data'][key]
            ax.plot(d['std'], d['I_MAP'], styles.get(Kval, 'o-'),
                    markersize=3, linewidth=1.2, label='K=%d' % Kval)
        mod_disp = '16-QAM' if mod == '16QAM' else mod
        ax.set_title(mod_disp, fontsize=13)
        ax.set_xlabel(r'Power diff. (std of $\mathbf{P}$)', fontsize=13)
        ax.set_ylabel(r'$I_{\mathrm{MAP}}$ (bits)', fontsize=13)
        ax.grid(True, alpha=0.3); ax.legend(fontsize=11, framealpha=0.5)
    plt.tight_layout()
    _save(fig, filename)


# ----------------------------------------------------------------------
def main():
    t0 = time.time()
    summary = {}

    print('Rigorous run: N_MI=%d, N_POW=%d' % (N_MI, N_POW))

    csi = sim_imperfect_csi('QPSK', user_k=0, n_samples=N_MI)
    plot_imperfect_csi(csi, 'QPSK', os.path.join(OUT, 'fig_it_csi.eps'))
    summary['csi'] = csi

    fair = sim_fairness('QPSK', n_samples=N_MI)
    plot_fairness(fair, 'QPSK', os.path.join(OUT, 'fig_it_fairness.eps'))
    summary['fairness'] = fair

    powu = sim_power_universality(snr_db=15, n_samples=N_POW)
    plot_power_universality(powu, os.path.join(OUT,
                            'fig_it_power_universality.eps'))
    summary['power_universality'] = powu

    with open(os.path.join(OUT, 'revision1_results.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print('\nAll revision-1 figures done in %.1f min' % ((time.time() - t0) / 60))


if __name__ == '__main__':
    main()
