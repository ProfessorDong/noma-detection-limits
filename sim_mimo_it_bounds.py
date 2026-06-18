#!/usr/bin/env python3
"""
MIMO-NOMA Information-Theoretic Bounds Simulation

Computes MIMO MI bounds for uplink NOMA with Nr receive antennas:
  - I_MMSE: MMSE (treating interference as noise) MI
  - I_MAP: Marginal MAP MI (CCMI)
  - I_oracle: Oracle MI (perfect cancellation + single-user CCMI)
  - Gaussian capacity reference
  - SER curves: MMSE, MMSE-SIC, Oracle, Exact MAP

Generates figures:
  fig_mimo_mi_comparison.eps  — MI hierarchy for Nr=2 and Nr=4
  fig_mimo_ser_comparison.eps — SER curves for Nr=2 and Nr=4
  fig_mimo_diversity_gain.eps — MAP-Oracle gap vs Nr
  fig_mimo_nr_mi_tradeoff.eps — I_MAP vs Nr at fixed SNR

Author: Liang Dong, Baylor University
"""

import sys
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)

import numpy as np
import torch
from itertools import product
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# R1: Type-42 (TrueType) fonts for IEEE production (avoid Type-3 rejection).
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['pdf.fonttype'] = 42
import time
import os
import warnings
warnings.filterwarnings('ignore')

# Import MIMO detectors from existing script
from sim_mimo_noma import (
    get_constellation, generate_mimo_data,
    detect_mmse, detect_mmse_sic, detect_oracle_sic, detect_exact_map,
    K, P_DEFAULT, SEED, DEVICE, setup_ieee
)

print(f"Using device: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

OUT_DIR = os.environ.get('NOMA_OUT_DIR', '.')


# ============================================================
# MIMO MI Computation Functions
# ============================================================

def compute_mimo_map_posteriors(y, H, user_k, S, sigma2, P_arr,
                                 batch_size=5000):
    """
    Compute marginal MAP posteriors p(x_k = s | y, H, P) for MIMO-NOMA.

    Enumerates all M^(K-1) interference hypotheses and marginalizes.

    Args:
        y: (n, Nr) complex received signal
        H: (n, Nr, K) complex channel matrix
        S: (M,) constellation
        sigma2: noise variance
        P_arr: (K,) power allocation

    Returns:
        posteriors: (n, M) array of p(s|y,H,P)
    """
    sqrt_P = np.sqrt(P_arr)
    K_total = len(P_arr)
    M = len(S)
    n = len(y)
    Nr = H.shape[1]

    other_users = [j for j in range(K_total) if j != user_k]
    combos = np.array(list(product(range(M), repeat=len(other_users))))
    n_combos = len(combos)

    # Precompute interference symbol vectors: (n_combos, K-1) complex
    interf_syms = np.zeros((n_combos, len(other_users)), dtype=complex)
    for i, j in enumerate(other_users):
        interf_syms[:, i] = sqrt_P[j] * S[combos[:, i]]
    interf_t = torch.tensor(interf_syms, dtype=torch.complex64, device=DEVICE)

    y_t = torch.tensor(y, dtype=torch.complex64, device=DEVICE)
    H_t = torch.tensor(H, dtype=torch.complex64, device=DEVICE)
    S_t = torch.tensor(S, dtype=torch.complex64, device=DEVICE)

    posteriors = np.zeros((n, M), dtype=np.float64)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        nb = end - start
        yb = y_t[start:end]   # (nb, Nr)
        Hb = H_t[start:end]   # (nb, Nr, K)

        H_other = Hb[:, :, other_users]  # (nb, Nr, K-1)
        # interf_vec: (nb, Nr, n_combos) = H_other @ interf_t.T
        interf_vec = torch.matmul(H_other, interf_t.T)  # (nb, Nr, n_combos)

        h_k = Hb[:, :, user_k]  # (nb, Nr)
        sigma2_inv = 1.0 / sigma2

        log_marg = torch.zeros(nb, M, device=DEVICE, dtype=torch.float64)
        for s_idx in range(M):
            sig_k = h_k * (sqrt_P[user_k] * S_t[s_idx])  # (nb, Nr)
            # residual: y - sig_k - interf  -> (nb, Nr, n_combos)
            res = yb.unsqueeze(2) - sig_k.unsqueeze(2) - interf_vec
            log_lik = -torch.sum(torch.abs(res) ** 2, dim=1).double() * sigma2_inv
            mx, _ = log_lik.max(dim=1, keepdim=True)
            log_marg[:, s_idx] = mx.squeeze(1) + torch.log(
                torch.sum(torch.exp(log_lik - mx), dim=1))

        # Normalize
        log_norm = torch.logsumexp(log_marg, dim=1, keepdim=True)
        log_post = log_marg - log_norm
        posteriors[start:end] = torch.exp(log_post).cpu().numpy()

    return posteriors


def compute_mimo_ccmi(posteriors, M):
    """
    Compute CCMI from posteriors: I_MAP = log2(M) - H(x_k | y, H, P).
    """
    p = np.clip(posteriors, 1e-30, 1.0)
    H_per_sample = -np.sum(p * np.log2(p), axis=1)
    H_cond = np.mean(H_per_sample)
    return np.log2(M) - H_cond


def compute_mimo_oracle_mi(y, H, user_k, x_all, S, sigma2, P_arr,
                             batch_size=20000):
    """
    Oracle MI: I(x_k; y | H, x_{-k}, P).
    After perfect cancellation, compute single-user MIMO CCMI.

    y_eff = H[:,:,k] * sqrt(P_k) * x_k + noise
    """
    sqrt_P = np.sqrt(P_arr)
    K_total = len(P_arr)
    M = len(S)
    n = len(y)
    Nr = H.shape[1]

    # Cancel all interferers
    y_eff = y.copy()
    for j in range(K_total):
        if j != user_k:
            y_eff -= H[:, :, j] * (sqrt_P[j] * x_all[:, j:j+1])

    # Now y_eff = H[:,:,k] * sqrt(P_k) * x_k + noise
    sqrt_Pk = sqrt_P[user_k]
    S_t = torch.tensor(S, dtype=torch.complex64, device=DEVICE)

    posteriors = np.zeros((n, M), dtype=np.float64)
    sigma2_inv = 1.0 / sigma2

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        nb = end - start
        y_b = torch.tensor(y_eff[start:end], dtype=torch.complex64, device=DEVICE)
        h_b = torch.tensor(H[start:end, :, user_k], dtype=torch.complex64, device=DEVICE)

        # (nb, Nr, M): hypothesis signals
        hyp = h_b.unsqueeze(2) * (sqrt_Pk * S_t.unsqueeze(0).unsqueeze(0))
        diff = y_b.unsqueeze(2) - hyp  # (nb, Nr, M)
        log_lik = -torch.sum(torch.abs(diff) ** 2, dim=1).double() * sigma2_inv  # (nb, M)

        log_norm = torch.logsumexp(log_lik, dim=1, keepdim=True)
        log_post = log_lik - log_norm
        posteriors[start:end] = torch.exp(log_post).cpu().numpy()

    return compute_mimo_ccmi(posteriors, M)


def compute_mimo_mmse_mi(y, H, user_k, S, sigma2, P_arr, x_k_idx,
                           batch_size=20000):
    """
    MMSE TIN MI (generalized MI) for MIMO.
    Uses MMSE filter output and computes GMI treating residual as Gaussian.

    GMI = log2(M) + (1/n) * sum_i log2 q(x_true | y)
    where q is the MMSE-based posterior.
    """
    sqrt_P = np.sqrt(P_arr)
    K_total = len(P_arr)
    M = len(S)
    n = len(y)
    Nr = H.shape[1]

    y_t = torch.tensor(y, dtype=torch.complex64, device=DEVICE)
    H_t = torch.tensor(H, dtype=torch.complex64, device=DEVICE)
    S_t = torch.tensor(S, dtype=torch.complex64, device=DEVICE)
    sqrt_P_t = torch.tensor(sqrt_P, dtype=torch.float32, device=DEVICE)

    log_q_true_all = np.zeros(n, dtype=np.float64)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        nb = end - start
        yb = y_t[start:end]
        Hb = H_t[start:end]
        idx_b = torch.tensor(x_k_idx[start:end], dtype=torch.long, device=DEVICE)

        H_eff = Hb * sqrt_P_t.unsqueeze(0).unsqueeze(0)  # (nb, Nr, K)
        cov = torch.bmm(H_eff, H_eff.conj().transpose(1, 2))
        cov += sigma2 * torch.eye(Nr, device=DEVICE).unsqueeze(0)

        h_k = H_eff[:, :, user_k]  # (nb, Nr)
        w_k = torch.linalg.solve(cov, h_k.unsqueeze(-1)).squeeze(-1)  # (nb, Nr)

        z_k = (w_k.conj() * yb).sum(dim=1)  # (nb,)
        g_k = (w_k.conj() * h_k).sum(dim=1)  # (nb,)

        # Effective noise variance for MMSE output: g_k(1-g_k)
        # g_k = h_k^H R^{-1} h_k is real positive; use .real to discard
        # any residual imaginary part from floating-point arithmetic
        g_real = g_k.real.double()
        noise_var = g_real * (1.0 - g_real)
        noise_var = torch.clamp(noise_var, min=1e-10)

        # Compute TIN posteriors based on MMSE output
        # z_k ≈ g_k * x_k + effective_noise
        # Use distance-based soft decisions
        dists = torch.abs(z_k.unsqueeze(1) - g_k.unsqueeze(1) * S_t.unsqueeze(0)) ** 2
        log_lik = -dists.double() / noise_var.double().unsqueeze(1)

        log_norm = torch.logsumexp(log_lik, dim=1)
        log_q_true = log_lik[torch.arange(nb, device=DEVICE), idx_b] - log_norm
        log_q_true_all[start:end] = log_q_true.cpu().numpy()

    I_mmse = np.log2(M) + np.mean(log_q_true_all) / np.log(2)
    return max(I_mmse, 0.0)


def compute_mimo_gaussian_rate(snr_db_arr, Nr, user_k, P_arr,
                                 n_samples=200000):
    """
    Ergodic Gaussian capacity reference for MIMO uplink.
    R_k = E_H[log2(1 + P_k * h_k^H (sum_{j!=k} P_j h_j h_j^H + sigma2 I)^{-1} h_k)]
    """
    K_total = len(P_arr)
    rates = []

    for snr_db in snr_db_arr:
        sigma2 = 10.0 ** (-snr_db / 10.0)

        np.random.seed(SEED + int(snr_db * 100))
        H = (np.random.randn(n_samples, Nr, K_total) +
             1j * np.random.randn(n_samples, Nr, K_total)) / np.sqrt(2)

        # R_k = E[log2(1 + P_k h_k^H R_{-k}^{-1} h_k)]
        # where R_{-k} = sum_{j!=k} P_j h_j h_j^H + sigma2 I
        H_t = torch.tensor(H, dtype=torch.complex64, device=DEVICE)
        batch = 20000
        rate_sum = 0.0
        count = 0

        for s in range(0, n_samples, batch):
            e = min(s + batch, n_samples)
            nb = e - s
            Hb = H_t[s:e]

            # Build interference-plus-noise covariance
            R = sigma2 * torch.eye(Nr, device=DEVICE, dtype=torch.complex64).unsqueeze(0).expand(nb, -1, -1).clone()
            for j in range(K_total):
                if j < user_k:  # Only lower-power users (not yet decoded) treated as interference
                    hj = Hb[:, :, j:j+1]  # (nb, Nr, 1)
                    R = R + P_arr[j] * torch.bmm(hj, hj.conj().transpose(1, 2))

            h_k = Hb[:, :, user_k:user_k+1]  # (nb, Nr, 1)
            # SINR = P_k * h_k^H R^{-1} h_k
            R_inv_hk = torch.linalg.solve(R, h_k)  # (nb, Nr, 1)
            sinr = P_arr[user_k] * torch.bmm(h_k.conj().transpose(1, 2), R_inv_hk).squeeze()
            rate_sum += torch.log2(1 + sinr.real).sum().item()
            count += nb

        rates.append(rate_sum / count)

    return np.array(rates)


# ============================================================
# Main MI Simulation
# ============================================================

def run_mimo_mi_simulation(mod_type, Nr, user_k=0, P_arr=None,
                            n_samples=200000):
    """
    Compute MIMO MI hierarchy: I_MMSE, I_MAP, I_oracle vs SNR.
    Also computes SER for all detectors.
    """
    if P_arr is None:
        P_arr = P_DEFAULT
    S = get_constellation(mod_type)
    M = len(S)

    if mod_type == '16QAM':
        snr_range = np.arange(5, 31, 2)
    else:
        snr_range = np.arange(0, 31, 2)

    results = {
        'I_MMSE': [], 'I_MAP': [], 'I_oracle': [], 'R_Gauss': [],
        'SER_MMSE': [], 'SER_MMSE_SIC': [], 'SER_Oracle': [], 'SER_MAP': [],
    }

    print(f"\n{'='*60}")
    print(f"  MIMO MI Simulation: {mod_type} | Nr={Nr} | K={K} | User {user_k+1}")
    print(f"  P = {P_arr} | n_samples = {n_samples}")
    print(f"{'='*60}")

    # Precompute Gaussian rates
    R_gauss = compute_mimo_gaussian_rate(snr_range, Nr, user_k, P_arr, n_samples)

    for i, snr_db in enumerate(snr_range):
        t0 = time.time()
        np.random.seed(SEED + int(snr_db * 100))
        y, H, x_all, x_idx, sigma2 = generate_mimo_data(
            n_samples, snr_db, mod_type, Nr, K, P_arr)
        x_k_idx = x_idx[:, user_k]

        # 1. MAP posteriors and CCMI
        posteriors = compute_mimo_map_posteriors(
            y, H, user_k, S, sigma2, P_arr)
        I_map = compute_mimo_ccmi(posteriors, M)

        # 2. Oracle MI
        I_oracle = compute_mimo_oracle_mi(
            y, H, user_k, x_all, S, sigma2, P_arr)

        # 3. MMSE MI
        I_mmse = compute_mimo_mmse_mi(
            y, H, user_k, S, sigma2, P_arr, x_k_idx)

        # 4. SER values
        p_mmse = detect_mmse(y, H, user_k, S, sigma2, P_arr)
        ser_mmse = np.mean(p_mmse != x_k_idx)

        p_sic = detect_mmse_sic(y, H, user_k, S, sigma2, P_arr)
        ser_sic = np.mean(p_sic != x_k_idx)

        p_orc = detect_oracle_sic(y, H, user_k, x_all, S, sigma2, P_arr)
        ser_orc = np.mean(p_orc != x_k_idx)

        p_map = detect_exact_map(y, H, user_k, S, sigma2, P_arr)
        ser_map = np.mean(p_map != x_k_idx)

        # Store
        results['I_MMSE'].append(I_mmse)
        results['I_MAP'].append(I_map)
        results['I_oracle'].append(I_oracle)
        results['R_Gauss'].append(R_gauss[i])
        results['SER_MMSE'].append(max(ser_mmse, 1e-6))
        results['SER_MMSE_SIC'].append(max(ser_sic, 1e-6))
        results['SER_Oracle'].append(max(ser_orc, 1e-6))
        results['SER_MAP'].append(max(ser_map, 1e-6))

        dt = time.time() - t0
        print(f"  SNR={snr_db:2d} dB | I_MMSE={I_mmse:.4f} | I_MAP={I_map:.4f} | "
              f"I_orc={I_oracle:.4f} | R_G={R_gauss[i]:.4f} | "
              f"MAP={ser_map:.2e} | ORC={ser_orc:.2e} | SIC={ser_sic:.2e} | "
              f"MMSE={ser_mmse:.2e} | {dt:.1f}s")

    for key in results:
        results[key] = np.array(results[key])

    return snr_range, results


# ============================================================
# Plotting Functions
# ============================================================

def plot_mimo_mi_comparison(snr_2, res_2, snr_4, res_4, filename):
    """
    Plot MI hierarchy for Nr=2 and Nr=4 side by side.
    """
    setup_ieee()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))

    colors = {'I_MMSE': '#d62728', 'I_MAP': '#1f77b4',
              'I_oracle': '#2ca02c', 'R_Gauss': '#9467bd'}
    markers = {'I_MMSE': 's', 'I_MAP': 'D', 'I_oracle': '^', 'R_Gauss': 'x'}
    labels = {'I_MMSE': r'$I_{\mathrm{MMSE}}$',
              'I_MAP': r'$I_{\mathrm{MAP}}$',
              'I_oracle': r'$I_{\mathrm{oracle}}$',
              'R_Gauss': r'$R_{\mathrm{Gauss}}$'}

    M = 4  # QPSK

    for ax_idx, (snr, res, Nr) in enumerate([(snr_2, res_2, 2), (snr_4, res_4, 4)]):
        ax = axes[ax_idx]
        ax.axhline(y=np.log2(M), color='gray', linestyle=':', linewidth=0.8,
                   alpha=0.5, label=f'$\\log_2 M = {np.log2(M):.0f}$')

        for key in ['I_MMSE', 'I_MAP', 'I_oracle', 'R_Gauss']:
            mfc = 'none' if key != 'I_MAP' else colors[key]
            ax.plot(snr, res[key], color=colors[key], marker=markers[key],
                    markersize=4, linewidth=1.2, label=labels[key],
                    markerfacecolor=mfc)

        ax.set_xlabel('SNR (dB)', fontsize=12)
        if ax_idx == 0:
            ax.set_ylabel('Mutual Information (bits)', fontsize=12)
        ax.set_title(f'QPSK, $N_r = {Nr}$', fontsize=12)
        ax.legend(fontsize=9, loc='upper left', framealpha=0.5)
        ax.set_ylim(bottom=-0.1)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(filename, format='eps')
    fig.savefig(filename.replace('.eps', '.png'), format='png', dpi=300)
    # PDF preserves the translucent legends that the EPS backend flattens.
    fig.savefig(filename.replace('.eps', '.pdf'), format='pdf')
    plt.close(fig)
    print(f"  Saved: {filename}")


def plot_mimo_ser_comparison(snr_2, res_2, snr_4, res_4, filename):
    """
    Plot SER curves for Nr=2 and Nr=4 side by side.
    """
    setup_ieee()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))

    det_styles = {
        'SER_MMSE': ('v', 'gray', ':', 'MMSE'),
        'SER_MMSE_SIC': ('s', '#d62728', '--', 'MMSE-SIC'),
        'SER_Oracle': ('^', '#2ca02c', '-.', 'Oracle'),
        'SER_MAP': ('D', '#1f77b4', '-', 'Exact MAP'),
    }

    for ax_idx, (snr, res, Nr) in enumerate([(snr_2, res_2, 2), (snr_4, res_4, 4)]):
        ax = axes[ax_idx]
        for key, (m, c, ls, label) in det_styles.items():
            vals = np.maximum(res[key], 1e-6)
            ax.semilogy(snr, vals, marker=m, color=c, linestyle=ls,
                        linewidth=1.2, label=label, markersize=4,
                        markerfacecolor='none')

        ax.set_xlabel('SNR (dB)', fontsize=12)
        if ax_idx == 0:
            ax.set_ylabel('SER', fontsize=12)
        ax.set_title(f'QPSK, $N_r = {Nr}$', fontsize=12)
        ax.legend(fontsize=9, loc='lower left', framealpha=0.5)
        ax.set_ylim(bottom=5e-6, top=1.5)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(filename, format='eps')
    fig.savefig(filename.replace('.eps', '.png'), format='png', dpi=300)
    # PDF preserves the translucent legends that the EPS backend flattens.
    fig.savefig(filename.replace('.eps', '.pdf'), format='pdf')
    plt.close(fig)
    print(f"  Saved: {filename}")


def plot_mimo_diversity_gain(results_by_nr, snr_ranges, filename):
    """
    Plot MAP-Oracle SER gap vs Nr, showing spatial diversity reduces the gap.
    At a reference SNR, plot gap ratio for each Nr.
    """
    setup_ieee()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))

    # Left: SER curves for MAP and Oracle across Nr values
    ax = axes[0]
    colors_nr = {1: '#ff7f0e', 2: '#d62728', 3: '#9467bd', 4: '#1f77b4'}
    for Nr in sorted(results_by_nr.keys()):
        snr = snr_ranges[Nr]
        res = results_by_nr[Nr]
        c = colors_nr.get(Nr, 'black')
        ax.semilogy(snr, np.maximum(res['SER_MAP'], 1e-6), 'D-',
                    color=c, markersize=4, linewidth=1.2,
                    label=f'MAP, $N_r={Nr}$')
        ax.semilogy(snr, np.maximum(res['SER_Oracle'], 1e-6), '^--',
                    color=c, markersize=4, linewidth=1.0,
                    label=f'Oracle, $N_r={Nr}$', markerfacecolor='none')
    ax.set_xlabel('SNR (dB)', fontsize=12)
    ax.set_ylabel('SER', fontsize=12)
    ax.set_title('MAP vs Oracle SER', fontsize=12)
    ax.legend(fontsize=8, loc='lower left', framealpha=0.5)
    ax.set_ylim(bottom=5e-6, top=1.5)
    ax.grid(True, alpha=0.3)

    # Right: MAP-Oracle SER ratio vs SNR for each Nr
    ax = axes[1]
    for Nr in sorted(results_by_nr.keys()):
        snr = snr_ranges[Nr]
        res = results_by_nr[Nr]
        ratio = np.maximum(res['SER_MAP'], 1e-6) / np.maximum(res['SER_Oracle'], 1e-6)
        c = colors_nr.get(Nr, 'black')
        ax.semilogy(snr, ratio, 'o-', color=c, markersize=4, linewidth=1.2,
                    label=f'$N_r={Nr}$')
    ax.set_xlabel('SNR (dB)', fontsize=12)
    ax.set_ylabel('SER$_{\\mathrm{MAP}}$ / SER$_{\\mathrm{Oracle}}$', fontsize=12)
    ax.set_title('MAP-Oracle Gap Ratio', fontsize=12)
    ax.legend(fontsize=10, framealpha=0.5)
    ax.axhline(y=1.0, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(filename, format='eps')
    fig.savefig(filename.replace('.eps', '.png'), format='png', dpi=300)
    # PDF preserves the translucent legends that the EPS backend flattens.
    fig.savefig(filename.replace('.eps', '.pdf'), format='pdf')
    plt.close(fig)
    print(f"  Saved: {filename}")


def plot_mimo_nr_mi_tradeoff(results_by_nr, snr_ranges, filename):
    """
    Plot I_MAP and I_oracle vs Nr at a few fixed SNR values.
    """
    setup_ieee()
    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    Nr_vals = sorted(results_by_nr.keys())
    # Pick reference SNRs
    ref_snrs = [6, 12, 18]
    colors_snr = {6: '#d62728', 12: '#1f77b4', 18: '#2ca02c'}

    for snr_ref in ref_snrs:
        mi_map_vals = []
        mi_orc_vals = []
        for Nr in Nr_vals:
            snr = snr_ranges[Nr]
            res = results_by_nr[Nr]
            idx = np.argmin(np.abs(snr - snr_ref))
            mi_map_vals.append(res['I_MAP'][idx])
            mi_orc_vals.append(res['I_oracle'][idx])

        c = colors_snr[snr_ref]
        ax.plot(Nr_vals, mi_map_vals, 'D-', color=c, markersize=5,
                linewidth=1.2, label=f'$I_{{\\mathrm{{MAP}}}}$, {snr_ref} dB')
        ax.plot(Nr_vals, mi_orc_vals, '^--', color=c, markersize=5,
                linewidth=1.0, markerfacecolor='none',
                label=f'$I_{{\\mathrm{{oracle}}}}$, {snr_ref} dB')

    ax.axhline(y=2.0, color='gray', linestyle=':', linewidth=0.8, alpha=0.5,
               label='$\\log_2 M = 2$')
    ax.set_xlabel('Number of Receive Antennas $N_r$', fontsize=8)
    ax.set_ylabel('Mutual Information (bits)', fontsize=8)
    ax.set_title('QPSK: MI vs. $N_r$', fontsize=8)
    ax.set_xticks(Nr_vals)
    ax.legend(fontsize=6.5, loc='lower right', framealpha=0.9, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    fig.savefig(filename, format='eps')
    fig.savefig(filename.replace('.eps', '.png'), format='png', dpi=300)
    # PDF preserves the translucent legends that the EPS backend flattens.
    fig.savefig(filename.replace('.eps', '.pdf'), format='pdf')
    plt.close(fig)
    print(f"  Saved: {filename}")


# ============================================================
# Main Execution
# ============================================================

def main():
    t0 = time.time()
    mod_type = 'QPSK'
    n_samples = 4000000  # R1: 4M for max rigor

    # ============================================================
    # Step 1: MIMO MI Simulations for Nr = 1, 2, 3, 4
    # ============================================================
    all_results = {}
    all_snr = {}

    for Nr in [1, 2, 3, 4]:
        snr, res = run_mimo_mi_simulation(mod_type, Nr=Nr, user_k=0,
                                           n_samples=n_samples)
        all_results[Nr] = res
        all_snr[Nr] = snr
        torch.cuda.empty_cache()

    # ============================================================
    # Step 2: Verification
    # ============================================================
    print(f"\n{'='*60}")
    print("  VERIFICATION: MIMO MI Ordering")
    print(f"{'='*60}")

    all_ok = True
    for Nr in [1, 2, 3, 4]:
        snr = all_snr[Nr]
        res = all_results[Nr]
        M = 4  # QPSK
        for i, snr_db in enumerate(snr):
            mmse_ok = res['I_MMSE'][i] <= res['I_MAP'][i] + 0.05
            map_ok = res['I_MAP'][i] <= res['I_oracle'][i] + 0.01
            ceiling_ok = res['I_MAP'][i] <= np.log2(M) + 0.01

            if not (mmse_ok and map_ok and ceiling_ok):
                print(f"  WARN: Nr={Nr} SNR={snr_db}: "
                      f"I_MMSE={res['I_MMSE'][i]:.4f}, I_MAP={res['I_MAP'][i]:.4f}, "
                      f"I_oracle={res['I_oracle'][i]:.4f}")
                all_ok = False

    if all_ok:
        print("  ALL CHECKS PASSED: MI ordering holds.")
    else:
        print("  Some MI ordering checks had warnings (small tolerance violations).")

    # ============================================================
    # Step 3: Generate Figures
    # ============================================================
    print(f"\n{'='*60}")
    print("  GENERATING FIGURES")
    print(f"{'='*60}")

    # Figure 5: MI comparison (Nr=2 and Nr=4)
    plot_mimo_mi_comparison(all_snr[2], all_results[2], all_snr[4], all_results[4],
                            os.path.join(OUT_DIR, 'fig_mimo_mi_comparison.eps'))

    # Figure 6: SER comparison (Nr=2 and Nr=4)
    plot_mimo_ser_comparison(all_snr[2], all_results[2], all_snr[4], all_results[4],
                             os.path.join(OUT_DIR, 'fig_mimo_ser_comparison.eps'))

    # Figure 7: Diversity gain (Nr=2 and Nr=4)
    div_results = {2: all_results[2], 4: all_results[4]}
    div_snr = {2: all_snr[2], 4: all_snr[4]}
    plot_mimo_diversity_gain(div_results, div_snr,
                             os.path.join(OUT_DIR, 'fig_mimo_diversity_gain.eps'))

    # Figure 8: Nr-MI tradeoff (all Nr values)
    plot_mimo_nr_mi_tradeoff(all_results, all_snr,
                              os.path.join(OUT_DIR, 'fig_mimo_nr_mi_tradeoff.eps'))

    # ------------------------------------------------------------
    # Persist all MIMO results so the figures can be re-plotted
    # (e.g., reformatted) later without re-running the simulation.
    # Load with replot_mimo.py.
    # ------------------------------------------------------------
    import pickle
    cache_path = os.path.join(OUT_DIR, 'sim_mimo_results.pkl')
    with open(cache_path, 'wb') as f:
        pickle.dump({'all_snr': all_snr, 'all_results': all_results,
                     'n_samples': n_samples, 'mod_type': mod_type}, f)
    print(f"\n  Saved all results to {cache_path} "
          f"(re-plot with replot_mimo.py, no simulation needed)")

    # ============================================================
    # Step 4: Summary
    # ============================================================
    print(f"\n{'='*60}")
    print("  SUMMARY: MIMO IT Bounds")
    print(f"{'='*60}")

    for Nr in [1, 2, 3, 4]:
        snr = all_snr[Nr]
        res = all_results[Nr]
        mid_idx = len(snr) // 2
        high_idx = -1
        print(f"\n  Nr={Nr}, QPSK:")
        print(f"    Mid SNR={snr[mid_idx]} dB:")
        print(f"      I_MMSE={res['I_MMSE'][mid_idx]:.4f}, "
              f"I_MAP={res['I_MAP'][mid_idx]:.4f}, "
              f"I_oracle={res['I_oracle'][mid_idx]:.4f}")
        print(f"      MAP-MMSE gap = {res['I_MAP'][mid_idx]-res['I_MMSE'][mid_idx]:.4f}")
        print(f"      Oracle-MAP gap = {res['I_oracle'][mid_idx]-res['I_MAP'][mid_idx]:.4f}")
        print(f"    High SNR={snr[high_idx]} dB:")
        print(f"      I_MAP={res['I_MAP'][high_idx]:.4f}, "
              f"I_oracle={res['I_oracle'][high_idx]:.4f}")
        print(f"      SER_MAP={res['SER_MAP'][high_idx]:.2e}, "
              f"SER_Oracle={res['SER_Oracle'][high_idx]:.2e}")

    elapsed = time.time() - t0
    print(f"\n  Total runtime: {elapsed/60:.1f} minutes")
    print("  Done.")


if __name__ == '__main__':
    main()
