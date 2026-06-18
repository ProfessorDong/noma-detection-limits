#!/usr/bin/env python3
"""
Information-Theoretic Bounds for Power-Domain NOMA Detection

Computes:
  - Constellation-Constrained Mutual Information (CCMI)
  - Oracle MI (after perfect interference cancellation)
  - Treating-Interference-as-Noise MI (TIN)
  - Gaussian capacity reference
  - Fano lower bound on error probability
  - PEP union upper bound on error probability
  - MAP, SIC, Oracle SIC SER for validation

Generates figures for Paper 1:
  "Information-Theoretic Limits of Power-Domain NOMA Detection"

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
from matplotlib.ticker import LogLocator
import time
import os
import warnings
warnings.filterwarnings('ignore')

# R1: force Type-42 (TrueType) fonts in EPS/PDF to avoid IEEE production
# rejection of Type-3 fonts. setup_ieee_style() does not touch these keys,
# so setting them once here persists through all plotting calls.
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['pdf.fonttype'] = 42

# Import reusable functions from existing simulation
from sim_learned_mud import (
    get_constellation, generate_noma_data,
    detect_exact_map, detect_conv_sic, detect_oracle_sic,
    setup_ieee_style,
    K, P_DEFAULT, SEED, DEVICE
)

print(f"Using device: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

OUT_DIR = os.environ.get('NOMA_OUT_DIR', '.')


# ============================================================
# Core MI Computation Functions
# ============================================================

def compute_log_marginal_posteriors(y_k, h_k, user_k, S, sigma2, P_arr,
                                    batch_size=20000):
    """
    Compute log marginal posterior p(x_k = s | y_k, h_k, P) for each symbol s.

    Returns:
        log_posteriors: (n_samples, M) array of log p(s|y,h,P)
        posteriors: (n_samples, M) array of p(s|y,h,P)
    """
    sqrt_P = np.sqrt(P_arr)
    K_total = len(P_arr)
    M = len(S)
    n = len(y_k)

    other_users = [j for j in range(K_total) if j != user_k]
    combos = np.array(list(product(range(M), repeat=len(other_users))))
    n_combos = len(combos)

    # Precompute interference vectors
    interference = np.zeros(n_combos, dtype=complex)
    for i, j in enumerate(other_users):
        interference += sqrt_P[j] * S[combos[:, i]]

    # Composite signals: for each candidate symbol s and each interference combo
    composites = np.zeros((M, n_combos), dtype=complex)
    for s_idx in range(M):
        composites[s_idx] = sqrt_P[user_k] * S[s_idx] + interference

    # Move to GPU
    composites_t = torch.tensor(composites, dtype=torch.complex64, device=DEVICE)
    y_t = torch.tensor(y_k, dtype=torch.complex64, device=DEVICE)
    h_t = torch.tensor(h_k, dtype=torch.complex64, device=DEVICE)
    sigma2_inv = 1.0 / sigma2

    log_posteriors = np.zeros((n, M), dtype=np.float64)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        y_b = y_t[start:end]
        h_b = h_t[start:end]
        nb = end - start

        log_marginal = torch.zeros(nb, M, device=DEVICE, dtype=torch.float64)

        for s_idx in range(M):
            # diff: (nb, n_combos)
            diff = y_b[:, None] - h_b[:, None] * composites_t[s_idx:s_idx+1, :]
            log_lik = -torch.abs(diff).pow(2).double() * sigma2_inv
            # Log-sum-exp for numerical stability
            max_ll, _ = log_lik.max(dim=1, keepdim=True)
            log_marginal[:, s_idx] = max_ll.squeeze(1) + torch.log(
                torch.sum(torch.exp(log_lik - max_ll), dim=1))

        # Normalize to get log posteriors (uniform prior)
        log_norm = torch.logsumexp(log_marginal, dim=1, keepdim=True)
        log_post = log_marginal - log_norm
        log_posteriors[start:end] = log_post.cpu().numpy()

    posteriors = np.exp(log_posteriors)
    return log_posteriors, posteriors


def compute_conditional_entropy(posteriors):
    """
    Compute H(x_k | y_k, h_k, P) = -E[ sum_s p(s|y,h) log2 p(s|y,h) ]

    Args:
        posteriors: (n_samples, M) array of p(s|y,h,P)

    Returns:
        H_cond: scalar conditional entropy (bits)
    """
    # Clip for numerical stability
    p = np.clip(posteriors, 1e-30, 1.0)
    # Per-sample entropy
    H_per_sample = -np.sum(p * np.log2(p), axis=1)
    return np.mean(H_per_sample)


def compute_ccmi(posteriors, M):
    """
    Constellation-Constrained Mutual Information:
    I_MAP = log2(M) - H(x_k | y_k, h_k, P)

    Args:
        posteriors: (n_samples, M)
        M: constellation size

    Returns:
        I_MAP: mutual information in bits
    """
    H_cond = compute_conditional_entropy(posteriors)
    return np.log2(M) - H_cond


def compute_oracle_mi(y_k, h_k, user_k, x_all, S, sigma2, P_arr,
                       batch_size=50000):
    """
    Oracle MI: I(x_k; y_k | h_k, x_{-k}, P)
    After perfect interference cancellation, this is single-user CCMI.

    Computes via effective observation: y_eff = h_k * sqrt(P_k) * x_k + n
    """
    sqrt_P = np.sqrt(P_arr)
    K_total = len(P_arr)
    M = len(S)
    n = len(y_k)

    # Cancel all interferers perfectly
    y_eff = y_k.copy()
    for j in range(K_total):
        if j != user_k:
            y_eff -= h_k * sqrt_P[j] * x_all[:, j]

    # Now y_eff = h_k * sqrt(P_k) * x_k + noise
    # Compute single-user posteriors
    sqrt_Pk = sqrt_P[user_k]
    S_t = torch.tensor(S, dtype=torch.complex64, device=DEVICE)

    log_posteriors_all = np.zeros((n, M), dtype=np.float64)
    sigma2_inv = 1.0 / sigma2

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        y_b = torch.tensor(y_eff[start:end], dtype=torch.complex64, device=DEVICE)
        h_b = torch.tensor(h_k[start:end], dtype=torch.complex64, device=DEVICE)
        nb = end - start

        # (nb, M): distance to each hypothesis
        hypotheses = h_b[:, None] * sqrt_Pk * S_t[None, :]  # (nb, M)
        diff = y_b[:, None] - hypotheses
        log_lik = -torch.abs(diff).pow(2).double() * sigma2_inv  # (nb, M)

        log_norm = torch.logsumexp(log_lik, dim=1, keepdim=True)
        log_post = log_lik - log_norm
        log_posteriors_all[start:end] = log_post.cpu().numpy()

    posteriors = np.exp(log_posteriors_all)
    H_cond = compute_conditional_entropy(posteriors)
    I_oracle = np.log2(M) - H_cond
    return I_oracle


def compute_tin_mi(y_k, h_k, user_k, S, sigma2, P_arr, x_k_idx,
                    batch_size=50000):
    """
    Treating-Interference-as-Noise MI as Generalized MI (GMI).

    Computes the GMI achieved by a TIN decoder on the ACTUAL NOMA channel:
        GMI = log2(M) + (1/n) * sum_i log2 q(x_k_true | y_k)
    where q is the TIN posterior (treating interference as Gaussian noise).

    This is guaranteed to satisfy GMI <= I_MAP since the MAP posterior
    maximizes E[log p(x|y)] over all posterior models.

    Args:
        y_k: actual NOMA observations (n,)
        h_k: channel coefficients (n,)
        user_k: user index
        S: constellation
        sigma2: noise variance
        P_arr: power allocation
        x_k_idx: true symbol indices for user k (n,)
    """
    sqrt_P = np.sqrt(P_arr)
    K_total = len(P_arr)
    M = len(S)
    n = len(y_k)

    P_interf = sum(P_arr[j] for j in range(K_total) if j != user_k)
    sqrt_Pk = sqrt_P[user_k]

    # Effective noise variance (treating interference as Gaussian)
    sigma2_eff = sigma2 + np.abs(h_k)**2 * P_interf  # (n,)

    S_t = torch.tensor(S, dtype=torch.complex64, device=DEVICE)

    # Compute log q(x_k_true | y_k) using TIN posterior on actual observations
    log_q_true_all = np.zeros(n, dtype=np.float64)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        y_b = torch.tensor(y_k[start:end], dtype=torch.complex64, device=DEVICE)
        h_b = torch.tensor(h_k[start:end], dtype=torch.complex64, device=DEVICE)
        s2_eff_b = torch.tensor(sigma2_eff[start:end], dtype=torch.float64,
                                 device=DEVICE)
        idx_b = torch.tensor(x_k_idx[start:end], dtype=torch.long, device=DEVICE)
        nb = end - start

        hypotheses = h_b[:, None] * sqrt_Pk * S_t[None, :]  # (nb, M)
        diff = y_b[:, None] - hypotheses
        log_lik = -torch.abs(diff).pow(2).double() / s2_eff_b[:, None]

        log_norm = torch.logsumexp(log_lik, dim=1)  # (nb,)
        # Extract log posterior for true symbol
        log_q_true = log_lik[torch.arange(nb, device=DEVICE), idx_b] - log_norm
        log_q_true_all[start:end] = log_q_true.cpu().numpy()

    # GMI = log2(M) + E[log2 q(x_true | y)]
    I_tin = np.log2(M) + np.mean(log_q_true_all) / np.log(2)
    return max(I_tin, 0.0)  # Ensure non-negative


def compute_gaussian_rates(snr_db_arr, user_k, P_arr, n_samples=500000):
    """
    Gaussian capacity reference: R_Gauss_k = E_h[log2(1 + SINR_k)]
    where SINR_k = P_k |h_k|^2 / (sum_{j<k} P_j |h_k|^2 + sigma^2)

    For downlink with ordered channels, user k's SINR depends on SIC order.
    User 1 (strongest channel) decodes last, so treats users with lower power as interference.
    """
    K_total = len(P_arr)
    rates = []

    for snr_db in snr_db_arr:
        sigma2 = 10.0 ** (-snr_db / 10.0)

        # Generate ordered Rayleigh channels
        np.random.seed(SEED + int(snr_db * 100))
        h_all = (np.random.randn(n_samples, K_total) +
                 1j * np.random.randn(n_samples, K_total)) / np.sqrt(2)
        order = np.argsort(-np.abs(h_all), axis=1)
        h_all = np.take_along_axis(h_all, order, axis=1)
        h_k = h_all[:, user_k]

        # SINR for user k (SIC decoding: higher power users decoded first)
        # Users with indices > user_k have higher power and are decoded first
        # Users with indices < user_k have lower power and are treated as interference
        P_interference = sum(P_arr[j] for j in range(user_k))
        sinr_k = (P_arr[user_k] * np.abs(h_k)**2) / \
                 (P_interference * np.abs(h_k)**2 + sigma2)
        R_k = np.mean(np.log2(1 + sinr_k))
        rates.append(R_k)

    return np.array(rates)


def compute_fano_bound(I_map, M):
    """
    Fano-type lower bound on symbol error rate:
    P_e >= (log2(M) - I_MAP - 1) / log2(M-1)

    For M=2 (BPSK), the bound is undefined (denominator = 0), so we return 0.
    For P_e to be meaningful, we need I_MAP < log2(M) - 1,
    otherwise the bound is negative (trivially satisfied).

    Args:
        I_map: mutual information in bits (scalar or array)
        M: constellation size

    Returns:
        P_e_lower: Fano lower bound on SER
    """
    if M <= 2:
        return np.zeros_like(I_map) if hasattr(I_map, '__len__') else 0.0
    bound = (np.log2(M) - I_map - 1.0) / np.log2(M - 1)
    return np.maximum(bound, 0.0)


def compute_binary_fano_bound(I_map):
    """
    Binary-entropy Fano lower bound on SER for BPSK (M=2), Corollary 2:
        P_e >= H_b^{-1}(1 - I_MAP),
    where H_b^{-1} is the inverse binary entropy on [0, 1/2]. Positive for all
    I_MAP < 1, i.e., at every finite SNR for BPSK.
    """
    I = np.asarray(I_map, dtype=float)
    target = np.clip(1.0 - I, 0.0, 1.0)   # = H_b(P_e) lower bound

    def Hb(p):
        p = np.clip(p, 1e-12, 1.0 - 1e-12)
        return -p * np.log2(p) - (1 - p) * np.log2(1 - p)

    # Invert H_b on [0, 1/2] by bisection (monotone increasing there).
    lo = np.zeros_like(target)
    hi = np.full_like(target, 0.5)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        too_low = Hb(mid) < target
        lo = np.where(too_low, mid, lo)
        hi = np.where(too_low, hi, mid)
    return 0.5 * (lo + hi)


def compute_pep_union_bound(snr_db_arr, user_k, mod_type, P_arr,
                             n_channel_samples=500000):
    """
    PEP union bound on MAP SER via pairwise error probabilities.

    For each pair of symbols (s, s') for user k, the PEP over the composite
    constellation gives the probability of deciding s' when s was sent.

    SER_k <= (1/M) sum_s sum_{s'!=s} Pr(s -> s' | h)

    For Rayleigh fading:
    PEP(s -> s') = E_h[ Q( |h_k| * d(s,s') / (sqrt(2) * sigma) ) ]

    We use the composite minimum distance: for each pair (s, s'), the
    pairwise distance depends on the interference realization.
    """
    S = get_constellation(mod_type)
    sqrt_P = np.sqrt(P_arr)
    K_total = len(P_arr)
    M = len(S)

    other_users = [j for j in range(K_total) if j != user_k]
    combos = np.array(list(product(range(M), repeat=len(other_users))))
    n_combos = len(combos)

    # Precompute interference for each combo
    interference = np.zeros(n_combos, dtype=complex)
    for i, j in enumerate(other_users):
        interference += sqrt_P[j] * S[combos[:, i]]

    # Composite constellation points for user k
    # For symbol s_idx, the composite points are: sqrt(P_k)*S[s_idx] + interference[c]
    # The MAP detector marginalizes over interference, so the relevant distances
    # are between marginal posteriors.

    # For union bound, we need the effective pairwise distances in the composite
    # constellation. The key distance is:
    # d_eff(s, s') = min over interference combos of |sqrt(P_k)*(S[s] - S[s'])|
    # = sqrt(P_k) * |S[s] - S[s']|  (independent of interference)
    # This is because the interference is the same for both hypotheses.

    d_min_user = sqrt_P[user_k] * np.min(
        [np.abs(S[i] - S[j]) for i in range(M) for j in range(i+1, M)])

    ser_bounds = []

    for snr_db in snr_db_arr:
        sigma2 = 10.0 ** (-snr_db / 10.0)
        sigma = np.sqrt(sigma2)

        # Generate channel samples (seed matches main MI loop)
        np.random.seed(SEED + int(snr_db * 100))
        h_all = (np.random.randn(n_channel_samples, K_total) +
                 1j * np.random.randn(n_channel_samples, K_total)) / np.sqrt(2)
        order = np.argsort(-np.abs(h_all), axis=1)
        h_all = np.take_along_axis(h_all, order, axis=1)
        h_k = h_all[:, user_k]
        h_abs = np.abs(h_k)

        # Union bound: SER <= sum over distinct symbol pairs
        # For each pair (s, s'), PEP = E_h[ Q( |h| * d(s,s') / (sqrt(2)*sigma) ) ]
        # But in NOMA, the effective distance depends on the actual composite constellation.

        # Full composite-constellation-aware PEP:
        # For MAP detector, the decision boundary between s and s' depends on
        # all M^(K-1) interference patterns. We compute the average PEP.

        # Simplified but tight approach: use the pairwise distances from the
        # composite constellation directly.
        total_pep = 0.0
        n_pairs = 0

        for s1 in range(M):
            for s2 in range(s1 + 1, M):
                # Distance between composite constellation clusters
                # d = sqrt(P_k) * |S[s1] - S[s2]|
                d_pair = sqrt_P[user_k] * np.abs(S[s1] - S[s2])

                # PEP over Rayleigh fading: E[ Q(|h|*d / (sqrt(2)*sigma)) ]
                # For Rayleigh |h| with E[|h|^2] = 1 (after ordering, it's different)
                # Use Monte Carlo with actual ordered channels
                arg = h_abs * d_pair / (np.sqrt(2) * sigma)
                pep = np.mean(0.5 * _erfc_approx(arg / np.sqrt(2)))
                total_pep += 2 * pep  # factor 2: s1->s2 and s2->s1
                n_pairs += 2

        # Union bound: SER <= (1/M) * sum of all PEPs
        ser_bound = total_pep / M
        ser_bounds.append(ser_bound)

    return np.array(ser_bounds)


def _erfc_approx(x):
    """Complementary error function using scipy or approximation."""
    # Use torch for GPU-accelerated computation
    x_t = torch.tensor(x, dtype=torch.float64, device='cpu')
    return torch.erfc(x_t).numpy()


def compute_composite_distances(user_k, mod_type, P_arr):
    """
    Compute minimum distance of composite constellation for user k.

    The composite constellation for user k consists of points:
    c = sqrt(P_k) * s_k + sum_{j!=k} sqrt(P_j) * s_j
    for all possible (s_k, s_{-k}) combinations.

    The minimum distance d_min,k is the minimum distance between
    points with DIFFERENT s_k values.
    """
    S = get_constellation(mod_type)
    sqrt_P = np.sqrt(P_arr)
    K_total = len(P_arr)
    M = len(S)

    other_users = [j for j in range(K_total) if j != user_k]
    combos = np.array(list(product(range(M), repeat=len(other_users))))
    n_combos = len(combos)

    # Precompute interference
    interference = np.zeros(n_combos, dtype=complex)
    for i, j in enumerate(other_users):
        interference += sqrt_P[j] * S[combos[:, i]]

    # Build full composite constellation labeled by user k's symbol
    # clusters[s_idx] = set of composite points when x_k = S[s_idx]
    clusters = {}
    for s_idx in range(M):
        clusters[s_idx] = sqrt_P[user_k] * S[s_idx] + interference

    # Minimum inter-cluster distance
    d_min = np.inf
    for s1 in range(M):
        for s2 in range(s1 + 1, M):
            for c1 in clusters[s1]:
                for c2 in clusters[s2]:
                    d = np.abs(c1 - c2)
                    d_min = min(d_min, d)

    # Also compute average inter-cluster distance and intra-cluster spread
    d_avg_inter = 0.0
    count = 0
    for s1 in range(M):
        for s2 in range(s1 + 1, M):
            for c1 in clusters[s1]:
                for c2 in clusters[s2]:
                    d_avg_inter += np.abs(c1 - c2)
                    count += 1
    d_avg_inter /= count

    # Single-user minimum distance (no NOMA)
    d_min_single = sqrt_P[user_k] * np.min(
        [np.abs(S[i] - S[j]) for i in range(M) for j in range(i+1, M)])

    return d_min, d_avg_inter, d_min_single, clusters


# ============================================================
# SER Computation Functions (reusing existing detectors)
# ============================================================

def compute_map_ser(snr_db, mod_type, user_k=0, P_arr=None, n_test=200000):
    """Compute exact MAP SER at a single SNR point."""
    if P_arr is None:
        P_arr = P_DEFAULT
    S = get_constellation(mod_type)
    np.random.seed(SEED + int(snr_db * 100))
    y, h_est, h_true, x_k, x_all, x_k_idx, sigma2 = generate_noma_data(
        n_test, snr_db, mod_type, user_k, P_arr)
    pred = detect_exact_map(y, h_est, user_k, S, sigma2, P_arr)
    return np.mean(pred != x_k_idx)


def compute_sic_ser(snr_db, mod_type, user_k=0, P_arr=None, n_test=200000):
    """Compute conventional SIC SER at a single SNR point."""
    if P_arr is None:
        P_arr = P_DEFAULT
    S = get_constellation(mod_type)
    np.random.seed(SEED + int(snr_db * 100))
    y, h_est, h_true, x_k, x_all, x_k_idx, sigma2 = generate_noma_data(
        n_test, snr_db, mod_type, user_k, P_arr)
    pred = detect_conv_sic(y, h_est, user_k, S, sigma2, P_arr)
    return np.mean(pred != x_k_idx)


def compute_oracle_ser(snr_db, mod_type, user_k=0, P_arr=None, n_test=200000):
    """Compute Oracle SIC SER at a single SNR point."""
    if P_arr is None:
        P_arr = P_DEFAULT
    S = get_constellation(mod_type)
    np.random.seed(SEED + int(snr_db * 100))
    y, h_est, h_true, x_k, x_all, x_k_idx, sigma2 = generate_noma_data(
        n_test, snr_db, mod_type, user_k, P_arr)
    pred = detect_oracle_sic(y, h_est, user_k, x_all, S, sigma2, P_arr)
    return np.mean(pred != x_k_idx)


# ============================================================
# Main Simulation: Compute All IT Quantities
# ============================================================

def run_mi_simulation(mod_type, user_k=0, P_arr=None, n_samples=500000):
    """
    Compute I_TIN, I_MAP, I_oracle vs SNR for one modulation.

    Returns dict with all MI quantities.
    """
    if P_arr is None:
        P_arr = P_DEFAULT
    S = get_constellation(mod_type)
    M = len(S)

    if mod_type == 'BPSK':
        snr_range = np.arange(0, 31, 2)
    elif mod_type == 'QPSK':
        snr_range = np.arange(0, 31, 2)
    else:
        snr_range = np.arange(5, 41, 2)

    results = {
        'I_TIN': [], 'I_MAP': [], 'I_oracle': [],
        'R_Gauss': [], 'H_cond_MAP': [],
        'MAP_SER': [], 'SIC_SER': [], 'Oracle_SER': [],
        'Fano_bound': [], 'PEP_bound': [],
    }

    print(f"\n{'='*60}")
    print(f"  MI Simulation: {mod_type} | User {user_k+1} | M={M}")
    print(f"  P = {P_arr} | n_samples = {n_samples}")
    print(f"{'='*60}")

    # Precompute Gaussian rates
    R_gauss = compute_gaussian_rates(snr_range, user_k, P_arr, n_samples)

    # Precompute PEP union bounds
    print("  Computing PEP union bounds...")
    pep_bounds = compute_pep_union_bound(snr_range, user_k, mod_type, P_arr,
                                          n_channel_samples=n_samples)

    for i, snr_db in enumerate(snr_range):
        t0 = time.time()
        np.random.seed(SEED + int(snr_db * 100))
        y, h_est, h_true, x_k, x_all, x_k_idx, sigma2 = generate_noma_data(
            n_samples, snr_db, mod_type, user_k, P_arr)

        # 1. Compute MAP posteriors and CCMI
        _, posteriors = compute_log_marginal_posteriors(
            y, h_est, user_k, S, sigma2, P_arr)
        I_map = compute_ccmi(posteriors, M)
        H_cond = compute_conditional_entropy(posteriors)

        # 2. Compute Oracle MI
        I_oracle = compute_oracle_mi(y, h_est, user_k, x_all, S, sigma2, P_arr)

        # 3. Compute TIN MI
        I_tin = compute_tin_mi(y, h_est, user_k, S, sigma2, P_arr, x_k_idx)

        # 4. SER values
        pred_map = detect_exact_map(y, h_est, user_k, S, sigma2, P_arr)
        ser_map = np.mean(pred_map != x_k_idx)

        pred_sic = detect_conv_sic(y, h_est, user_k, S, sigma2, P_arr)
        ser_sic = np.mean(pred_sic != x_k_idx)

        pred_oracle = detect_oracle_sic(y, h_est, user_k, x_all, S, sigma2, P_arr)
        ser_oracle = np.mean(pred_oracle != x_k_idx)

        # 5. Fano bound
        fano = compute_fano_bound(I_map, M)

        # Store
        results['I_TIN'].append(I_tin)
        results['I_MAP'].append(I_map)
        results['I_oracle'].append(I_oracle)
        results['R_Gauss'].append(R_gauss[i])
        results['H_cond_MAP'].append(H_cond)
        results['MAP_SER'].append(max(ser_map, 1e-6))
        results['SIC_SER'].append(max(ser_sic, 1e-6))
        results['Oracle_SER'].append(max(ser_oracle, 1e-6))
        results['Fano_bound'].append(fano)
        results['PEP_bound'].append(pep_bounds[i])

        dt = time.time() - t0
        print(f"  SNR={snr_db:2d} dB | I_TIN={I_tin:.4f} | I_MAP={I_map:.4f} | "
              f"I_oracle={I_oracle:.4f} | R_G={R_gauss[i]:.4f} | "
              f"MAP={ser_map:.2e} | ORC={ser_oracle:.2e} | SIC={ser_sic:.2e} | "
              f"Fano={fano:.2e} | PEP={pep_bounds[i]:.2e} | {dt:.1f}s")

    # Convert to arrays
    for key in results:
        results[key] = np.array(results[key])

    return snr_range, results


def run_power_mi_simulation(snr_db_fixed=15, user_k=0, mod_type='QPSK',
                             n_samples=500000):
    """
    Compute I_MAP vs power differentiation at a fixed SNR.
    Validates identifiability theory through IT lens.
    """
    S = get_constellation(mod_type)
    M = len(S)

    # Power configurations: equally spaced in std(P) along simplex path
    # P(t) = P_equal + t * (P_default - P_equal)
    # std(P(t)) = t * 0.1247, giving equally-spaced std values
    # t=0: equal; t=1: default [0.2,0.3,0.5]; t=2.4: extreme [0.013,0.254,0.734]
    P_equal = np.array([1/3, 1/3, 1/3])
    P_default = np.array([0.2, 0.3, 0.5])
    direction = P_default - P_equal
    n_points = 15
    t_max = 2.4
    t_vals = np.linspace(0, t_max, n_points)
    power_configs = []
    for t in t_vals:
        P = P_equal + t * direction
        P = np.maximum(P, 0.001)
        P /= P.sum()
        power_configs.append(P)

    # Power differentiation metric: std of power allocation
    diff_metric = [np.std(P) for P in power_configs]

    results = {
        'P_configs': power_configs,
        'diff_metric': diff_metric,
        'I_MAP': [], 'I_TIN': [], 'I_oracle': [],
    }

    sigma2 = 10.0 ** (-snr_db_fixed / 10.0)

    print(f"\n{'='*60}")
    print(f"  Power-MI Analysis: {mod_type} | SNR={snr_db_fixed} dB | User {user_k+1}")
    print(f"{'='*60}")

    for P_arr in power_configs:
        np.random.seed(SEED)
        y, h_est, h_true, x_k, x_all, x_k_idx, sigma2_val = generate_noma_data(
            n_samples, snr_db_fixed, mod_type, user_k, P_arr)

        _, posteriors = compute_log_marginal_posteriors(
            y, h_est, user_k, S, sigma2_val, P_arr)
        I_map = compute_ccmi(posteriors, M)

        I_oracle = compute_oracle_mi(y, h_est, user_k, x_all, S, sigma2_val, P_arr)
        I_tin = compute_tin_mi(y, h_est, user_k, S, sigma2_val, P_arr, x_k_idx)

        results['I_MAP'].append(I_map)
        results['I_TIN'].append(I_tin)
        results['I_oracle'].append(I_oracle)

        print(f"  P={P_arr} | std={np.std(P_arr):.4f} | "
              f"I_TIN={I_tin:.4f} | I_MAP={I_map:.4f} | I_oracle={I_oracle:.4f}")

    for key in ['I_MAP', 'I_TIN', 'I_oracle']:
        results[key] = np.array(results[key])

    return results


# ============================================================
# Plotting Functions
# ============================================================

def plot_mi_comparison(snr_range, results_dict, filename):
    """
    Plot I_TIN, I_MAP, I_oracle vs SNR for all modulations.
    Figure 1 in Paper 1.
    """
    setup_ieee_style()
    mod_types = list(results_dict.keys())
    n_mods = len(mod_types)

    fig, axes = plt.subplots(1, n_mods, figsize=(3.5 * n_mods, 3.0))
    if n_mods == 1:
        axes = [axes]

    colors = {'I_TIN': '#d62728', 'I_MAP': '#1f77b4', 'I_oracle': '#2ca02c',
              'R_Gauss': '#9467bd'}
    markers = {'I_TIN': 's', 'I_MAP': 'D', 'I_oracle': '^', 'R_Gauss': 'x'}
    labels = {'I_TIN': r'$I_{\mathrm{TIN}}$',
              'I_MAP': r'$I_{\mathrm{MAP}}$',
              'I_oracle': r'$I_{\mathrm{oracle}}$',
              'R_Gauss': r'$R_{\mathrm{Gauss}}$'}

    for ax_idx, mod_type in enumerate(mod_types):
        ax = axes[ax_idx]
        snr = snr_range[mod_type]
        res = results_dict[mod_type]
        M = {'BPSK': 2, 'QPSK': 4, '16QAM': 16}[mod_type]

        # Plot capacity ceiling
        ax.axhline(y=np.log2(M), color='gray', linestyle=':', linewidth=0.8,
                   alpha=0.5, label=f'$\\log_2 M = {np.log2(M):.0f}$')

        for key in ['I_TIN', 'I_MAP', 'I_oracle', 'R_Gauss']:
            ax.plot(snr, res[key], color=colors[key], marker=markers[key],
                    markersize=4, linewidth=1.2, label=labels[key],
                    markerfacecolor='none' if key != 'I_MAP' else colors[key])

        ax.set_xlabel('SNR (dB)')
        if ax_idx == 0:
            ax.set_ylabel('Mutual Information (bits)')
        mod_display = mod_type if mod_type != '16QAM' else '16-QAM'
        ax.set_title(f'{mod_display}')
        ax.legend(fontsize=9, loc='upper left', framealpha=0.9)
        ax.set_ylim(bottom=-0.1)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(filename, format='eps')
    fig.savefig(filename.replace('.eps', '.png'), format='png', dpi=300)
    plt.close(fig)
    print(f"  Saved: {filename}")


def plot_mi_gaps(snr_range, results_dict, filename):
    """
    Plot Delta_struct = I_MAP - I_TIN and Delta_oracle = I_oracle - I_MAP vs SNR.
    Figure 2 in Paper 1.
    """
    setup_ieee_style()
    mod_types = list(results_dict.keys())

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))

    colors_mod = {'BPSK': '#1f77b4', 'QPSK': '#ff7f0e', '16QAM': '#2ca02c'}
    markers_mod = {'BPSK': 'o', 'QPSK': 's', '16QAM': 'D'}

    # Left: Delta_struct = I_MAP - I_TIN
    ax = axes[0]
    for mod_type in mod_types:
        snr = snr_range[mod_type]
        res = results_dict[mod_type]
        delta_struct = res['I_MAP'] - res['I_TIN']
        mod_display = mod_type if mod_type != '16QAM' else '16-QAM'
        ax.plot(snr, delta_struct, color=colors_mod[mod_type],
                marker=markers_mod[mod_type], markersize=4, linewidth=1.2,
                label=mod_display, markerfacecolor='none')
    ax.set_xlabel('SNR (dB)', fontsize=12)
    ax.set_ylabel(r'$\Delta_{\mathrm{struct}}$ (bits)', fontsize=12)
    ax.set_title(r'Structural gain: $I_{\mathrm{MAP}} - I_{\mathrm{TIN}}$', fontsize=12)
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=-0.02)

    # Right: Delta_oracle = I_oracle - I_MAP
    ax = axes[1]
    for mod_type in mod_types:
        snr = snr_range[mod_type]
        res = results_dict[mod_type]
        delta_oracle = res['I_oracle'] - res['I_MAP']
        mod_display = mod_type if mod_type != '16QAM' else '16-QAM'
        ax.plot(snr, delta_oracle, color=colors_mod[mod_type],
                marker=markers_mod[mod_type], markersize=4, linewidth=1.2,
                label=mod_display, markerfacecolor='none')
    ax.set_xlabel('SNR (dB)', fontsize=12)
    ax.set_ylabel(r'$\Delta_{\mathrm{oracle}}$ (bits)', fontsize=12)
    ax.set_title(r'Oracle gap: $I_{\mathrm{oracle}} - I_{\mathrm{MAP}}$', fontsize=12)
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=-0.02)

    plt.tight_layout()
    fig.savefig(filename, format='eps')
    fig.savefig(filename.replace('.eps', '.png'), format='png', dpi=300)
    plt.close(fig)
    print(f"  Saved: {filename}")


def plot_error_bounds(snr_range, results_dict, filename):
    """
    Plot Fano lower bound, PEP upper bound, MAP SER, Oracle SER vs SNR.
    Figure 3 in Paper 1.
    """
    setup_ieee_style()
    mod_types = list(results_dict.keys())
    n_mods = len(mod_types)

    fig, axes = plt.subplots(1, n_mods, figsize=(3.5 * n_mods, 3.0))
    if n_mods == 1:
        axes = [axes]

    for ax_idx, mod_type in enumerate(mod_types):
        ax = axes[ax_idx]
        snr = snr_range[mod_type]
        res = results_dict[mod_type]

        # MAP SER
        ax.semilogy(snr, res['MAP_SER'], 'D-', color='#1f77b4',
                    markersize=4, linewidth=1.2, label='MAP SER')

        # Oracle SER
        ax.semilogy(snr, res['Oracle_SER'], '^-.', color='#2ca02c',
                    markersize=4, linewidth=1.2, label='Oracle SER',
                    markerfacecolor='none')

        # SIC SER
        ax.semilogy(snr, res['SIC_SER'], 's--', color='#d62728',
                    markersize=4, linewidth=1.0, label='Conv. SIC SER',
                    markerfacecolor='none')

        # Fano lower bound. For BPSK (M=2) the symbol-level bound is undefined,
        # so plot the binary-entropy Fano bound of Corollary 2 instead.
        M_mod = {'BPSK': 2, 'QPSK': 4, '16QAM': 16}[mod_type]
        if M_mod == 2:
            bfano = compute_binary_fano_bound(np.asarray(res['I_MAP']))
            valid = bfano > 0
            if np.any(valid):
                ax.semilogy(snr[valid], bfano[valid], 'v:', color='#9467bd',
                            markersize=5, linewidth=1.5,
                            label='Fano lower bound (Cor. 2)')
        else:
            fano = res['Fano_bound']
            valid = fano > 0
            if np.any(valid):
                ax.semilogy(snr[valid], fano[valid], 'v:', color='#9467bd',
                            markersize=5, linewidth=1.5, label='Fano lower bound')

        # Oracle PEP bound (genie-aided benchmark; not an upper bound on MAP SER)
        ax.semilogy(snr, res['PEP_bound'], 'x--', color='#8c564b',
                    markersize=5, linewidth=1.2, label='Oracle PEP bound')

        ax.set_xlabel('SNR (dB)')
        if ax_idx == 0:
            ax.set_ylabel('Symbol Error Rate')
        mod_display = mod_type if mod_type != '16QAM' else '16-QAM'
        ax.set_title(f'{mod_display}')
        # Subplot (a) legend is translucent so the curves behind it show through;
        # remaining panels keep a near-opaque legend.
        ax.legend(fontsize=9, loc='lower left',
                  framealpha=0.5 if ax_idx == 0 else 0.9)
        ax.set_ylim(bottom=5e-6, top=1.5)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(filename, format='eps')
    fig.savefig(filename.replace('.eps', '.png'), format='png', dpi=300)
    # Also save a PDF: the EPS backend flattens the translucent subplot-(a)
    # legend, so the manuscript includes this figure as PDF to keep transparency.
    fig.savefig(filename.replace('.eps', '.pdf'), format='pdf')
    plt.close(fig)
    print(f"  Saved: {filename}")


def plot_fa_capacity_gap(snr_range, results_dict, filename):
    """
    Plot R_Gauss vs R_FA (CCMI) vs SNR — finite-alphabet capacity gap.
    Figure 4 in Paper 1.
    """
    setup_ieee_style()
    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    colors_mod = {'BPSK': '#1f77b4', 'QPSK': '#ff7f0e', '16QAM': '#2ca02c'}

    for mod_type in results_dict:
        snr = snr_range[mod_type]
        res = results_dict[mod_type]
        M = {'BPSK': 2, 'QPSK': 4, '16QAM': 16}[mod_type]
        mod_display = mod_type if mod_type != '16QAM' else '16-QAM'

        # Gaussian rate (dashed)
        ax.plot(snr, res['R_Gauss'], '--', color=colors_mod[mod_type],
                linewidth=1.0, label=f'{mod_display} Gaussian')

        # CCMI (solid)
        ax.plot(snr, res['I_MAP'], '-', color=colors_mod[mod_type],
                marker='o', markersize=3, linewidth=1.2,
                label=f'{mod_display} CCMI ($I_{{\\mathrm{{MAP}}}}$)')

        # Ceiling
        ax.axhline(y=np.log2(M), color=colors_mod[mod_type],
                   linestyle=':', linewidth=0.5, alpha=0.4)

    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('Rate (bits/channel use)')
    ax.set_title('Finite-Alphabet vs. Gaussian Capacity')
    ax.legend(fontsize=5.5, loc='upper left', framealpha=0.9, ncol=1)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=-0.1)

    fig.savefig(filename, format='eps')
    fig.savefig(filename.replace('.eps', '.png'), format='png', dpi=300)
    plt.close(fig)
    print(f"  Saved: {filename}")


def plot_power_mi(results, filename):
    """
    Plot I_MAP vs power differentiation at fixed SNR.
    Figure 5 in Paper 1: IT validation of identifiability.
    """
    setup_ieee_style()
    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    diff = results['diff_metric']

    ax.plot(diff, results['I_MAP'], 'D-', color='#1f77b4',
            markersize=5, linewidth=1.2, label=r'$I_{\mathrm{MAP}}$')
    ax.plot(diff, results['I_TIN'], 's--', color='#d62728',
            markersize=4, linewidth=1.0, label=r'$I_{\mathrm{TIN}}$',
            markerfacecolor='none')
    ax.plot(diff, results['I_oracle'], '^-.', color='#2ca02c',
            markersize=4, linewidth=1.0, label=r'$I_{\mathrm{oracle}}$',
            markerfacecolor='none')

    # Annotate key power configs: equal (first), default ~[0.2,0.3,0.5], extreme (last)
    n_pts = len(results['P_configs'])
    # Index closest to t=1 (default): t_vals has equal spacing, t=1 is at index ~n/t_max
    idx_default = min(range(n_pts), key=lambda i: abs(np.std(results['P_configs'][i]) - 0.1247))
    for i, P in enumerate(results['P_configs']):
        if i in [0, idx_default, n_pts - 1]:
            # Round to 2 decimals but absorb the rounding residual into the
            # largest component so the displayed vector sums to exactly 1.00.
            r = [round(float(p), 2) for p in P]
            resid = round(1.0 - sum(r), 2)
            jmax = int(np.argmax(P))
            r[jmax] = round(r[jmax] + resid, 2)
            label = f'({r[0]:.2f},{r[1]:.2f},{r[2]:.2f})'
            ax.annotate(label, (diff[i], results['I_MAP'][i]),
                       textcoords="offset points", xytext=(5, 8),
                       fontsize=7.5, color='#1f77b4')

    ax.set_xlabel('Power Differentiation (std of $\\mathbf{P}$)', fontsize=8)
    ax.set_ylabel('Mutual Information (bits)', fontsize=8)
    ax.set_title('QPSK, SNR = 15 dB: MI vs. Power Allocation', fontsize=8)
    ax.legend(fontsize=7, loc='best', framealpha=0.5)
    ax.grid(True, alpha=0.3)

    fig.savefig(filename, format='eps')
    fig.savefig(filename.replace('.eps', '.png'), format='png', dpi=300)
    # Also save a PDF: the EPS backend flattens the translucent legend, so the
    # manuscript includes this figure as PDF to preserve transparency.
    fig.savefig(filename.replace('.eps', '.pdf'), format='pdf')
    plt.close(fig)
    print(f"  Saved: {filename}")


def plot_constellation_distance(filename):
    """
    Plot d_min,k(P) analysis and its connection to MI saturation.
    Figure 6 in Paper 1.
    """
    setup_ieee_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))

    mod_types = ['BPSK', 'QPSK', '16QAM']
    colors_mod = {'BPSK': '#1f77b4', 'QPSK': '#ff7f0e', '16QAM': '#2ca02c'}

    # Left: d_min vs power differentiation for all modulations
    ax = axes[0]
    power_configs = [
        np.array([1/3, 1/3, 1/3]),
        np.array([0.30, 0.33, 0.37]),
        np.array([0.25, 0.35, 0.40]),
        np.array([0.25, 0.33, 0.42]),
        np.array([0.20, 0.30, 0.50]),
        np.array([0.15, 0.25, 0.60]),
        np.array([0.10, 0.20, 0.70]),
        np.array([0.05, 0.15, 0.80]),
    ]
    diff_metric = [np.std(P) for P in power_configs]

    for mod_type in mod_types:
        d_mins = []
        for P in power_configs:
            d_min, _, _, _ = compute_composite_distances(0, mod_type, P)
            d_mins.append(d_min)
        mod_display = mod_type if mod_type != '16QAM' else '16-QAM'
        ax.plot(diff_metric, d_mins, 'o-', color=colors_mod[mod_type],
                markersize=4, linewidth=1.2, label=mod_display)

    ax.set_xlabel('Power Differentiation (std of $\\mathbf{P}$)')
    ax.set_ylabel(r'$d_{\min,1}(\mathbf{P})$')
    ax.set_title('Composite Minimum Distance')
    ax.legend(fontsize=7, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # Right: d_min ratio (composite / single-user) vs power
    ax = axes[1]
    for mod_type in mod_types:
        ratios = []
        for P in power_configs:
            d_min, _, d_single, _ = compute_composite_distances(0, mod_type, P)
            ratios.append(d_min / d_single if d_single > 0 else 0)
        mod_display = mod_type if mod_type != '16QAM' else '16-QAM'
        ax.plot(diff_metric, ratios, 'o-', color=colors_mod[mod_type],
                markersize=4, linewidth=1.2, label=mod_display)

    ax.set_xlabel('Power Differentiation (std of $\\mathbf{P}$)')
    ax.set_ylabel(r'$d_{\min,1}(\mathbf{P}) / d_{\min,1}^{\mathrm{single}}$')
    ax.set_title('Distance Ratio (vs. Single-User)')
    ax.legend(fontsize=7, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1.0, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)

    plt.tight_layout()
    fig.savefig(filename, format='eps')
    fig.savefig(filename.replace('.eps', '.png'), format='png', dpi=300)
    plt.close(fig)
    print(f"  Saved: {filename}")


# ============================================================
# Main Execution
# ============================================================

def main():
    t0 = time.time()
    n_samples = 4000000  # Monte Carlo samples per SNR point (R1: 4M for max rigor)

    # ============================================================
    # Step 1: MI Simulations for All Modulations
    # ============================================================
    all_snr = {}
    all_results = {}

    for mod_type in ['BPSK', 'QPSK', '16QAM']:
        snr_range, results = run_mi_simulation(
            mod_type, user_k=0, P_arr=P_DEFAULT, n_samples=n_samples)
        all_snr[mod_type] = snr_range
        all_results[mod_type] = results

    # ============================================================
    # Step 2: Verification
    # ============================================================
    print("\n" + "="*60)
    print("  VERIFICATION: MI Ordering I_TIN <= I_MAP <= I_oracle")
    print("="*60)

    all_ok = True
    for mod_type in ['BPSK', 'QPSK', '16QAM']:
        res = all_results[mod_type]
        snr = all_snr[mod_type]
        M = {'BPSK': 2, 'QPSK': 4, '16QAM': 16}[mod_type]

        for i, snr_db in enumerate(snr):
            tin_ok = res['I_TIN'][i] <= res['I_MAP'][i] + 1e-6
            map_ok = res['I_MAP'][i] <= res['I_oracle'][i] + 1e-6
            ceiling_ok = res['I_MAP'][i] <= np.log2(M) + 1e-6

            if not (tin_ok and map_ok and ceiling_ok):
                print(f"  FAIL: {mod_type} SNR={snr_db}: "
                      f"I_TIN={res['I_TIN'][i]:.4f}, I_MAP={res['I_MAP'][i]:.4f}, "
                      f"I_oracle={res['I_oracle'][i]:.4f}")
                all_ok = False

        # Check high-SNR convergence
        if res['I_MAP'][-1] < np.log2(M) * 0.9:
            print(f"  WARNING: {mod_type} I_MAP at max SNR = {res['I_MAP'][-1]:.4f} "
                  f"< 0.9 * log2(M) = {np.log2(M)*0.9:.4f}")

    if all_ok:
        print("  ALL CHECKS PASSED: MI ordering holds at every SNR point.")

    # Verify Fano bound
    print("\n  VERIFICATION: Fano bound <= MAP SER")
    fano_ok = True
    for mod_type in ['BPSK', 'QPSK', '16QAM']:
        res = all_results[mod_type]
        for i in range(len(all_snr[mod_type])):
            if res['Fano_bound'][i] > 0 and res['Fano_bound'][i] > res['MAP_SER'][i] + 1e-3:
                print(f"  FAIL: {mod_type} SNR={all_snr[mod_type][i]}: "
                      f"Fano={res['Fano_bound'][i]:.4f} > MAP={res['MAP_SER'][i]:.4f}")
                fano_ok = False
    if fano_ok:
        print("  ALL CHECKS PASSED: Fano bound <= MAP SER at every point.")

    # ============================================================
    # Step 3: Generate Figures
    # ============================================================
    print("\n" + "="*60)
    print("  GENERATING FIGURES")
    print("="*60)

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
    print("\n" + "="*60)
    print("  POWER-MI ANALYSIS")
    print("="*60)
    power_results = run_power_mi_simulation(snr_db_fixed=15, n_samples=n_samples)
    plot_power_mi(power_results,
                  os.path.join(OUT_DIR, 'fig_it_power_mi.eps'))

    # ------------------------------------------------------------
    # Persist all Monte Carlo results so the figures can be
    # re-plotted (e.g., reformatted) later without re-running the
    # simulation. Load with replot_it_bounds.py.
    # ------------------------------------------------------------
    import pickle
    cache_path = os.path.join(OUT_DIR, 'sim_it_bounds_results.pkl')
    with open(cache_path, 'wb') as f:
        pickle.dump({'all_snr': all_snr,
                     'all_results': all_results,
                     'power_results': power_results,
                     'n_samples': n_samples,
                     'P': P_DEFAULT,
                     'SEED': SEED}, f)
    print(f"\n  Saved all results to {cache_path} "
          f"(re-plot with replot_it_bounds.py, no simulation needed)")

    # Figure 6: Constellation distance analysis
    print("\n" + "="*60)
    print("  CONSTELLATION DISTANCE ANALYSIS")
    print("="*60)
    plot_constellation_distance(
        os.path.join(OUT_DIR, 'fig_it_constellation_distance.eps'))

    # ============================================================
    # Step 4: Summary Table
    # ============================================================
    print("\n" + "="*60)
    print("  SUMMARY: Key IT Quantities")
    print("="*60)

    for mod_type in ['BPSK', 'QPSK', '16QAM']:
        res = all_results[mod_type]
        snr = all_snr[mod_type]
        M = {'BPSK': 2, 'QPSK': 4, '16QAM': 16}[mod_type]

        # Pick mid-SNR and high-SNR reference points
        mid_idx = len(snr) // 2
        high_idx = -1

        print(f"\n  {mod_type} (M={M}):")
        print(f"    Mid SNR={snr[mid_idx]} dB:")
        print(f"      I_TIN={res['I_TIN'][mid_idx]:.4f}, "
              f"I_MAP={res['I_MAP'][mid_idx]:.4f}, "
              f"I_oracle={res['I_oracle'][mid_idx]:.4f}")
        print(f"      Delta_struct={res['I_MAP'][mid_idx]-res['I_TIN'][mid_idx]:.4f}, "
              f"Delta_oracle={res['I_oracle'][mid_idx]-res['I_MAP'][mid_idx]:.4f}")
        print(f"    High SNR={snr[high_idx]} dB:")
        print(f"      I_TIN={res['I_TIN'][high_idx]:.4f}, "
              f"I_MAP={res['I_MAP'][high_idx]:.4f}, "
              f"I_oracle={res['I_oracle'][high_idx]:.4f}, "
              f"log2(M)={np.log2(M):.4f}")

    # Composite distance summary
    print("\n  Composite Constellation Distances (P = [0.2, 0.3, 0.5]):")
    for mod_type in ['BPSK', 'QPSK', '16QAM']:
        d_min, d_avg, d_single, _ = compute_composite_distances(0, mod_type, P_DEFAULT)
        mod_display = mod_type if mod_type != '16QAM' else '16-QAM'
        print(f"    {mod_display}: d_min={d_min:.4f}, d_single={d_single:.4f}, "
              f"ratio={d_min/d_single:.4f}")

    elapsed = time.time() - t0
    print(f"\n  Total runtime: {elapsed/60:.1f} minutes")
    print("  Done.")


if __name__ == '__main__':
    main()
