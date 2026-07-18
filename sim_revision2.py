#!/usr/bin/env python3
"""
Paper 1 R2 revision additions (Reviewer 4, second round).

Part A [R4-1] Nearest-neighbor multiplicity factor.
    Deterministic count on the composite constellation. Reports, for
    BPSK/QPSK/16-QAM at K=3 and P=[0.2,0.3,0.5], the average number of
    minimum-distance neighbors of user 1 in the composite constellation
    vs. the single-user constellation, the ratio beta = N_comp/N_su, and
    the distance ratio rho_1. These quantify the multiplicity factor that
    multiplies the rho^{-2d} distance term in the MAP/oracle SER ratio
    (Proposition 5).

Part B [R4-2] Spatially correlated MIMO channels.
    Exponential (Kronecker) receive correlation R_r = [r^{|i-j|}].
    Sweeps the MI hierarchy vs SNR for Nr in {3,4} at r in {0,0.7,0.9} to
    test whether the Nr>=K convergence survives correlation. Produces
    fig_mimo_correlation and a compact summary table.

Part C [R4-4] Approximate-MAP (Expectation Propagation) detector.
    EP is the representative "message passing" approximate-MAP receiver;
    sphere decoding attains exact MAP and coincides with the MAP curve.
    Adds the EP SER curve to fig_mimo_ser_comparison for Nr=2 and Nr=4
    (QPSK, K=3), reusing the cached 4M MMSE-SIC/MAP/Oracle curves so the
    previously reported values are unchanged.

Author: Liang Dong, Baylor University
"""

import os
import time
import pickle
from itertools import product

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['pdf.fonttype'] = 42
import warnings
warnings.filterwarnings('ignore')

from sim_mimo_noma import (
    get_constellation, detect_mmse_sic, detect_oracle_sic, detect_exact_map,
    K, P_DEFAULT, SEED, DEVICE, setup_ieee,
)
from sim_mimo_it_bounds import (
    compute_mimo_map_posteriors, compute_mimo_ccmi,
    compute_mimo_oracle_mi, compute_mimo_mmse_mi,
)

OUT_DIR = os.environ.get('NOMA_OUT_DIR', '.')
print(f"Using device: {DEVICE}")


# ============================================================
# Part A: Nearest-neighbor multiplicity factor (deterministic)
# ============================================================
def multiplicity_factor(mod_type, P_arr, user_k=0, tol=1e-3):
    """Average minimum-distance neighbor multiplicity in the composite
    constellation vs the single-user constellation, for user_k."""
    S = get_constellation(mod_type)
    M = len(S)
    Kt = len(P_arr)
    sqrt_P = np.sqrt(P_arr)

    # Composite constellation c(s) = sum_j sqrt(P_j) s_j over all M^K vectors.
    combos = np.array(list(product(range(M), repeat=Kt)))          # (M^K, K)
    c = (sqrt_P[None, :] * S[combos]).sum(axis=1)                   # (M^K,)
    sk = combos[:, user_k]

    # Pairwise distances (M^K x M^K) -- fine for M^K <= 4096.
    D = np.abs(c[:, None] - c[None, :])
    diff_k = (sk[:, None] != sk[None, :])                          # differ in x_k
    # Composite minimum distance among pairs that differ in x_k.
    d_min = D[diff_k].min()
    thr = d_min * (1.0 + tol)
    # Neighbor count per point at (approx) the composite minimum distance,
    # restricted to competitors that differ in x_k (the SER-relevant events).
    N_comp = np.mean(((D <= thr) & diff_k).sum(axis=1))

    # Single-user constellation for user_k: sqrt(P_k) * S.
    su = sqrt_P[user_k] * S
    Dsu = np.abs(su[:, None] - su[None, :])
    np.fill_diagonal(Dsu, np.inf)
    d_su = Dsu.min()
    N_su = np.mean((Dsu <= d_su * (1.0 + tol)).sum(axis=1))

    rho = d_min / d_su
    beta = N_comp / N_su
    return dict(mod=mod_type, M=M, d_min=d_min, d_su=d_su, rho=rho,
                N_comp=N_comp, N_su=N_su, beta=beta)


def run_part_A():
    print("\n" + "=" * 64)
    print("  PART A [R4-1]: Nearest-neighbor multiplicity factor")
    print(f"  K={K}, P={P_DEFAULT}, user 1")
    print("=" * 64)
    print(f"  {'mod':7s} {'M^K':>6s} {'rho_1':>8s} {'N_comp':>8s} "
          f"{'N_su':>6s} {'beta':>7s} {'beta*rho^-2K':>13s}")
    rows = []
    for mod in ['BPSK', 'QPSK', '16QAM']:
        r = multiplicity_factor(mod, P_DEFAULT, user_k=0)
        d = K  # user 1 diversity order = K
        ratio = r['beta'] * r['rho'] ** (-2 * d)
        rows.append(r)
        print(f"  {mod:7s} {r['M']**K:6d} {r['rho']:8.4f} {r['N_comp']:8.3f} "
              f"{r['N_su']:6.2f} {r['beta']:7.3f} {ratio:13.3e}")
    return rows


# ============================================================
# Part B: Spatially correlated MIMO channels
# ============================================================
def corr_sqrt(Nr, r):
    """Matrix square root of the exponential correlation R=[r^|i-j|]."""
    idx = np.arange(Nr)
    R = r ** np.abs(idx[:, None] - idx[None, :])
    w, V = np.linalg.eigh(R)
    return V @ np.diag(np.sqrt(np.maximum(w, 0))) @ V.conj().T


def generate_mimo_data_corr(n_samples, snr_db, mod_type, Nr, r,
                            Kt=3, P_arr=None):
    """Uplink MIMO-NOMA with exponential receive correlation r."""
    if P_arr is None:
        P_arr = P_DEFAULT
    sqrt_P = np.sqrt(P_arr)
    S = get_constellation(mod_type)
    M = len(S)
    sigma2 = 10.0 ** (-snr_db / 10.0)

    Rh = corr_sqrt(Nr, r).astype(np.complex64)           # (Nr, Nr)
    G = (np.random.randn(n_samples, Nr, Kt) +
         1j * np.random.randn(n_samples, Nr, Kt)) / np.sqrt(2)
    # Color the receive dimension: H[:, :, k] = Rh @ G[:, :, k].
    H = np.einsum('ab,nbk->nak', Rh, G)                  # (n, Nr, K)

    x_idx = np.random.randint(0, M, size=(n_samples, Kt))
    x_all = S[x_idx]
    x_w = sqrt_P[None, :] * x_all
    y = np.einsum('ijk,ik->ij', H, x_w)
    noise = np.sqrt(sigma2 / 2) * (np.random.randn(n_samples, Nr) +
                                   1j * np.random.randn(n_samples, Nr))
    return y + noise, H, x_all, x_idx, sigma2


def run_part_B(n_samples=1500000):
    print("\n" + "=" * 64)
    print("  PART B [R4-2]: Spatially correlated MIMO channels")
    print("=" * 64)
    mod = 'QPSK'
    S = get_constellation(mod)
    M = len(S)
    snr_range = np.arange(0, 31, 2)  # step 2 so the 16 dB reference is on-grid
    r_vals = [0.0, 0.7, 0.9]
    Nr_vals = [3, 4]

    out = {}  # (Nr, r) -> dict of arrays
    for Nr in Nr_vals:
        for r in r_vals:
            res = {'I_MMSE': [], 'I_MAP': [], 'I_oracle': []}
            for snr_db in snr_range:
                np.random.seed(SEED + int(snr_db * 100) + int(r * 1000) + Nr)
                y, H, x_all, x_idx, sigma2 = generate_mimo_data_corr(
                    n_samples, snr_db, mod, Nr, r)
                xk = x_idx[:, 0]
                post = compute_mimo_map_posteriors(y, H, 0, S, sigma2, P_DEFAULT)
                I_map = compute_mimo_ccmi(post, M)
                I_orc = compute_mimo_oracle_mi(y, H, 0, x_all, S, sigma2, P_DEFAULT)
                I_mmse = compute_mimo_mmse_mi(y, H, 0, S, sigma2, P_DEFAULT, xk)
                res['I_MMSE'].append(I_mmse)
                res['I_MAP'].append(I_map)
                res['I_oracle'].append(I_orc)
                torch.cuda.empty_cache()
            for k in res:
                res[k] = np.array(res[k])
            out[(Nr, r)] = res
            i16 = int(np.argmin(np.abs(snr_range - 16)))
            print(f"  Nr={Nr} r={r:.1f} @16dB: I_MMSE={res['I_MMSE'][i16]:.3f} "
                  f"I_MAP={res['I_MAP'][i16]:.3f} I_orc={res['I_oracle'][i16]:.3f} "
                  f"| MAP-MMSE gap={res['I_MAP'][i16]-res['I_MMSE'][i16]:.3f}")
    plot_correlation(snr_range, out, r_vals,
                     os.path.join(OUT_DIR, 'fig_mimo_correlation.eps'))
    with open(os.path.join(OUT_DIR, 'sim_revision2_corr.pkl'), 'wb') as f:
        pickle.dump({'snr': snr_range, 'out': out, 'r_vals': r_vals,
                     'n_samples': n_samples}, f)
    return snr_range, out


def plot_correlation(snr, out, r_vals, filename):
    setup_ieee()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    colors = {0.0: '#1f77b4', 0.7: '#ff7f0e', 0.9: '#d62728'}
    # Left: Nr=3 (=K) MI hierarchy, iid vs correlated.
    ax = axes[0]
    for r in r_vals:
        res = out[(3, r)]
        ax.plot(snr, res['I_MMSE'], 's--', color=colors[r], markersize=4,
                linewidth=1.1, markerfacecolor='none',
                label=fr'$I_{{\mathrm{{MMSE}}}}$, $r={r}$')
        ax.plot(snr, res['I_MAP'], 'D-', color=colors[r], markersize=4,
                linewidth=1.2, label=fr'$I_{{\mathrm{{MAP}}}}$, $r={r}$')
    ax.axhline(2.0, color='gray', ls=':', lw=0.8, alpha=0.5)
    ax.set_xlabel('SNR (dB)', fontsize=12)
    ax.set_ylabel('Mutual Information (bits)', fontsize=12)
    ax.set_title(r'$N_r = K = 3$', fontsize=12)
    ax.legend(fontsize=8.5, loc='lower right', framealpha=0.5, ncol=1)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=-0.05)
    # Right: MAP-MMSE gap vs correlation r at 16 dB, for Nr=3 and Nr=4.
    ax = axes[1]
    i16 = int(np.argmin(np.abs(snr - 16)))
    for Nr, mk, c in [(3, 'D-', '#d62728'), (4, 'o-', '#1f77b4')]:
        gaps = [out[(Nr, r)]['I_MAP'][i16] - out[(Nr, r)]['I_MMSE'][i16]
                for r in r_vals]
        ax.plot(r_vals, gaps, mk, color=c, markersize=5, linewidth=1.2,
                label=fr'$N_r={Nr}$')
    ax.set_xlabel('Receive correlation $r$', fontsize=12)
    ax.set_ylabel(r'$I_{\mathrm{MAP}}-I_{\mathrm{MMSE}}$ (bits), 16 dB',
                  fontsize=10)
    ax.set_title('MMSE gap vs correlation', fontsize=12)
    ax.legend(fontsize=11, framealpha=0.5)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(filename, format='eps')
    fig.savefig(filename.replace('.eps', '.png'), format='png', dpi=300)
    fig.savefig(filename.replace('.eps', '.pdf'), format='pdf')
    plt.close(fig)
    print(f"  Saved: {filename}")


# ============================================================
# Part C: Expectation-Propagation approximate-MAP detector
# ============================================================
def detect_ep(y, H, user_k, S, sigma2, P_arr, n_iter=10, damping=0.2,
              batch_size=20000):
    """Expectation-Propagation MIMO detector (complex domain).

    Approximate marginal MAP by iterated Gaussian moment matching against
    the discrete prior on S. Returns hard decisions for user_k.
    """
    sqrt_P = np.sqrt(P_arr).astype(np.float32)
    Kt = len(P_arr)
    M = len(S)
    n = len(y)
    Nr = H.shape[1]
    Es = float(np.mean(np.abs(S) ** 2))          # symbol energy (=1 here)

    S_t = torch.tensor(S, dtype=torch.complex64, device=DEVICE)
    res = np.zeros(n, dtype=int)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        nb = end - start
        yb = torch.tensor(y[start:end], dtype=torch.complex64, device=DEVICE)
        Hb = torch.tensor(H[start:end], dtype=torch.complex64, device=DEVICE)
        spt = torch.tensor(sqrt_P, device=DEVICE)
        A = Hb * spt.view(1, 1, Kt)              # (nb, Nr, K)
        Ah = A.conj().transpose(1, 2)            # (nb, K, Nr)
        AhA = torch.bmm(Ah, A)                   # (nb, K, K)
        Ahy = torch.bmm(Ah, yb.unsqueeze(-1)).squeeze(-1)   # (nb, K)
        eyeK = torch.eye(Kt, device=DEVICE, dtype=torch.complex64).unsqueeze(0)

        # EP factor natural params: precision gamma (real>0), precision-mean lam (cplx).
        gamma = torch.full((nb, Kt), 1.0 / Es, device=DEVICE)          # real
        lam = torch.zeros((nb, Kt), dtype=torch.complex64, device=DEVICE)
        mcav_k = None
        vcav_k = None

        for _ in range(n_iter):
            Prec = AhA / sigma2 + torch.diag_embed(gamma.to(torch.complex64))
            b = Ahy / sigma2 + lam
            Sigma = torch.linalg.solve(Prec, eyeK)                     # (nb,K,K)
            mu = torch.bmm(Sigma, b.unsqueeze(-1)).squeeze(-1)         # (nb,K)
            Sii = torch.diagonal(Sigma, dim1=1, dim2=2).real           # (nb,K)
            Sii = torch.clamp(Sii, min=1e-9)

            prec_post = 1.0 / Sii                                       # (nb,K) real
            precmean_post = mu / Sii.to(torch.complex64)               # (nb,K) cplx
            prec_cav = prec_post - gamma                               # real
            prec_cav = torch.clamp(prec_cav, min=1e-6)
            precmean_cav = precmean_post - lam
            vcav = 1.0 / prec_cav                                       # real
            mcav = precmean_cav * vcav.to(torch.complex64)            # cplx

            # Moment match against uniform prior on S.
            d2 = torch.abs(mcav.unsqueeze(-1) - S_t.view(1, 1, M)) ** 2  # (nb,K,M)
            logw = -d2 / vcav.unsqueeze(-1)
            logw = logw - logw.max(dim=-1, keepdim=True).values
            w = torch.exp(logw)
            w = w / w.sum(dim=-1, keepdim=True)                        # (nb,K,M)
            mu_hat = (w.to(torch.complex64) * S_t.view(1, 1, M)).sum(-1)  # (nb,K)
            e2 = torch.abs(S_t.view(1, 1, M) - mu_hat.unsqueeze(-1)) ** 2
            v_hat = (w * e2).sum(-1)                                    # (nb,K) real
            v_hat = torch.clamp(v_hat, min=1e-8)

            prec_f_new = 1.0 / v_hat - prec_cav                        # real
            precmean_f_new = mu_hat / v_hat.to(torch.complex64) - precmean_cav
            valid = prec_f_new > 1e-6                                  # keep old if not
            gamma_new = torch.where(valid, prec_f_new, gamma)
            lam_new = torch.where(valid, precmean_f_new, lam)

            gamma = (1 - damping) * gamma + damping * gamma_new
            lam = (1 - damping) * lam + damping * lam_new
            mcav_k, vcav_k = mcav[:, user_k], vcav[:, user_k]

        # Final decision: marginal posterior of user_k from last cavity.
        d2k = torch.abs(mcav_k.unsqueeze(-1) - S_t.view(1, M)) ** 2
        logw = -d2k / vcav_k.unsqueeze(-1)
        res[start:end] = logw.argmax(dim=-1).cpu().numpy()
    return res


def run_part_C(n_samples=4000000):
    print("\n" + "=" * 64)
    print("  PART C [R4-4]: EP approximate-MAP SER (adds curve to Fig. 9)")
    print("=" * 64)
    mod = 'QPSK'
    S = get_constellation(mod)
    cache = pickle.load(open(os.path.join(OUT_DIR, 'sim_mimo_results.pkl'), 'rb'))
    all_snr, all_res = cache['all_snr'], cache['all_results']

    ep_ser = {}
    for Nr in [2]:  # EP shown for the overloaded case R4 asked about (Nr=2<K).
        snr_range = all_snr[Nr]
        ser = []
        for snr_db in snr_range:
            # Reproduce the exact realizations used by the cached 4M run.
            np.random.seed(SEED + int(snr_db * 100))
            sigma2 = 10.0 ** (-snr_db / 10.0)
            Hc = (np.random.randn(n_samples, Nr, K) +
                  1j * np.random.randn(n_samples, Nr, K)) / np.sqrt(2)
            xi = np.random.randint(0, len(S), size=(n_samples, K))
            xa = S[xi]
            xw = np.sqrt(P_DEFAULT)[None, :] * xa
            y = np.einsum('ijk,ik->ij', Hc, xw)
            noise = np.sqrt(sigma2 / 2) * (np.random.randn(n_samples, Nr) +
                                           1j * np.random.randn(n_samples, Nr))
            y = y + noise
            p_ep = detect_ep(y, Hc, 0, S, sigma2, P_DEFAULT)
            s = np.mean(p_ep != xi[:, 0])
            ser.append(max(s, 1e-6))
            torch.cuda.empty_cache()
        ep_ser[Nr] = np.array(ser)
        i16 = int(np.argmin(np.abs(snr_range - 16)))
        print(f"  Nr={Nr}: EP SER @16dB={ep_ser[Nr][i16]:.3e} "
              f"| MMSE-SIC={all_res[Nr]['SER_MMSE_SIC'][i16]:.3e} "
              f"| MAP={all_res[Nr]['SER_MAP'][i16]:.3e} "
              f"| @30dB EP={ep_ser[Nr][-1]:.3e} MAP={all_res[Nr]['SER_MAP'][-1]:.3e}")

    plot_ser_with_ep(all_snr, all_res, ep_ser,
                     os.path.join(OUT_DIR, 'fig_mimo_ser_comparison.eps'))
    with open(os.path.join(OUT_DIR, 'sim_revision2_ep.pkl'), 'wb') as f:
        pickle.dump({'ep_ser': ep_ser, 'all_snr': all_snr}, f)
    return ep_ser


def plot_ser_with_ep(all_snr, all_res, ep_ser, filename):
    """Regenerate fig_mimo_ser_comparison with the EP curve added."""
    setup_ieee()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    det_styles = [
        ('SER_MMSE', 'v', 'gray', ':', 'MMSE'),
        ('SER_MMSE_SIC', 's', '#d62728', '--', 'MMSE-SIC'),
        ('SER_Oracle', '^', '#2ca02c', '-.', 'Oracle'),
        ('SER_MAP', 'D', '#1f77b4', '-', 'Exact MAP'),
    ]
    for ax_idx, Nr in enumerate([2, 4]):
        ax = axes[ax_idx]
        snr = all_snr[Nr]
        res = all_res[Nr]
        for key, m, c, ls, label in det_styles:
            ax.semilogy(snr, np.maximum(res[key], 1e-6), marker=m, color=c,
                        linestyle=ls, linewidth=1.2, label=label,
                        markersize=4, markerfacecolor='none')
        # EP approximate-MAP curve (overloaded panel only).
        if Nr in ep_ser:
            ax.semilogy(snr, np.maximum(ep_ser[Nr], 1e-6), marker='o',
                        color='#ff7f0e', linestyle='-', linewidth=1.2,
                        label='EP (approx. MAP)', markersize=4,
                        markerfacecolor='none')
        ax.set_xlabel('SNR (dB)', fontsize=12)
        if ax_idx == 0:
            ax.set_ylabel('SER', fontsize=12)
        ax.set_title(f'QPSK, $N_r = {Nr}$', fontsize=12)
        ax.legend(fontsize=9.5, loc='lower left', framealpha=0.5)
        ax.set_ylim(bottom=5e-6, top=1.5)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(filename, format='eps')
    fig.savefig(filename.replace('.eps', '.png'), format='png', dpi=300)
    fig.savefig(filename.replace('.eps', '.pdf'), format='pdf')
    plt.close(fig)
    print(f"  Saved: {filename}")


if __name__ == '__main__':
    t0 = time.time()
    run_part_A()
    run_part_B()
    run_part_C()
    print(f"\n  Total runtime: {(time.time()-t0)/60:.1f} min")
    print("  Done.")
