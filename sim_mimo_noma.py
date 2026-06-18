#!/usr/bin/env python3
"""
MIMO-NOMA simulation: Uplink with multi-antenna BS.
Generates figures for the MIMO extension of the paper.
"""
import sys
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from itertools import product
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time, os, warnings
warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

K = 3
P_DEFAULT = np.array([0.2, 0.3, 0.5])
OUT_DIR = '/home/dong/Workspace/WritePaper/noma_sic'


def get_constellation(mod_type):
    if mod_type == 'BPSK':
        return np.array([1.0+0j, -1.0+0j])
    elif mod_type == 'QPSK':
        return np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
    elif mod_type == '16QAM':
        vals = [-3, -1, 1, 3]
        pts = np.array([a+1j*b for a in vals for b in vals])
        return pts / np.sqrt(np.mean(np.abs(pts)**2))
    else:
        raise ValueError(f"Unknown: {mod_type}")


# ============================================================
# Data Generation
# ============================================================
def generate_mimo_data(n_samples, snr_db, mod_type, Nr, K=3, P_arr=None):
    """
    Uplink MIMO-NOMA: y = sum_k h_k sqrt(P_k) x_k + n
    y in C^Nr, h_k in C^Nr (independent Rayleigh per user).
    Returns: y (n,Nr), H (n,Nr,K), x_all (n,K), x_indices (n,K), sigma2
    """
    if P_arr is None:
        P_arr = P_DEFAULT
    sqrt_P = np.sqrt(P_arr)
    S = get_constellation(mod_type)
    M = len(S)
    sigma2 = 10.0 ** (-snr_db / 10.0)

    H = (np.random.randn(n_samples, Nr, K) +
         1j * np.random.randn(n_samples, Nr, K)) / np.sqrt(2)

    x_indices = np.random.randint(0, M, size=(n_samples, K))
    x_all = S[x_indices]

    x_weighted = sqrt_P[np.newaxis, :] * x_all  # (n, K)
    y = np.einsum('ijk,ik->ij', H, x_weighted)  # (n, Nr)
    noise = np.sqrt(sigma2 / 2) * (np.random.randn(n_samples, Nr) +
                                    1j * np.random.randn(n_samples, Nr))
    y = y + noise
    return y, H, x_all, x_indices, sigma2


# ============================================================
# Baseline Detectors (GPU-accelerated)
# ============================================================
def detect_mmse(y, H, user_k, S, sigma2, P_arr, batch_size=20000):
    """Linear MMSE detector for user k."""
    sqrt_P = np.sqrt(P_arr)
    n = len(y)
    M = len(S)
    Nr = H.shape[1]

    y_t = torch.tensor(y, dtype=torch.complex64, device=DEVICE)
    H_t = torch.tensor(H, dtype=torch.complex64, device=DEVICE)
    S_t = torch.tensor(S, dtype=torch.complex64, device=DEVICE)
    sqrt_P_t = torch.tensor(sqrt_P, dtype=torch.float32, device=DEVICE)

    results = np.zeros(n, dtype=int)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        yb = y_t[start:end]
        Hb = H_t[start:end]
        nb = end - start

        H_eff = Hb * sqrt_P_t.unsqueeze(0).unsqueeze(0)  # (nb, Nr, K)
        cov = torch.bmm(H_eff, H_eff.conj().transpose(1, 2))
        cov += sigma2 * torch.eye(Nr, device=DEVICE).unsqueeze(0)

        h_k = H_eff[:, :, user_k]  # (nb, Nr)
        w_k = torch.linalg.solve(cov, h_k.unsqueeze(-1)).squeeze(-1)

        z_k = (w_k.conj() * yb).sum(dim=1)
        g_k = (w_k.conj() * h_k).sum(dim=1)

        dists = torch.abs(z_k.unsqueeze(1) - g_k.unsqueeze(1) * S_t.unsqueeze(0)) ** 2
        results[start:end] = dists.argmin(dim=1).cpu().numpy()
    return results


def detect_mmse_sic(y, H, user_k, S, sigma2, P_arr, batch_size=20000):
    """MMSE-SIC: decode from highest power down, cancel, detect target."""
    sqrt_P = np.sqrt(P_arr)
    K_total = len(P_arr)
    n = len(y)
    Nr = H.shape[1]

    y_t = torch.tensor(y, dtype=torch.complex64, device=DEVICE)
    H_t = torch.tensor(H, dtype=torch.complex64, device=DEVICE)
    S_t = torch.tensor(S, dtype=torch.complex64, device=DEVICE)
    sqrt_P_t = torch.tensor(sqrt_P, dtype=torch.float32, device=DEVICE)

    results = np.zeros(n, dtype=int)
    sic_order = list(range(K_total - 1, -1, -1))  # highest power first

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        nb = end - start
        y_res = y_t[start:end].clone()
        Hb = H_t[start:end]

        remaining = list(range(K_total))

        for j in sic_order:
            rem_idx = remaining.copy()
            H_rem = Hb[:, :, rem_idx] * sqrt_P_t[rem_idx].unsqueeze(0).unsqueeze(0)
            cov = torch.bmm(H_rem, H_rem.conj().transpose(1, 2))
            cov += sigma2 * torch.eye(Nr, device=DEVICE).unsqueeze(0)

            j_pos = rem_idx.index(j)
            h_j = H_rem[:, :, j_pos]
            w_j = torch.linalg.solve(cov, h_j.unsqueeze(-1)).squeeze(-1)

            z_j = (w_j.conj() * y_res).sum(dim=1)
            g_j = (w_j.conj() * h_j).sum(dim=1)
            dists = torch.abs(z_j.unsqueeze(1) - g_j.unsqueeze(1) * S_t.unsqueeze(0)) ** 2
            det_idx = dists.argmin(dim=1)

            if j == user_k:
                results[start:end] = det_idx.cpu().numpy()
                break

            x_hat_j = S_t[det_idx]
            y_res = y_res - Hb[:, :, j] * (sqrt_P[j] * x_hat_j.unsqueeze(1))
            remaining.remove(j)

    return results


def detect_oracle_sic(y, H, user_k, x_all, S, sigma2, P_arr, batch_size=20000):
    """Oracle SIC: perfect cancellation of higher-power users, then MMSE."""
    sqrt_P = np.sqrt(P_arr)
    K_total = len(P_arr)
    n = len(y)
    Nr = H.shape[1]

    y_t = torch.tensor(y, dtype=torch.complex64, device=DEVICE)
    H_t = torch.tensor(H, dtype=torch.complex64, device=DEVICE)
    x_t = torch.tensor(x_all, dtype=torch.complex64, device=DEVICE)
    S_t = torch.tensor(S, dtype=torch.complex64, device=DEVICE)

    results = np.zeros(n, dtype=int)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        nb = end - start
        y_res = y_t[start:end].clone()
        Hb = H_t[start:end]

        # Cancel users with higher power than user_k
        for j in range(K_total - 1, user_k, -1):
            y_res = y_res - Hb[:, :, j] * (sqrt_P[j] * x_t[start:end, j:j+1])

        # MMSE for user_k considering remaining lower-power users
        rem = list(range(user_k + 1))  # users 0..user_k
        H_rem = Hb[:, :, rem] * torch.tensor(
            sqrt_P[rem], dtype=torch.float32, device=DEVICE).unsqueeze(0).unsqueeze(0)
        cov = torch.bmm(H_rem, H_rem.conj().transpose(1, 2))
        cov += sigma2 * torch.eye(Nr, device=DEVICE).unsqueeze(0)

        k_pos = rem.index(user_k)
        h_k = H_rem[:, :, k_pos]
        w_k = torch.linalg.solve(cov, h_k.unsqueeze(-1)).squeeze(-1)

        z_k = (w_k.conj() * y_res).sum(dim=1)
        g_k = (w_k.conj() * h_k).sum(dim=1)
        dists = torch.abs(z_k.unsqueeze(1) - g_k.unsqueeze(1) * S_t.unsqueeze(0)) ** 2
        results[start:end] = dists.argmin(dim=1).cpu().numpy()
    return results


def detect_exact_map(y, H, user_k, S, sigma2, P_arr, batch_size=10000):
    """Exact marginal MAP via exhaustive enumeration over |S|^{K-1}."""
    sqrt_P = np.sqrt(P_arr)
    K_total = len(P_arr)
    M = len(S)
    n = len(y)

    other_users = [j for j in range(K_total) if j != user_k]
    combos = np.array(list(product(range(M), repeat=len(other_users))))
    n_combos = len(combos)

    # Precompute interference symbol vectors: (n_combos, K-1) complex
    interf_syms = np.zeros((n_combos, len(other_users)), dtype=complex)
    for i, j in enumerate(other_users):
        interf_syms[:, i] = sqrt_P[j] * S[combos[:, i]]
    interf_t = torch.tensor(interf_syms, dtype=torch.complex64, device=DEVICE)  # (n_combos, K-1)

    y_t = torch.tensor(y, dtype=torch.complex64, device=DEVICE)
    H_t = torch.tensor(H, dtype=torch.complex64, device=DEVICE)
    S_t = torch.tensor(S, dtype=torch.complex64, device=DEVICE)

    x_hat = np.zeros(n, dtype=int)
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

        log_marg = torch.zeros(nb, M, device=DEVICE)
        for s_idx in range(M):
            sig_k = h_k * (sqrt_P[user_k] * S_t[s_idx])  # (nb, Nr)
            # residual: y - sig_k - interf  → (nb, Nr, n_combos)
            res = yb.unsqueeze(2) - sig_k.unsqueeze(2) - interf_vec
            log_lik = -torch.sum(torch.abs(res) ** 2, dim=1) * sigma2_inv  # (nb, n_combos)
            mx, _ = log_lik.max(dim=1, keepdim=True)
            log_marg[:, s_idx] = mx.squeeze(1) + torch.log(
                torch.sum(torch.exp(log_lik - mx), dim=1))

        x_hat[start:end] = log_marg.argmax(dim=1).cpu().numpy()
    return x_hat


# ============================================================
# Learned MIMO-MUD
# ============================================================
class LearnedMIMOMUD(nn.Module):
    """
    Iterative learned detector that unfolds soft parallel interference
    cancellation. Maintains soft beliefs for ALL K users and cancels
    multi-user interference using these beliefs at each iteration.
    """
    def __init__(self, M, K, Nr, d=128, L=4, user_k=0):
        super().__init__()
        self.M, self.K, self.Nr, self.d, self.L = M, K, Nr, d, L
        self.user_k = user_k

        # Per-iteration refinement: takes IC-MF output (2) + current logits (M)
        # + other users' soft beliefs (K-1)*M + P (K) + sigma2 (1)
        refine_dim = 2 + M + (K - 1) * M + K + 1
        self.refine = nn.ModuleList([
            nn.Sequential(
                nn.Linear(refine_dim, d), nn.ReLU(), nn.Dropout(0.05),
                nn.Linear(d, d), nn.ReLU(),
                nn.Linear(d, M),
            ) for _ in range(L)
        ])

        self.register_buffer('S_re', torch.zeros(M))
        self.register_buffer('S_im', torch.zeros(M))

    def set_constellation(self, S_complex):
        self.S_re.copy_(torch.tensor(np.real(S_complex), dtype=torch.float32))
        self.S_im.copy_(torch.tensor(np.imag(S_complex), dtype=torch.float32))

    def forward(self, y_complex, H_complex, H_eff, P_vec, sigma2_val,
                init_logits_all):
        """
        y_complex: (batch, Nr) complex
        H_complex: (batch, Nr, K) complex
        H_eff: (batch, Nr, K) complex — H * sqrt(P)
        P_vec: (batch, K) real
        sigma2_val: (batch,) real
        init_logits_all: list of K tensors, each (batch, M) — initial logits from MMSE
        """
        batch = y_complex.shape[0]
        uk = self.user_k

        # Current beliefs for all users (start from MMSE initialization)
        logits = [l.clone() for l in init_logits_all]

        for l_iter in range(self.L):
            # Compute soft estimates for ALL users
            mus = []  # complex soft estimates
            for k in range(self.K):
                pi_k = torch.softmax(logits[k], dim=1)  # (batch, M)
                mu_re = (pi_k * self.S_re.unsqueeze(0)).sum(dim=1)  # (batch,)
                mu_im = (pi_k * self.S_im.unsqueeze(0)).sum(dim=1)
                mus.append(torch.complex(mu_re, mu_im))

            # For user_k: cancel all other users' interference
            y_ic = y_complex.clone()
            for j in range(self.K):
                if j == uk:
                    continue
                y_ic = y_ic - H_eff[:, :, j] * mus[j].unsqueeze(1)

            # Matched filter for user_k on cancelled signal
            h_k = H_eff[:, :, uk]  # (batch, Nr)
            mf = (h_k.conj() * y_ic).sum(dim=1)  # (batch,)
            h_norm = (h_k.conj() * h_k).sum(dim=1).real + 1e-10
            mf_norm = mf / h_norm  # (batch,) — estimate of x_k

            # Gather other users' current beliefs
            other_beliefs = []
            for j in range(self.K):
                if j == uk:
                    continue
                other_beliefs.append(torch.softmax(logits[j], dim=1))
            other_flat = torch.cat(other_beliefs, dim=1)  # (batch, (K-1)*M)

            # Refinement input
            ref_in = torch.cat([
                mf_norm.real.unsqueeze(1), mf_norm.imag.unsqueeze(1),
                logits[uk],
                other_flat,
                P_vec, sigma2_val.unsqueeze(1),
            ], dim=1)

            logits[uk] = logits[uk] + self.refine[l_iter](ref_in)

            # Also update other users using the same IC approach
            for j in range(self.K):
                if j == uk:
                    continue
                y_ic_j = y_complex.clone()
                # Cancel all users except j using current soft estimates
                for j2 in range(self.K):
                    if j2 == j:
                        continue
                    mu_j2 = mus[j2] if j2 != uk else None
                    if j2 == uk:
                        # Use updated user_k beliefs
                        pi_uk = torch.softmax(logits[uk], dim=1)
                        mu_uk_re = (pi_uk * self.S_re.unsqueeze(0)).sum(dim=1)
                        mu_uk_im = (pi_uk * self.S_im.unsqueeze(0)).sum(dim=1)
                        mu_j2 = torch.complex(mu_uk_re, mu_uk_im)
                    y_ic_j = y_ic_j - H_eff[:, :, j2] * mu_j2.unsqueeze(1)

                h_j = H_eff[:, :, j]
                mf_j = (h_j.conj() * y_ic_j).sum(dim=1)
                h_j_norm = (h_j.conj() * h_j).sum(dim=1).real + 1e-10
                mf_j_norm = mf_j / h_j_norm
                # Simple update: use MF distance as new logits
                dists = torch.abs(mf_j_norm.unsqueeze(1) -
                                  torch.complex(self.S_re, self.S_im).unsqueeze(0)) ** 2
                logits[j] = -dists / (sigma2_val.unsqueeze(1) / h_j_norm.unsqueeze(1) + 1e-10)

        return logits[uk]


def compute_mmse_init_logits(y_t, H_eff, sigma2, S_t, K_, Nr, device, bs=20000):
    """Compute initial per-user logits from MMSE for all K users."""
    n = y_t.shape[0]
    M = S_t.shape[0]
    all_logits = [[] for _ in range(K_)]

    for s in range(0, n, bs):
        e = min(s + bs, n)
        Hb = H_eff[s:e]
        yb = y_t[s:e]
        nb_ = e - s
        cov = torch.bmm(Hb, Hb.conj().transpose(1, 2))
        cov += sigma2 * torch.eye(Nr, device=device).unsqueeze(0)

        for k in range(K_):
            hk = Hb[:, :, k]
            wk = torch.linalg.solve(cov, hk.unsqueeze(-1)).squeeze(-1)
            zk = (wk.conj() * yb).sum(dim=1)
            gk = (wk.conj() * hk).sum(dim=1)
            est = zk / (gk + 1e-10)
            # Convert to logits: negative distance to constellation points
            dists = torch.abs(est.unsqueeze(1) - S_t.unsqueeze(0)) ** 2
            h_norm = gk.real + 1e-10
            logits_k = -dists / (sigma2 / h_norm.unsqueeze(1) + 1e-10)
            all_logits[k].append(logits_k.float())

    return [torch.cat(ll) for ll in all_logits]


# ============================================================
# Training
# ============================================================
def train_mimo_model(mod_type, Nr, user_k=0, P_arr=None, d=128, L=4,
                     n_train=400000, epochs=40, batch_size=1024, lr=1e-3):
    if P_arr is None:
        P_arr = P_DEFAULT
    S = get_constellation(mod_type)
    M = len(S)
    sqrt_P = np.sqrt(P_arr)

    snr_range = np.arange(0, 31, 2) if mod_type != '16QAM' else np.arange(5, 36, 2)

    print(f"\n  Training MIMO {mod_type} | Nr={Nr} | K={K} | L={L} | d={d}")
    n_per = n_train // len(snr_range)

    # Generate training data
    y_all, H_all, x_idx_all, s2_all = [], [], [], []
    for snr_db in snr_range:
        y, H, x_a, x_idx, s2 = generate_mimo_data(n_per, snr_db, mod_type, Nr, K, P_arr)
        y_all.append(y); H_all.append(H)
        x_idx_all.append(x_idx[:, user_k]); s2_all.append(np.full(n_per, s2))

    y_tr = np.concatenate(y_all)
    H_tr = np.concatenate(H_all)
    x_idx_tr = np.concatenate(x_idx_all)
    s2_tr = np.concatenate(s2_all)

    perm = np.random.permutation(len(y_tr))
    y_tr, H_tr, x_idx_tr, s2_tr = y_tr[perm], H_tr[perm], x_idx_tr[perm], s2_tr[perm]

    # Store as complex tensors on CPU for DataLoader
    y_tr_t = torch.tensor(y_tr, dtype=torch.complex64)
    H_tr_t = torch.tensor(H_tr, dtype=torch.complex64)
    s2_tr_t = torch.tensor(s2_tr, dtype=torch.float32)
    labels_tr = torch.tensor(x_idx_tr, dtype=torch.long)
    P_vec_tr = torch.tensor(P_arr, dtype=torch.float32).unsqueeze(0).expand(len(y_tr), -1).clone()

    # Validation
    n_val = 40000
    n_vp = n_val // len(snr_range)
    y_v, H_v, xi_v, s2_v = [], [], [], []
    for snr_db in snr_range:
        y, H, xa, xi, s2 = generate_mimo_data(n_vp, snr_db, mod_type, Nr, K, P_arr)
        y_v.append(y); H_v.append(H); xi_v.append(xi[:, user_k]); s2_v.append(np.full(n_vp, s2))
    y_val_t = torch.tensor(np.concatenate(y_v), dtype=torch.complex64)
    H_val_t = torch.tensor(np.concatenate(H_v), dtype=torch.complex64)
    s2_val_t = torch.tensor(np.concatenate(s2_v), dtype=torch.float32)
    labels_val = torch.tensor(np.concatenate(xi_v), dtype=torch.long)
    P_vec_val = torch.tensor(P_arr, dtype=torch.float32).unsqueeze(0).expand(len(y_val_t), -1).clone()

    train_ds = TensorDataset(y_tr_t, H_tr_t, P_vec_tr, s2_tr_t, labels_tr)
    val_ds = TensorDataset(y_val_t, H_val_t, P_vec_val, s2_val_t, labels_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, pin_memory=True)

    S_t = torch.tensor(S, dtype=torch.complex64, device=DEVICE)
    sqrt_P_t = torch.tensor(sqrt_P, dtype=torch.float32, device=DEVICE)

    model = LearnedMIMOMUD(M, K, Nr, d=d, L=L, user_k=user_k).to(DEVICE)
    model.set_constellation(S)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    patience = 8

    for epoch in range(epochs):
        model.train()
        total_loss, nb_ = 0.0, 0
        for batch in train_loader:
            y_b, H_b, P_b, s2_b, lab_b = [t.to(DEVICE) for t in batch]
            H_eff_b = H_b * sqrt_P_t.unsqueeze(0).unsqueeze(0)

            # Compute MMSE init logits on-the-fly
            nb2 = y_b.shape[0]
            cov = torch.bmm(H_eff_b, H_eff_b.conj().transpose(1, 2))
            cov += s2_b.unsqueeze(1).unsqueeze(2) * torch.eye(Nr, device=DEVICE).unsqueeze(0)
            init_logits = []
            for k in range(K):
                hk = H_eff_b[:, :, k]
                wk = torch.linalg.solve(cov, hk.unsqueeze(-1)).squeeze(-1)
                zk = (wk.conj() * y_b).sum(dim=1)
                gk = (wk.conj() * hk).sum(dim=1)
                est = zk / (gk + 1e-10)
                dists = torch.abs(est.unsqueeze(1) - S_t.unsqueeze(0)) ** 2
                hn = gk.real + 1e-10
                init_logits.append((-dists / (s2_b.unsqueeze(1) / hn.unsqueeze(1) + 1e-10)).float())

            logits = model(y_b, H_b, H_eff_b, P_b, s2_b, init_logits)
            loss = criterion(logits, lab_b)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            nb_ += 1
        scheduler.step()

        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            model.eval()
            correct, total = 0, 0
            vl = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    y_b, H_b, P_b, s2_b, lab_b = [t.to(DEVICE) for t in batch]
                    H_eff_b = H_b * sqrt_P_t.unsqueeze(0).unsqueeze(0)
                    nb2 = y_b.shape[0]
                    cov = torch.bmm(H_eff_b, H_eff_b.conj().transpose(1, 2))
                    cov += s2_b.unsqueeze(1).unsqueeze(2) * torch.eye(Nr, device=DEVICE).unsqueeze(0)
                    init_logits = []
                    for k in range(K):
                        hk = H_eff_b[:, :, k]
                        wk = torch.linalg.solve(cov, hk.unsqueeze(-1)).squeeze(-1)
                        zk = (wk.conj() * y_b).sum(dim=1)
                        gk = (wk.conj() * hk).sum(dim=1)
                        est = zk / (gk + 1e-10)
                        dists = torch.abs(est.unsqueeze(1) - S_t.unsqueeze(0)) ** 2
                        hn = gk.real + 1e-10
                        init_logits.append((-dists / (s2_b.unsqueeze(1) / hn.unsqueeze(1) + 1e-10)).float())
                    logits = model(y_b, H_b, H_eff_b, P_b, s2_b, init_logits)
                    vl += criterion(logits, lab_b).item()
                    correct += (logits.argmax(1) == lab_b).sum().item()
                    total += lab_b.size(0)
            vla = vl / len(val_loader)
            print(f"    Epoch {epoch+1:3d}/{epochs} | TrLoss: {total_loss/nb_:.4f} | "
                  f"VlLoss: {vla:.4f} | VlAcc: {correct/total:.4f}")
            if vla < best_val_loss:
                best_val_loss = vla
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"    Early stopping at epoch {epoch+1}")
                    break

    if best_state:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    return model


def evaluate_mimo_model(model, y, H, P_arr, sigma2, user_k, S, batch_size=2048):
    """Evaluate trained MIMO model."""
    model.eval()
    n = len(y)
    Nr = y.shape[1]
    sqrt_P = np.sqrt(P_arr)
    sqrt_P_t = torch.tensor(sqrt_P, dtype=torch.float32, device=DEVICE)
    S_t = torch.tensor(S, dtype=torch.complex64, device=DEVICE)

    y_t = torch.tensor(y, dtype=torch.complex64, device=DEVICE)
    H_t = torch.tensor(H, dtype=torch.complex64, device=DEVICE)
    P_vec = torch.tensor(P_arr, dtype=torch.float32, device=DEVICE).unsqueeze(0).expand(n, -1)
    s2t = torch.full((n,), sigma2, dtype=torch.float32, device=DEVICE)

    preds = []
    with torch.no_grad():
        for s in range(0, n, batch_size):
            e = min(s + batch_size, n)
            yb = y_t[s:e]
            Hb = H_t[s:e]
            nb = e - s
            H_eff_b = Hb * sqrt_P_t.unsqueeze(0).unsqueeze(0)

            cov = torch.bmm(H_eff_b, H_eff_b.conj().transpose(1, 2))
            cov += sigma2 * torch.eye(Nr, device=DEVICE).unsqueeze(0)
            init_logits = []
            for k in range(K):
                hk = H_eff_b[:, :, k]
                wk = torch.linalg.solve(cov, hk.unsqueeze(-1)).squeeze(-1)
                zk = (wk.conj() * yb).sum(dim=1)
                gk = (wk.conj() * hk).sum(dim=1)
                est = zk / (gk + 1e-10)
                dists = torch.abs(est.unsqueeze(1) - S_t.unsqueeze(0)) ** 2
                hn = gk.real + 1e-10
                init_logits.append((-dists / (sigma2 / hn.unsqueeze(1) + 1e-10)).float())

            logits = model(yb, Hb, H_eff_b, P_vec[s:e], s2t[s:e], init_logits)
            preds.append(logits.argmax(1).cpu().numpy())
    return np.concatenate(preds)


# ============================================================
# Main Simulation
# ============================================================
def run_mimo_simulation(mod_type, Nr, user_k=0, n_test=200000, model=None):
    """Run full SER vs SNR simulation for MIMO-NOMA."""
    S = get_constellation(mod_type)
    P_arr = P_DEFAULT
    snr_range = np.arange(0, 31, 3) if mod_type != '16QAM' else np.arange(5, 36, 3)

    metric = 'BER' if mod_type == 'BPSK' else 'SER'
    names = ['MMSE', 'MMSE-SIC', 'Oracle SIC', 'Exact MAP', 'Proposed']
    results = {n: [] for n in names}

    print(f"\n{'='*60}")
    print(f"  MIMO-NOMA: {mod_type} | Nr={Nr} | K={K} | User {user_k+1}")
    print(f"{'='*60}")

    for snr_db in snr_range:
        np.random.seed(SEED + int(snr_db * 100))
        y, H, x_all, x_idx, sigma2 = generate_mimo_data(n_test, snr_db, mod_type, Nr, K, P_arr)
        x_k_idx = x_idx[:, user_k]

        p_mmse = detect_mmse(y, H, user_k, S, sigma2, P_arr)
        ser_mmse = np.mean(p_mmse != x_k_idx)
        results['MMSE'].append(ser_mmse)

        p_sic = detect_mmse_sic(y, H, user_k, S, sigma2, P_arr)
        ser_sic = np.mean(p_sic != x_k_idx)
        results['MMSE-SIC'].append(ser_sic)

        p_orc = detect_oracle_sic(y, H, user_k, x_all, S, sigma2, P_arr)
        ser_orc = np.mean(p_orc != x_k_idx)
        results['Oracle SIC'].append(ser_orc)

        p_map = detect_exact_map(y, H, user_k, S, sigma2, P_arr)
        ser_map = np.mean(p_map != x_k_idx)
        results['Exact MAP'].append(ser_map)

        if model is not None:
            p_prop = evaluate_mimo_model(model, y, H, P_arr, sigma2, user_k, S)
            ser_prop = np.mean(p_prop != x_k_idx)
            results['Proposed'].append(ser_prop)

        prop_str = f" | Prop={results['Proposed'][-1]:.2e}" if model else ""
        print(f"  SNR={snr_db:2d} | MMSE={ser_mmse:.2e} | SIC={ser_sic:.2e} | "
              f"Orc={ser_orc:.2e} | MAP={ser_map:.2e}{prop_str}")

    if model is None:
        del results['Proposed']
    return snr_range, results


# ============================================================
# Plotting
# ============================================================
def setup_ieee():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'font.size': 9, 'axes.labelsize': 10, 'legend.fontsize': 7,
        'xtick.labelsize': 8, 'ytick.labelsize': 8,
        'lines.linewidth': 1.2, 'lines.markersize': 5,
        'figure.figsize': (3.5, 2.8), 'figure.dpi': 300,
        'savefig.dpi': 300, 'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02, 'text.usetex': False,
        'axes.grid': True, 'grid.alpha': 0.3,
    })

MARKERS = {
    'MMSE': ('v', 'gray'),
    'MMSE-SIC': ('s', '#d62728'),
    'Oracle SIC': ('^', '#2ca02c'),
    'Exact MAP': ('D', '#1f77b4'),
    'Proposed': ('o', '#ff7f0e'),
}
LINESTYLES = {
    'MMSE': ':', 'MMSE-SIC': '--', 'Oracle SIC': '-.',
    'Exact MAP': '-', 'Proposed': '-',
}


def plot_ser(snr_range, results, mod_type, Nr, filename, user_k=0):
    setup_ieee()
    fig, ax = plt.subplots()
    metric = 'BER' if mod_type == 'BPSK' else 'SER'
    order = ['MMSE', 'MMSE-SIC', 'Oracle SIC', 'Exact MAP', 'Proposed']

    for name in order:
        if name not in results:
            continue
        vals = np.maximum(np.array(results[name]), 1e-6)
        m, c = MARKERS[name]
        ls = LINESTYLES[name]
        lw = 2.0 if name == 'Proposed' else 1.2
        ms = 6 if name == 'Proposed' else 5
        mfc = c if name == 'Proposed' else 'none'
        ax.semilogy(snr_range, vals, marker=m, color=c, linestyle=ls,
                    linewidth=lw, label=name, markersize=ms, markerfacecolor=mfc)

    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel(metric)
    mod_d = mod_type if mod_type != '16QAM' else '16-QAM'
    ax.set_title(f'MIMO-NOMA: {mod_d}, $N_r={Nr}$, $K={K}$, User {user_k+1}')
    ax.legend(loc='lower left', framealpha=0.9)
    ax.set_ylim(bottom=5e-6)

    fig.savefig(filename, format='eps')
    fig.savefig(filename.replace('.eps', '.png'), format='png', dpi=300)
    plt.close(fig)
    print(f"  Saved: {filename}")


def plot_nr_comparison(results_by_nr, mod_type, filename, user_k=0):
    """Plot SER vs SNR for multiple Nr values, comparing Proposed vs MMSE-SIC."""
    setup_ieee()
    fig, ax = plt.subplots()
    metric = 'BER' if mod_type == 'BPSK' else 'SER'
    colors_nr = {2: '#d62728', 4: '#1f77b4', 8: '#2ca02c'}

    for Nr, (snr_range, results) in sorted(results_by_nr.items()):
        c = colors_nr.get(Nr, 'black')
        vals_sic = np.maximum(np.array(results['MMSE-SIC']), 1e-6)
        ax.semilogy(snr_range, vals_sic, '--', color=c, marker='s', markersize=4,
                    markerfacecolor='none', linewidth=1.0,
                    label=f'MMSE-SIC, $N_r={Nr}$')
        if 'Proposed' in results:
            vals_prop = np.maximum(np.array(results['Proposed']), 1e-6)
            ax.semilogy(snr_range, vals_prop, '-', color=c, marker='o', markersize=5,
                        linewidth=1.5, label=f'Proposed, $N_r={Nr}$')
        vals_map = np.maximum(np.array(results['Exact MAP']), 1e-6)
        ax.semilogy(snr_range, vals_map, ':', color=c, marker='D', markersize=4,
                    markerfacecolor='none', linewidth=1.0,
                    label=f'MAP, $N_r={Nr}$')

    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel(metric)
    mod_d = mod_type if mod_type != '16QAM' else '16-QAM'
    ax.set_title(f'MIMO-NOMA: {mod_d}, $K={K}$, User {user_k+1}')
    ax.legend(loc='lower left', framealpha=0.9, fontsize=6, ncol=2)
    ax.set_ylim(bottom=5e-6)

    fig.savefig(filename, format='eps')
    fig.savefig(filename.replace('.eps', '.png'), format='png', dpi=300)
    plt.close(fig)
    print(f"  Saved: {filename}")


# ============================================================
# Main
# ============================================================
def main():
    t0 = time.time()

    # --- QPSK, Nr=4 (main result) ---
    Nr = 4
    mod = 'QPSK'
    model_4 = train_mimo_model(mod, Nr, user_k=0, d=128, L=4,
                               n_train=400000, epochs=40)
    snr4, res4 = run_mimo_simulation(mod, Nr, user_k=0, n_test=200000, model=model_4)
    plot_ser(snr4, res4, mod, Nr, os.path.join(OUT_DIR, 'fig_mimo_qpsk_nr4.eps'))
    torch.cuda.empty_cache()

    # --- QPSK, Nr=2 ---
    Nr = 2
    model_2 = train_mimo_model(mod, Nr, user_k=0, d=128, L=4,
                               n_train=400000, epochs=40)
    snr2, res2 = run_mimo_simulation(mod, Nr, user_k=0, n_test=200000, model=model_2)
    plot_ser(snr2, res2, mod, Nr, os.path.join(OUT_DIR, 'fig_mimo_qpsk_nr2.eps'))
    torch.cuda.empty_cache()

    # --- Nr comparison plot ---
    plot_nr_comparison({2: (snr2, res2), 4: (snr4, res4)}, mod,
                       os.path.join(OUT_DIR, 'fig_mimo_nr_comparison.eps'))

    # --- BPSK, Nr=4 ---
    mod_b = 'BPSK'
    model_b4 = train_mimo_model(mod_b, 4, user_k=0, d=128, L=4,
                                n_train=400000, epochs=30)
    snr_b4, res_b4 = run_mimo_simulation(mod_b, 4, user_k=0, n_test=200000, model=model_b4)
    plot_ser(snr_b4, res_b4, mod_b, 4, os.path.join(OUT_DIR, 'fig_mimo_bpsk_nr4.eps'))
    torch.cuda.empty_cache()

    # --- 16-QAM, Nr=4 ---
    mod_q = '16QAM'
    model_q4 = train_mimo_model(mod_q, 4, user_k=0, d=128, L=4,
                                n_train=400000, epochs=50)
    snr_q4, res_q4 = run_mimo_simulation(mod_q, 4, user_k=0, n_test=200000, model=model_q4)
    plot_ser(snr_q4, res_q4, mod_q, 4, os.path.join(OUT_DIR, 'fig_mimo_16qam_nr4.eps'))
    torch.cuda.empty_cache()

    # --- Print summary ---
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    for label, (sr, res) in [('QPSK Nr=4', (snr4, res4)),
                              ('QPSK Nr=2', (snr2, res2)),
                              ('BPSK Nr=4', (snr_b4, res_b4)),
                              ('16QAM Nr=4', (snr_q4, res_q4))]:
        # Find a reference SNR where SIC is around 1e-2
        sic_arr = np.array(res['MMSE-SIC'])
        ref_idx = np.argmin(np.abs(sic_arr - 1e-2))
        if sic_arr[ref_idx] > 0.5:
            ref_idx = len(sr) - 1
        snr_ref = sr[ref_idx]
        print(f"\n  {label} at SNR={snr_ref} dB:")
        for name in ['MMSE', 'MMSE-SIC', 'Oracle SIC', 'Exact MAP', 'Proposed']:
            if name in res:
                print(f"    {name:15s}: {res[name][ref_idx]:.2e}")

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed/60:.1f} min")
    print(f"  Figures saved to: {OUT_DIR}")


if __name__ == '__main__':
    main()
