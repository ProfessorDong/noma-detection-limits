#!/usr/bin/env python3
"""
Comprehensive simulation for:
  "Learned Multiuser Detection for Power-Domain NOMA
   Beyond Successive Interference Cancellation"

Generates all figures for the Numerical Results section.
Runs on GPU (CUDA) when available.

Author: Simulation code for Dong & Entezami
"""

import sys
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)  # line-buffered

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from itertools import product
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, LogFormatterMathtext
import time
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Global Configuration
# ============================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

# System parameters
K = 3
P_DEFAULT = np.array([0.2, 0.3, 0.5])
SQRT_P_DEFAULT = np.sqrt(P_DEFAULT)

# Output directory
OUT_DIR = '/home/dong/Workspace/WritePaper/noma_sic'


# ============================================================
# Constellation Definitions
# ============================================================
def get_constellation(mod_type):
    """Return normalized constellation points (unit average energy)."""
    if mod_type == 'BPSK':
        return np.array([1.0 + 0j, -1.0 + 0j])
    elif mod_type == 'QPSK':
        pts = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
        return pts
    elif mod_type == '16QAM':
        vals = [-3, -1, 1, 3]
        pts = np.array([a + 1j*b for a in vals for b in vals])
        pts = pts / np.sqrt(np.mean(np.abs(pts)**2))
        return pts
    else:
        raise ValueError(f"Unknown modulation: {mod_type}")


# ============================================================
# Data Generation (NumPy, for baselines)
# ============================================================
def generate_noma_data(n_samples, snr_db, mod_type, user_k=0,
                       P_arr=None, csi_error_var=0.0):
    """
    Generate downlink NOMA data with ordered Rayleigh fading channels.

    Returns: y_k, h_k_est, h_k_true, x_k, x_all, x_k_indices, sigma2
    """
    if P_arr is None:
        P_arr = P_DEFAULT
    sqrt_P = np.sqrt(P_arr)
    S = get_constellation(mod_type)
    M = len(S)

    # Noise variance: SNR = P_total / sigma^2 = 1 / sigma^2
    sigma2 = 10.0 ** (-snr_db / 10.0)

    # Generate K independent Rayleigh fading channels, sort by |h| descending
    h_all = (np.random.randn(n_samples, K) +
             1j * np.random.randn(n_samples, K)) / np.sqrt(2)
    order = np.argsort(-np.abs(h_all), axis=1)
    h_all = np.take_along_axis(h_all, order, axis=1)

    # Generate symbols for all users
    x_indices = np.random.randint(0, M, size=(n_samples, K))
    x_all = S[x_indices]  # (n_samples, K)

    # Superposed signal
    s_super = np.sum(sqrt_P[np.newaxis, :] * x_all, axis=1)  # (n_samples,)

    # Received signal at user k
    h_k = h_all[:, user_k]
    noise = np.sqrt(sigma2 / 2) * (np.random.randn(n_samples) +
                                    1j * np.random.randn(n_samples))
    y_k = h_k * s_super + noise

    # CSI estimation error
    if csi_error_var > 0:
        e_k = np.sqrt(csi_error_var / 2) * (np.random.randn(n_samples) +
                                              1j * np.random.randn(n_samples))
        h_k_est = h_k + e_k
    else:
        h_k_est = h_k.copy()

    x_k = x_all[:, user_k]
    x_k_idx = x_indices[:, user_k]

    return y_k, h_k_est, h_k, x_k, x_all, x_k_idx, sigma2


# ============================================================
# Baseline Detectors
# ============================================================
def detect_single_user(y_k, h_k, user_k, S, sigma2, P_arr):
    """Treats all interference as noise."""
    sqrt_Pk = np.sqrt(P_arr[user_k])
    distances = np.abs(y_k[:, None] - h_k[:, None] * sqrt_Pk * S[None, :]) ** 2
    return np.argmin(distances, axis=1)


def detect_conv_sic(y_k, h_k, user_k, S, sigma2, P_arr):
    """Conventional SIC with sequential hard decisions."""
    sqrt_P = np.sqrt(P_arr)
    K_total = len(P_arr)
    y_res = y_k.copy()

    # Cancel from highest power (user K-1) down to user_k+1
    for j in range(K_total - 1, user_k, -1):
        dists = np.abs(y_res[:, None] - h_k[:, None] * sqrt_P[j] * S[None, :]) ** 2
        x_hat_j = S[np.argmin(dists, axis=1)]
        y_res = y_res - h_k * sqrt_P[j] * x_hat_j

    # Detect user k
    dists = np.abs(y_res[:, None] - h_k[:, None] * sqrt_P[user_k] * S[None, :]) ** 2
    return np.argmin(dists, axis=1)


def detect_oracle_sic(y_k, h_k, user_k, x_all, S, sigma2, P_arr):
    """Oracle SIC with genie-aided perfect cancellation."""
    sqrt_P = np.sqrt(P_arr)
    K_total = len(P_arr)
    y_res = y_k.copy()

    for j in range(K_total - 1, user_k, -1):
        y_res = y_res - h_k * sqrt_P[j] * x_all[:, j]

    dists = np.abs(y_res[:, None] - h_k[:, None] * sqrt_P[user_k] * S[None, :]) ** 2
    return np.argmin(dists, axis=1)


def detect_exact_map(y_k, h_k, user_k, S, sigma2, P_arr, batch_size=20000):
    """
    Exact marginal MAP: enumerate all |S|^(K-1) combinations.
    Uses GPU for acceleration. Processes in batches for memory.
    """
    sqrt_P = np.sqrt(P_arr)
    K_total = len(P_arr)
    M = len(S)
    n = len(y_k)

    other_users = [j for j in range(K_total) if j != user_k]
    combos = np.array(list(product(range(M), repeat=len(other_users))))
    n_combos = len(combos)

    interference = np.zeros(n_combos, dtype=complex)
    for i, j in enumerate(other_users):
        interference += sqrt_P[j] * S[combos[:, i]]

    # Pre-compute all composite signals for each candidate s (M x n_combos)
    composites = np.zeros((M, n_combos), dtype=complex)
    for s_idx in range(M):
        composites[s_idx] = sqrt_P[user_k] * S[s_idx] + interference

    # Move to GPU
    composites_t = torch.tensor(composites, dtype=torch.complex64, device=DEVICE)
    y_t = torch.tensor(y_k, dtype=torch.complex64, device=DEVICE)
    h_t = torch.tensor(h_k, dtype=torch.complex64, device=DEVICE)
    sigma2_inv = 1.0 / sigma2

    x_hat_idx = np.zeros(n, dtype=int)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        y_b = y_t[start:end]       # (nb,)
        h_b = h_t[start:end]       # (nb,)
        nb = end - start

        log_marginal = torch.zeros(nb, M, device=DEVICE)

        for s_idx in range(M):
            diff = y_b[:, None] - h_b[:, None] * composites_t[s_idx:s_idx+1, :]
            log_lik = -torch.abs(diff).pow(2) * sigma2_inv
            max_ll, _ = log_lik.max(dim=1, keepdim=True)
            log_marginal[:, s_idx] = max_ll.squeeze(1) + torch.log(
                torch.sum(torch.exp(log_lik - max_ll), dim=1))

        x_hat_idx[start:end] = log_marginal.argmax(dim=1).cpu().numpy()

    return x_hat_idx


# ============================================================
# Learned MUD Architecture (PyTorch, GPU)
# ============================================================
class LearnedMUD(nn.Module):
    """
    Unrolled power-aware message-passing detector.
    Implements the architecture from Section IV of the paper.
    """
    def __init__(self, M, K, d=128, L=4, user_k=0, use_attention=True):
        super().__init__()
        self.M = M
        self.K = K
        self.d = d
        self.L = L
        self.user_k = user_k
        self.use_attention = use_attention

        # Input: 4 (signal) + M (distances) + 2 (channel) + K (powers) + 1 (noise)
        input_dim = 4 + M + 2 + K + 1

        # Input embedding phi_in
        self.phi_in = nn.Sequential(
            nn.Linear(input_dim, d),
            nn.ReLU(),
            nn.Linear(d, d),
        )

        # Output projection phi_out (shared)
        self.phi_out = nn.Linear(d, M)

        # Iterative update modules psi^(l)
        # Input: z (d) + MF projection (2) + power embedding (K-1)
        update_dim = d + 2 + (K - 1)
        self.psi = nn.ModuleList([
            nn.Sequential(
                nn.Linear(update_dim, d),
                nn.ReLU(),
                nn.Linear(d, d),
            ) for _ in range(L)
        ])

        # Attention modules (power-aware interference attention)
        if use_attention:
            self.attn_query = nn.ModuleList([nn.Linear(d, d) for _ in range(L)])
            self.attn_key = nn.ModuleList([nn.Linear(1, d) for _ in range(L)])
            self.attn_value = nn.ModuleList([nn.Linear(1, d) for _ in range(L)])
            self.attn_out = nn.ModuleList([nn.Linear(d, d) for _ in range(L)])

        # Constellation points (set later)
        self.register_buffer('S_re', torch.zeros(M))
        self.register_buffer('S_im', torch.zeros(M))

    def set_constellation(self, S_complex):
        self.S_re = torch.tensor(np.real(S_complex), dtype=torch.float32,
                                 device=self.S_re.device)
        self.S_im = torch.tensor(np.imag(S_complex), dtype=torch.float32,
                                 device=self.S_im.device)

    def forward(self, y_re, y_im, y_abs, y_ang,
                dist_features, h_abs, h_ang, h_re, h_im,
                P_vec, sigma2_val, sqrt_Pk_val):
        batch = y_re.shape[0]

        # Build input features
        inp = torch.cat([
            y_re.unsqueeze(1), y_im.unsqueeze(1),
            y_abs.unsqueeze(1), y_ang.unsqueeze(1),
            dist_features,
            h_abs.unsqueeze(1), h_ang.unsqueeze(1),
            P_vec,
            sigma2_val.unsqueeze(1),
        ], dim=1)

        z = self.phi_in(inp)

        # Power embedding: sqrt(P_j) for j != user_k
        other_idx = [j for j in range(self.K) if j != self.user_k]
        rho = torch.sqrt(torch.clamp(P_vec[:, other_idx], min=1e-8))

        for l in range(self.L):
            # Soft estimate
            logits = self.phi_out(z)
            pi = torch.softmax(logits, dim=1)

            # mu_k = sum_s s * pi(s) (complex soft estimate)
            mu_re = (pi * self.S_re.unsqueeze(0)).sum(dim=1)
            mu_im = (pi * self.S_im.unsqueeze(0)).sum(dim=1)

            # Residual: r = y - h * sqrt(Pk) * mu
            hmu_re = sqrt_Pk_val * (h_re * mu_re - h_im * mu_im)
            hmu_im = sqrt_Pk_val * (h_re * mu_im + h_im * mu_re)
            r_re = y_re - hmu_re
            r_im = y_im - hmu_im

            # Matched filter: h* r
            mf_re = h_re * r_re + h_im * r_im
            mf_im = h_re * r_im - h_im * r_re

            # MLP update with residual connection
            psi_in = torch.cat([z, mf_re.unsqueeze(1), mf_im.unsqueeze(1), rho], dim=1)
            z = z + self.psi[l](psi_in)

            # Attention update
            if self.use_attention:
                K_minus_1 = rho.shape[1]
                q = self.attn_query[l](z)  # (batch, d)
                # Create per-interferer keys and values
                k_all = []
                v_all = []
                for ii in range(K_minus_1):
                    k_ii = self.attn_key[l](rho[:, ii:ii+1])  # (batch, d)
                    v_ii = self.attn_value[l](rho[:, ii:ii+1])
                    k_all.append(k_ii)
                    v_all.append(v_ii)
                keys = torch.stack(k_all, dim=1)    # (batch, K-1, d)
                values = torch.stack(v_all, dim=1)  # (batch, K-1, d)

                # Attention weights
                attn_w = torch.bmm(q.unsqueeze(1), keys.transpose(1, 2))  # (batch,1,K-1)
                attn_w = attn_w / (self.d ** 0.5)
                attn_w = torch.softmax(attn_w, dim=2)

                attn_out = torch.bmm(attn_w, values).squeeze(1)  # (batch, d)
                z = z + self.attn_out[l](attn_out)

        return self.phi_out(z)


def prepare_features(y_k, h_k, P_arr, sigma2, user_k, S, device):
    """Convert numpy data to GPU tensors with proper features."""
    n = len(y_k)
    sqrt_Pk = np.sqrt(P_arr[user_k])

    y_re = torch.tensor(np.real(y_k), dtype=torch.float32, device=device)
    y_im = torch.tensor(np.imag(y_k), dtype=torch.float32, device=device)
    y_abs = torch.sqrt(y_re**2 + y_im**2)
    y_ang = torch.atan2(y_im, y_re)

    h_re = torch.tensor(np.real(h_k), dtype=torch.float32, device=device)
    h_im = torch.tensor(np.imag(h_k), dtype=torch.float32, device=device)
    h_abs = torch.sqrt(h_re**2 + h_im**2)
    h_ang = torch.atan2(h_im, h_re)

    # Constellation distances
    S_t = torch.tensor(S, dtype=torch.complex64, device=device)
    y_t = torch.tensor(y_k, dtype=torch.complex64, device=device)
    h_t = torch.tensor(h_k, dtype=torch.complex64, device=device)
    dists = torch.abs(y_t[:, None] - h_t[:, None] * sqrt_Pk * S_t[None, :])
    dist_features = dists.float()

    P_vec = torch.tensor(P_arr, dtype=torch.float32, device=device).unsqueeze(0).expand(n, -1)
    sigma2_t = torch.full((n,), sigma2, dtype=torch.float32, device=device)

    return (y_re, y_im, y_abs, y_ang, dist_features,
            h_abs, h_ang, h_re, h_im, P_vec, sigma2_t, sqrt_Pk)


# ============================================================
# Training
# ============================================================
def train_model(mod_type, user_k=0, P_arr=None, d=128, L=4,
                use_attention=True, snr_range_db=None,
                n_train=200000, n_val=40000, epochs=80,
                batch_size=1024, lr=1e-3, csi_error_var=0.0):
    """
    Train the learned MUD across a range of SNR values.
    """
    if P_arr is None:
        P_arr = P_DEFAULT
    S = get_constellation(mod_type)
    M = len(S)

    if snr_range_db is None:
        if mod_type == 'BPSK':
            snr_range_db = np.arange(0, 31, 2)
        elif mod_type == 'QPSK':
            snr_range_db = np.arange(0, 31, 2)
        else:
            snr_range_db = np.arange(5, 36, 2)

    print(f"\n  Training {mod_type} | L={L} | attn={use_attention} | "
          f"CSI_err={csi_error_var:.3f} | P={P_arr}")

    # Generate training data across SNR range
    y_all, h_all, x_idx_all, sigma2_all = [], [], [], []
    n_per_snr = n_train // len(snr_range_db)

    for snr_db in snr_range_db:
        y, h_est, h_true, x_k, x_all_sym, x_k_idx, s2 = generate_noma_data(
            n_per_snr, snr_db, mod_type, user_k, P_arr, csi_error_var)
        y_all.append(y)
        h_all.append(h_est)
        x_idx_all.append(x_k_idx)
        sigma2_all.append(np.full(n_per_snr, s2))

    y_train = np.concatenate(y_all)
    h_train = np.concatenate(h_all)
    x_idx_train = np.concatenate(x_idx_all)
    sigma2_train = np.concatenate(sigma2_all)

    # Shuffle
    perm = np.random.permutation(len(y_train))
    y_train, h_train, x_idx_train, sigma2_train = (
        y_train[perm], h_train[perm], x_idx_train[perm], sigma2_train[perm])

    # Validation data
    y_val_all, h_val_all, x_idx_val_all, sigma2_val_all = [], [], [], []
    n_val_per = n_val // len(snr_range_db)
    for snr_db in snr_range_db:
        y, h_est, h_true, x_k, x_all_sym, x_k_idx, s2 = generate_noma_data(
            n_val_per, snr_db, mod_type, user_k, P_arr, csi_error_var)
        y_val_all.append(y)
        h_val_all.append(h_est)
        x_idx_val_all.append(x_k_idx)
        sigma2_val_all.append(np.full(n_val_per, s2))

    y_val = np.concatenate(y_val_all)
    h_val = np.concatenate(h_val_all)
    x_idx_val = np.concatenate(x_idx_val_all)
    sigma2_val_arr = np.concatenate(sigma2_val_all)

    # Build model
    model = LearnedMUD(M, K, d=d, L=L, user_k=user_k,
                       use_attention=use_attention).to(DEVICE)
    model.set_constellation(S)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    sqrt_Pk = np.sqrt(P_arr[user_k])

    # Prepare training tensors
    def make_tensors(y_np, h_np, sigma2_np, x_idx_np):
        n = len(y_np)
        y_re = torch.tensor(np.real(y_np), dtype=torch.float32)
        y_im = torch.tensor(np.imag(y_np), dtype=torch.float32)
        y_abs = torch.sqrt(y_re**2 + y_im**2)
        y_ang = torch.atan2(y_im, y_re)
        h_re = torch.tensor(np.real(h_np), dtype=torch.float32)
        h_im = torch.tensor(np.imag(h_np), dtype=torch.float32)
        h_abs = torch.sqrt(h_re**2 + h_im**2)
        h_ang = torch.atan2(h_im, h_re)

        S_c = torch.tensor(S, dtype=torch.complex64)
        y_c = torch.tensor(y_np, dtype=torch.complex64)
        h_c = torch.tensor(h_np, dtype=torch.complex64)
        dists = torch.abs(y_c[:, None] - h_c[:, None] * sqrt_Pk * S_c[None, :]).float()

        P_vec = torch.tensor(P_arr, dtype=torch.float32).unsqueeze(0).expand(n, -1)
        sigma2_t = torch.tensor(sigma2_np, dtype=torch.float32)
        labels = torch.tensor(x_idx_np, dtype=torch.long)

        return (y_re, y_im, y_abs, y_ang, dists,
                h_abs, h_ang, h_re, h_im, P_vec, sigma2_t, labels)

    train_tensors = make_tensors(y_train, h_train, sigma2_train, x_idx_train)
    val_tensors = make_tensors(y_val, h_val, sigma2_val_arr, x_idx_val)

    train_ds = TensorDataset(*train_tensors)
    val_ds = TensorDataset(*val_tensors)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              pin_memory=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False,
                            pin_memory=True, num_workers=0)

    best_val_loss = float('inf')
    best_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for batch_data in train_loader:
            batch_data = [t.to(DEVICE) for t in batch_data]
            (y_re, y_im, y_abs, y_ang, dists,
             h_abs, h_ang, h_re, h_im, P_vec, sigma2_t, labels) = batch_data

            logits = model(y_re, y_im, y_abs, y_ang, dists,
                          h_abs, h_ang, h_re, h_im,
                          P_vec, sigma2_t, sqrt_Pk)

            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / n_batches

        # Validation
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_data in val_loader:
                    batch_data = [t.to(DEVICE) for t in batch_data]
                    (y_re, y_im, y_abs, y_ang, dists,
                     h_abs, h_ang, h_re, h_im, P_vec, sigma2_t, labels) = batch_data

                    logits = model(y_re, y_im, y_abs, y_ang, dists,
                                  h_abs, h_ang, h_re, h_im,
                                  P_vec, sigma2_t, sqrt_Pk)

                    val_loss += criterion(logits, labels).item()
                    preds = logits.argmax(dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

            val_acc = correct / total
            val_loss_avg = val_loss / len(val_loader)
            print(f"    Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {avg_loss:.4f} | "
                  f"Val Loss: {val_loss_avg:.4f} | "
                  f"Val Acc: {val_acc:.4f}")

            if val_loss_avg < best_val_loss:
                best_val_loss = val_loss_avg
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})

    return model


def evaluate_learned_mud(model, y_k, h_k, P_arr, sigma2, user_k, S,
                         batch_size=4096):
    """Evaluate trained model on test data."""
    model.eval()
    sqrt_Pk = np.sqrt(P_arr[user_k])
    n = len(y_k)
    all_preds = []

    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            feats = prepare_features(y_k[start:end], h_k[start:end],
                                     P_arr, sigma2, user_k, S, DEVICE)
            (y_re, y_im, y_abs, y_ang, dists,
             h_abs, h_ang, h_re, h_im, P_vec, sigma2_t, _sqrt_Pk) = feats

            logits = model(y_re, y_im, y_abs, y_ang, dists,
                          h_abs, h_ang, h_re, h_im,
                          P_vec, sigma2_t, sqrt_Pk)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)

    return np.concatenate(all_preds)


# ============================================================
# Main Simulation: BER/SER vs SNR
# ============================================================
def run_ber_ser_simulation(mod_type, user_k=0, P_arr=None,
                           n_test=200000, csi_error_var=0.0,
                           model=None):
    """
    Run full simulation for one modulation type.
    Returns dict of {method_name: (snr_arr, error_rate_arr)}.
    """
    if P_arr is None:
        P_arr = P_DEFAULT
    S = get_constellation(mod_type)
    M = len(S)

    if mod_type == 'BPSK':
        snr_range = np.arange(0, 31, 3)
    elif mod_type == 'QPSK':
        snr_range = np.arange(0, 31, 3)
    else:
        snr_range = np.arange(5, 36, 3)

    results = {name: [] for name in
               ['Single-User', 'Conv. SIC', 'Oracle SIC', 'Exact MAP', 'Proposed']}

    metric = 'BER' if mod_type == 'BPSK' else 'SER'
    print(f"\n{'='*60}")
    print(f"  Evaluating {mod_type} | {metric} vs SNR | User {user_k+1}")
    print(f"  P = {P_arr} | CSI error = {csi_error_var}")
    print(f"{'='*60}")

    for snr_db in snr_range:
        np.random.seed(SEED + int(snr_db * 100))
        y, h_est, h_true, x_k, x_all, x_k_idx, sigma2 = generate_noma_data(
            n_test, snr_db, mod_type, user_k, P_arr, csi_error_var)

        # Single-user
        pred_su = detect_single_user(y, h_est, user_k, S, sigma2, P_arr)
        ser_su = np.mean(pred_su != x_k_idx)
        results['Single-User'].append(ser_su)

        # Conv SIC
        pred_sic = detect_conv_sic(y, h_est, user_k, S, sigma2, P_arr)
        ser_sic = np.mean(pred_sic != x_k_idx)
        results['Conv. SIC'].append(ser_sic)

        # Oracle SIC
        pred_oracle = detect_oracle_sic(y, h_est, user_k, x_all, S, sigma2, P_arr)
        ser_oracle = np.mean(pred_oracle != x_k_idx)
        results['Oracle SIC'].append(ser_oracle)

        # Exact MAP
        pred_map = detect_exact_map(y, h_est, user_k, S, sigma2, P_arr)
        ser_map = np.mean(pred_map != x_k_idx)
        results['Exact MAP'].append(ser_map)

        # Proposed learned MUD
        if model is not None:
            pred_learned = evaluate_learned_mud(model, y, h_est, P_arr,
                                                sigma2, user_k, S)
            ser_learned = np.mean(pred_learned != x_k_idx)
            results['Proposed'].append(ser_learned)

        print(f"  SNR={snr_db:2d} dB | SU={ser_su:.2e} | "
              f"SIC={ser_sic:.2e} | Oracle={ser_oracle:.2e} | "
              f"MAP={ser_map:.2e}" +
              (f" | Proposed={results['Proposed'][-1]:.2e}" if model else ""))

    if model is None:
        del results['Proposed']

    return snr_range, results


# ============================================================
# Plotting Functions (IEEE-quality)
# ============================================================
def setup_ieee_style():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'font.size': 9,
        'axes.labelsize': 10,
        'axes.titlesize': 10,
        'legend.fontsize': 7.5,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'lines.linewidth': 1.2,
        'lines.markersize': 5,
        'figure.figsize': (3.5, 2.8),
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
        'text.usetex': False,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linewidth': 0.5,
    })

MARKERS = {
    'Single-User': ('v', 'gray'),
    'Conv. SIC': ('s', '#d62728'),
    'Oracle SIC': ('^', '#2ca02c'),
    'Exact MAP': ('D', '#1f77b4'),
    'Proposed': ('o', '#ff7f0e'),
}
LINESTYLES = {
    'Single-User': ':',
    'Conv. SIC': '--',
    'Oracle SIC': '-.',
    'Exact MAP': '-',
    'Proposed': '-',
}


def plot_ber_ser(snr_range, results, mod_type, filename, user_k=0):
    """Plot BER/SER vs SNR for one modulation."""
    setup_ieee_style()
    fig, ax = plt.subplots()

    metric = 'BER' if mod_type == 'BPSK' else 'SER'

    plot_order = ['Single-User', 'Conv. SIC', 'Oracle SIC', 'Exact MAP', 'Proposed']

    for name in plot_order:
        if name not in results:
            continue
        vals = np.array(results[name])
        vals = np.maximum(vals, 1e-6)  # floor for log plot
        marker, color = MARKERS[name]
        ls = LINESTYLES[name]
        lw = 2.0 if name == 'Proposed' else 1.2
        ax.semilogy(snr_range, vals, marker=marker, color=color,
                    linestyle=ls, linewidth=lw, label=name,
                    markevery=1, markersize=5 if name != 'Proposed' else 6,
                    markerfacecolor='none' if name != 'Proposed' else color,
                    zorder=10 if name == 'Proposed' else 5)

    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel(metric)
    ax.legend(loc='lower left', framealpha=0.9, edgecolor='gray')
    ax.set_ylim(bottom=5e-5)

    mod_display = mod_type if mod_type != '16QAM' else '16-QAM'
    ax.set_title(f'{mod_display}, $K={K}$ users, User {user_k+1}')

    fig.savefig(filename, format='eps')
    fig.savefig(filename.replace('.eps', '.png'), format='png', dpi=300)
    plt.close(fig)
    print(f"  Saved: {filename}")


def plot_constellation_geometry(filename):
    """
    Plot composite constellation for equal vs differentiated power.
    Supports the identifiability theory (Section III).
    """
    setup_ieee_style()
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.0))

    S_qpsk = get_constellation('QPSK')
    M = len(S_qpsk)

    configs = [
        (np.array([1/3, 1/3, 1/3]), 'Equal Power\n$P_1{=}P_2{=}P_3{=}1/3$'),
        (P_DEFAULT, 'Differentiated Power\n$P_1{=}0.2,\\, P_2{=}0.3,\\, P_3{=}0.5$'),
    ]

    colors_user1 = {0: '#1f77b4', 1: '#d62728', 2: '#2ca02c', 3: '#ff7f0e'}

    for ax_idx, (P_arr, title) in enumerate(configs):
        ax = axes[ax_idx]
        sqrt_P = np.sqrt(P_arr)

        for combo in product(range(M), repeat=K):
            s1, s2, s3 = combo
            c_point = sqrt_P[0]*S_qpsk[s1] + sqrt_P[1]*S_qpsk[s2] + sqrt_P[2]*S_qpsk[s3]

            color = colors_user1[s1]
            ax.plot(np.real(c_point), np.imag(c_point), 'o',
                   color=color, markersize=2.5, alpha=0.7)

        ax.set_xlabel('In-phase')
        ax.set_ylabel('Quadrature')
        ax.set_title(title, fontsize=8)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        # Count unique points
        all_points = []
        for combo in product(range(M), repeat=K):
            s1, s2, s3 = combo
            c_point = sqrt_P[0]*S_qpsk[s1] + sqrt_P[1]*S_qpsk[s2] + sqrt_P[2]*S_qpsk[s3]
            all_points.append(c_point)
        unique = len(set([round(p.real, 6) + 1j*round(p.imag, 6) for p in all_points]))
        total = len(all_points)
        ax.text(0.02, 0.98, f'{unique}/{total} distinct',
               transform=ax.transAxes, fontsize=7, va='top',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.8))

    # Legend for user 1 symbols
    from matplotlib.lines import Line2D
    S_labels = [f'$s_1 = {i+1}$' for i in range(M)]
    legend_elements = [Line2D([0], [0], marker='o', color='w',
                             markerfacecolor=colors_user1[i],
                             markersize=5, label=S_labels[i])
                      for i in range(M)]
    fig.legend(handles=legend_elements, loc='lower center',
              ncol=4, fontsize=7, framealpha=0.9,
              bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
    fig.savefig(filename, format='eps')
    fig.savefig(filename.replace('.eps', '.png'), format='png', dpi=300)
    plt.close(fig)
    print(f"  Saved: {filename}")


def plot_power_impact(filename, model_proposed=None):
    """
    Show detection performance for different power allocations.
    Validates the identifiability theory.
    """
    setup_ieee_style()
    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    mod_type = 'QPSK'
    S = get_constellation(mod_type)
    user_k = 0
    n_test = 200000
    snr_range = np.arange(0, 31, 3)

    power_configs = [
        (np.array([1/3, 1/3, 1/3]), 'Equal ($1/3,1/3,1/3$)', ':'),
        (np.array([0.25, 0.35, 0.40]), 'Mild ($0.25,0.35,0.40$)', '--'),
        (np.array([0.2, 0.3, 0.5]), 'Moderate ($0.2,0.3,0.5$)', '-'),
        (np.array([0.1, 0.2, 0.7]), 'Strong ($0.1,0.2,0.7$)', '-.'),
    ]

    colors = ['#d62728', '#ff7f0e', '#1f77b4', '#2ca02c']
    n_test_pwr = 100000  # Reduced for MAP speed

    for (P_arr, label, ls), color in zip(power_configs, colors):
        ser_map_list = []
        for snr_db in snr_range:
            np.random.seed(SEED + int(snr_db * 100) + hash(str(P_arr)) % 10000)
            y, h_est, h_true, x_k, x_all, x_k_idx, sigma2 = generate_noma_data(
                n_test_pwr, snr_db, mod_type, user_k, P_arr)
            pred_map = detect_exact_map(y, h_est, user_k, S, sigma2, P_arr)
            ser = np.mean(pred_map != x_k_idx)
            ser_map_list.append(max(ser, 1e-6))
            print(f"    Power impact: P={P_arr}, SNR={snr_db}, SER={ser:.2e}")

        ax.semilogy(snr_range, ser_map_list, linestyle=ls, color=color,
                    linewidth=1.5, label=label, marker='o', markersize=4)

    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('SER (Exact MAP)')
    ax.set_title(f'QPSK, $K={K}$, User 1: Power Allocation Impact')
    ax.legend(fontsize=6.5, loc='lower left', framealpha=0.9)
    ax.set_ylim(bottom=5e-5)

    fig.savefig(filename, format='eps')
    fig.savefig(filename.replace('.eps', '.png'), format='png', dpi=300)
    plt.close(fig)
    print(f"  Saved: {filename}")


def plot_imperfect_csi(filename, models_csi=None):
    """
    Compare proposed vs SIC under imperfect CSI.
    """
    setup_ieee_style()
    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    mod_type = 'QPSK'
    S = get_constellation(mod_type)
    user_k = 0
    n_test = 200000
    snr_range = np.arange(0, 31, 3)

    csi_errors = [0.0, 0.01, 0.05]
    colors_sic = ['#d62728', '#ff6666', '#ffaaaa']
    colors_prop = ['#1f77b4', '#6699cc', '#aaccee']
    ls_sic = ['--', '--', '--']
    ls_prop = ['-', '-', '-']
    markers_sic = ['s', 's', 's']
    markers_prop = ['o', 'o', 'o']

    for i, csi_err in enumerate(csi_errors):
        ser_sic_list = []
        ser_proposed_list = []

        label_suffix = f'$\\sigma_e^2={csi_err}$' if csi_err > 0 else 'Perfect CSI'

        for snr_db in snr_range:
            np.random.seed(SEED + int(snr_db * 100))
            y, h_est, h_true, x_k, x_all, x_k_idx, sigma2 = generate_noma_data(
                n_test, snr_db, mod_type, user_k, csi_error_var=csi_err)

            pred_sic = detect_conv_sic(y, h_est, user_k, S, sigma2, P_DEFAULT)
            ser_sic_list.append(max(np.mean(pred_sic != x_k_idx), 1e-6))

            if models_csi is not None and i < len(models_csi) and models_csi[i] is not None:
                pred_prop = evaluate_learned_mud(models_csi[i], y, h_est,
                                                 P_DEFAULT, sigma2, user_k, S)
                ser_proposed_list.append(max(np.mean(pred_prop != x_k_idx), 1e-6))

        ax.semilogy(snr_range, ser_sic_list, linestyle=ls_sic[i],
                    color=colors_sic[i], marker=markers_sic[i], markersize=4,
                    linewidth=1.2, label=f'Conv. SIC, {label_suffix}',
                    markerfacecolor='none')

        if ser_proposed_list:
            ax.semilogy(snr_range, ser_proposed_list, linestyle=ls_prop[i],
                        color=colors_prop[i], marker=markers_prop[i], markersize=5,
                        linewidth=1.5, label=f'Proposed, {label_suffix}')

    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('SER')
    ax.set_title(f'QPSK, $K={K}$, User 1: Imperfect CSI')
    ax.legend(fontsize=6, loc='lower left', framealpha=0.9, ncol=2)
    ax.set_ylim(bottom=5e-5)

    fig.savefig(filename, format='eps')
    fig.savefig(filename.replace('.eps', '.png'), format='png', dpi=300)
    plt.close(fig)
    print(f"  Saved: {filename}")


def plot_per_user(snr_range, results_per_user, mod_type, filename):
    """
    Compare performance across all K users.
    """
    setup_ieee_style()
    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    metric = 'BER' if mod_type == 'BPSK' else 'SER'
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    for uk in range(K):
        # SIC
        vals_sic = np.array(results_per_user[uk]['Conv. SIC'])
        vals_sic = np.maximum(vals_sic, 1e-6)
        ax.semilogy(snr_range, vals_sic, '--', color=colors[uk],
                    marker='s', markersize=4, markerfacecolor='none',
                    linewidth=1.0, label=f'SIC, User {uk+1}')

        # Proposed
        if 'Proposed' in results_per_user[uk]:
            vals_prop = np.array(results_per_user[uk]['Proposed'])
            vals_prop = np.maximum(vals_prop, 1e-6)
            ax.semilogy(snr_range, vals_prop, '-', color=colors[uk],
                        marker='o', markersize=5, linewidth=1.5,
                        label=f'Proposed, User {uk+1}')

    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel(metric)
    ax.set_title(f'{mod_type}, $K={K}$ users: Per-User Comparison')
    ax.legend(fontsize=6, loc='lower left', framealpha=0.9, ncol=2)
    ax.set_ylim(bottom=5e-5)

    fig.savefig(filename, format='eps')
    fig.savefig(filename.replace('.eps', '.png'), format='png', dpi=300)
    plt.close(fig)
    print(f"  Saved: {filename}")


def run_ablation(mod_type='QPSK', snr_db=15, user_k=0, n_test=200000):
    """Run ablation study at a single SNR point."""
    S = get_constellation(mod_type)
    P_arr = P_DEFAULT

    print(f"\n{'='*60}")
    print(f"  Ablation Study: {mod_type} at SNR = {snr_db} dB")
    print(f"{'='*60}")

    # Generate test data
    np.random.seed(SEED)
    y, h_est, h_true, x_k, x_all, x_k_idx, sigma2 = generate_noma_data(
        n_test, snr_db, mod_type, user_k, P_arr)

    # Baseline: Conv SIC
    pred_sic = detect_conv_sic(y, h_est, user_k, S, sigma2, P_arr)
    ser_sic = np.mean(pred_sic != x_k_idx)

    # Oracle SIC
    pred_oracle = detect_oracle_sic(y, h_est, user_k, x_all, S, sigma2, P_arr)
    ser_oracle = np.mean(pred_oracle != x_k_idx)

    # Exact MAP
    pred_map = detect_exact_map(y, h_est, user_k, S, sigma2, P_arr)
    ser_map = np.mean(pred_map != x_k_idx)

    results = {
        'Conv. SIC': ser_sic,
        'Oracle SIC': ser_oracle,
        'Exact MAP': ser_map,
    }

    # Ablation configurations
    ablation_configs = [
        {'L': 4, 'use_attention': True, 'name': 'Proposed ($L{=}4$, attention)'},
        {'L': 4, 'use_attention': False, 'name': 'Without attention ($L{=}4$)'},
        {'L': 2, 'use_attention': True, 'name': '$L{=}2$ iterations'},
        {'L': 1, 'use_attention': True, 'name': '$L{=}1$ iteration'},
    ]

    snr_range_train = np.arange(0, 31, 2) if mod_type != '16QAM' else np.arange(5, 36, 2)

    for cfg in ablation_configs:
        model = train_model(mod_type, user_k=user_k, d=128, L=cfg['L'],
                           use_attention=cfg['use_attention'],
                           snr_range_db=snr_range_train,
                           n_train=150000, epochs=60, batch_size=1024)
        pred = evaluate_learned_mud(model, y, h_est, P_arr, sigma2, user_k, S)
        ser = np.mean(pred != x_k_idx)
        results[cfg['name']] = ser
        print(f"  {cfg['name']}: SER = {ser:.2e}")
        del model
        torch.cuda.empty_cache()

    print(f"\n  Conv. SIC:  SER = {ser_sic:.2e}")
    print(f"  Oracle SIC: SER = {ser_oracle:.2e}")
    print(f"  Exact MAP:  SER = {ser_map:.2e}")

    return results


# ============================================================
# Main Execution
# ============================================================
def main():
    t0 = time.time()

    # ---- Figure: Composite Constellation Geometry ----
    print("\n" + "="*60)
    print("  FIGURE: Composite Constellation Geometry")
    print("="*60)
    plot_constellation_geometry(os.path.join(OUT_DIR, 'fig_constellation_geometry.eps'))

    # ---- Figure: Power Allocation Impact ----
    print("\n" + "="*60)
    print("  FIGURE: Power Allocation Impact")
    print("="*60)
    plot_power_impact(os.path.join(OUT_DIR, 'fig_power_impact.eps'))

    # ---- Train models and run BER/SER simulations ----
    trained_models = {}

    for mod_type in ['BPSK', 'QPSK', '16QAM']:
        print(f"\n{'#'*60}")
        print(f"  MODULATION: {mod_type}")
        print(f"{'#'*60}")

        # Train model
        if mod_type == 'BPSK':
            snr_train = np.arange(0, 31, 2)
        elif mod_type == 'QPSK':
            snr_train = np.arange(0, 31, 2)
        else:
            snr_train = np.arange(5, 36, 2)

        model = train_model(mod_type, user_k=0, d=128, L=4,
                           use_attention=True,
                           snr_range_db=snr_train,
                           n_train=200000, epochs=80, batch_size=1024)
        trained_models[mod_type] = model

        # Evaluate
        snr_range, results = run_ber_ser_simulation(
            mod_type, user_k=0, n_test=200000, model=model)

        # Plot
        mod_lower = mod_type.lower()
        filename = os.path.join(OUT_DIR, f'fig_{mod_lower}_performance.eps')
        plot_ber_ser(snr_range, results, mod_type, filename)

        torch.cuda.empty_cache()

    # ---- Per-User Comparison (QPSK) ----
    print("\n" + "="*60)
    print("  FIGURE: Per-User Comparison (QPSK)")
    print("="*60)

    # Train models for users 1, 2, 3
    per_user_models = {}
    per_user_models[0] = trained_models['QPSK']  # Already trained for user 0

    for uk in [1, 2]:
        per_user_models[uk] = train_model('QPSK', user_k=uk, d=128, L=4,
                                          use_attention=True,
                                          n_train=150000, epochs=60,
                                          batch_size=1024)

    S_qpsk = get_constellation('QPSK')
    snr_range_pu = np.arange(0, 31, 3)
    results_per_user = {}

    for uk in range(K):
        results_per_user[uk] = {'Conv. SIC': [], 'Proposed': []}
        for snr_db in snr_range_pu:
            np.random.seed(SEED + int(snr_db * 100))
            y, h_est, h_true, x_k, x_all, x_k_idx, sigma2 = generate_noma_data(
                200000, snr_db, 'QPSK', uk)

            pred_sic = detect_conv_sic(y, h_est, uk, S_qpsk, sigma2, P_DEFAULT)
            results_per_user[uk]['Conv. SIC'].append(np.mean(pred_sic != x_k_idx))

            pred_prop = evaluate_learned_mud(per_user_models[uk], y, h_est,
                                              P_DEFAULT, sigma2, uk, S_qpsk)
            results_per_user[uk]['Proposed'].append(np.mean(pred_prop != x_k_idx))

        print(f"  User {uk+1} done.")

    plot_per_user(snr_range_pu, results_per_user, 'QPSK',
                  os.path.join(OUT_DIR, 'fig_per_user_qpsk.eps'))

    torch.cuda.empty_cache()

    # ---- Imperfect CSI (QPSK) ----
    print("\n" + "="*60)
    print("  FIGURE: Imperfect CSI (QPSK)")
    print("="*60)

    models_csi = []
    for csi_err in [0.0, 0.01, 0.05]:
        if csi_err == 0.0:
            models_csi.append(trained_models['QPSK'])
        else:
            m = train_model('QPSK', user_k=0, d=128, L=4,
                           use_attention=True, csi_error_var=csi_err,
                           n_train=150000, epochs=60, batch_size=1024)
            models_csi.append(m)

    plot_imperfect_csi(os.path.join(OUT_DIR, 'fig_imperfect_csi.eps'),
                       models_csi=models_csi)

    torch.cuda.empty_cache()

    # ---- Ablation Study ----
    print("\n" + "="*60)
    print("  ABLATION STUDY")
    print("="*60)
    ablation_results = run_ablation('QPSK', snr_db=15, n_test=200000)

    # Print ablation table
    print("\n  Ablation Results (QPSK, SNR=15 dB):")
    print("  " + "-"*50)
    for name, ser in ablation_results.items():
        print(f"  {name:45s} : {ser:.2e}")

    # ---- Summary Table ----
    print("\n" + "="*60)
    print("  PERFORMANCE SUMMARY")
    print("="*60)

    # Recompute at a reference SNR for summary
    summary_snrs = {'BPSK': 18, 'QPSK': 18, '16QAM': 24}
    for mod_type, ref_snr in summary_snrs.items():
        S = get_constellation(mod_type)
        np.random.seed(SEED)
        y, h_est, h_true, x_k, x_all, x_k_idx, sigma2 = generate_noma_data(
            300000, ref_snr, mod_type, 0)

        pred_sic = detect_conv_sic(y, h_est, 0, S, sigma2, P_DEFAULT)
        ser_sic = np.mean(pred_sic != x_k_idx)

        pred_oracle = detect_oracle_sic(y, h_est, 0, x_all, S, sigma2, P_DEFAULT)
        ser_oracle = np.mean(pred_oracle != x_k_idx)

        pred_map = detect_exact_map(y, h_est, 0, S, sigma2, P_DEFAULT)
        ser_map = np.mean(pred_map != x_k_idx)

        pred_prop = evaluate_learned_mud(trained_models[mod_type], y, h_est,
                                          P_DEFAULT, sigma2, 0, S)
        ser_prop = np.mean(pred_prop != x_k_idx)

        print(f"\n  {mod_type} at SNR={ref_snr} dB:")
        print(f"    Conv. SIC:  {ser_sic:.2e}")
        print(f"    Oracle SIC: {ser_oracle:.2e}")
        print(f"    Exact MAP:  {ser_map:.2e}")
        print(f"    Proposed:   {ser_prop:.2e}")

    elapsed = time.time() - t0
    print(f"\n  Total simulation time: {elapsed/60:.1f} minutes")
    print("  All figures saved to:", OUT_DIR)


if __name__ == '__main__':
    main()
