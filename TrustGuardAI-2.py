# trustguardai.py
import time
import math
import numpy as np
from typing import Tuple, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, accuracy_score

# ---------------------------
# Utilities: windowing, normalization
# ---------------------------

def sliding_windows(X: np.ndarray, L: int, stride: int = 1) -> np.ndarray:
    """
    Convert multivariate time series X (T x d) into sliding windows of shape (N, L, d).
    """
    T, d = X.shape
    windows = []
    for start in range(0, T - L + 1, stride):
        windows.append(X[start:start+L])
    return np.stack(windows, axis=0)  # (N, L, d)

def zscore_fit_transform(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma == 0] = 1.0
    return (X - mu) / sigma, mu, sigma

def zscore_transform(X: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return (X - mu) / sigma

# ---------------------------
# Dataset wrapper
# ---------------------------

class WindowDataset(Dataset):
    def __init__(self, windows: np.ndarray):
        # windows: (N, L, d)
        self.windows = torch.tensor(windows, dtype=torch.float32)

    def __len__(self):
        return self.windows.shape[0]

    def __getitem__(self, idx):
        return self.windows[idx]  # (L, d)

# ---------------------------
# Model: LSTM Autoencoder + Attention + Dropout
# ---------------------------

class AttentionModule(nn.Module):
    """
    Produces attention mask A in [0,1] of shape (batch, L, d) from input windows.
    Simple per-time-step, per-feature attention via a small feedforward network.
    """
    def __init__(self, L: int, d: int, hidden: int = 128):
        super().__init__()
        # We'll compute attention per time step using a linear layer applied to features
        self.fc1 = nn.Linear(d, hidden)
        self.fc2 = nn.Linear(hidden, d)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # x: (batch, L, d)
        # apply per-time-step
        b, L, d = x.shape
        x_flat = x.view(-1, d)  # (b*L, d)
        h = F.relu(self.fc1(x_flat))
        a = self.sigmoid(self.fc2(h))  # (b*L, d)
        A = a.view(b, L, d)
        return A  # (batch, L, d)

class LSTMAutoencoderWithAttention(nn.Module):
    def __init__(self, input_dim: int, hidden1: int = 256, hidden2: int = 128, latent_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.latent_dim = latent_dim
        self.dropout_rate = dropout

        # Attention module
        self.attention = AttentionModule(L=50, d=input_dim, hidden=128)

        # Encoder LSTMs
        self.enc_lstm1 = nn.LSTM(input_dim, hidden1, batch_first=True)
        self.enc_dropout1 = nn.Dropout(dropout)
        self.enc_lstm2 = nn.LSTM(hidden1, hidden2, batch_first=True)
        self.enc_dropout2 = nn.Dropout(dropout)

        # Latent projection
        self.latent_fc = nn.Linear(hidden2, latent_dim)

        # Decoder LSTMs (mirror)
        self.dec_lstm1 = nn.LSTM(latent_dim, hidden2, batch_first=True)
        self.dec_dropout1 = nn.Dropout(dropout)
        self.dec_lstm2 = nn.LSTM(hidden2, hidden1, batch_first=True)
        self.dec_dropout2 = nn.Dropout(dropout)

        # Output projection
        self.output_fc = nn.Linear(hidden1, input_dim)

    def encode(self, x):  # x: (b, L, d)
        # x already attended externally if needed
        out, _ = self.enc_lstm1(x)  # (b, L, hidden1)
        out = self.enc_dropout1(out)
        out, _ = self.enc_lstm2(out)  # (b, L, hidden2)
        out = self.enc_dropout2(out)
        # take last time-step
        last = out[:, -1, :]  # (b, hidden2)
        z = self.latent_fc(last)  # (b, latent_dim)
        return z

    def decode(self, z, L: int):
        # z: (b, latent_dim)
        # repeat latent for L time steps
        b = z.size(0)
        z_rep = z.unsqueeze(1).repeat(1, L, 1)  # (b, L, latent_dim)
        out, _ = self.dec_lstm1(z_rep)
        out = self.dec_dropout1(out)
        out, _ = self.dec_lstm2(out)
        out = self.dec_dropout2(out)
        recon = self.output_fc(out)  # (b, L, d)
        return recon

    def forward(self, x):  # x: (b, L, d)
        # compute attention mask
        A = self.attention(x)  # (b, L, d)
        x_att = x * A
        z = self.encode(x_att)
        recon = self.decode(z, x.shape[1])
        return recon, A

# ---------------------------
# Training and evaluation helpers
# ---------------------------

def train_model(model: nn.Module,
                train_loader: DataLoader,
                val_loader: DataLoader,
                device: torch.device,
                lr: float = 1e-3,
                weight_decay: float = 1e-5,
                max_epochs: int = 100,
                patience: int = 10):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    model.to(device)

    best_val = float('inf')
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)  # (b, L, d)
            optimizer.zero_grad()
            recon, _ = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch.size(0)
        train_loss /= len(train_loader.dataset)

        # validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                recon, _ = model(batch)
                loss = criterion(recon, batch)
                val_loss += loss.item() * batch.size(0)
        val_loss /= len(val_loader.dataset)

        # early stopping
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        print(f"Epoch {epoch:03d} Train Loss: {train_loss:.6f} Val Loss: {val_loss:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model

# ---------------------------
# Inference: single-pass and MC-dropout
# ---------------------------

def inference_window_score(model: nn.Module, window: np.ndarray, device: torch.device) -> Dict:
    """
    Compute single deterministic pass: reconstruction, attention, anomaly score (MSE per window).
    window: (L, d) numpy
    returns dict with keys: recon (L,d), attention (L,d), score (float)
    """
    model.eval()
    x = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(device)  # (1,L,d)
    with torch.no_grad():
        recon, A = model(x)
    recon = recon.squeeze(0).cpu().numpy()
    A = A.squeeze(0).cpu().numpy()
    mse = np.mean((window - recon) ** 2)
    return {"recon": recon, "attention": A, "score": mse}

def mc_dropout_scores(model: nn.Module, window: np.ndarray, device: torch.device, K: int = 10) -> Dict:
    """
    Perform K stochastic forward passes with dropout active to estimate mean score and variance.
    Returns mean_score, var_score, mean_attention, attention_var, per_pass_scores list.
    """
    model.train()  # enable dropout
    x = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(device)
    scores = []
    attentions = []
    with torch.no_grad():
        for _ in range(K):
            recon, A = model(x)
            recon = recon.squeeze(0).cpu().numpy()
            A_np = A.squeeze(0).cpu().numpy()
            mse = np.mean((window - recon) ** 2)
            scores.append(mse)
            attentions.append(A_np)
    model.eval()
    scores = np.array(scores)
    attentions = np.stack(attentions, axis=0)  # (K, L, d)
    mean_score = float(scores.mean())
    var_score = float(scores.var(ddof=0))
    mean_attention = attentions.mean(axis=0)
    attention_var = attentions.var(axis=0)
    # confidence as exp(-variance) as in paper
    confidence = math.exp(-var_score)
    return {
        "mean_score": mean_score,
        "var_score": var_score,
        "confidence": confidence,
        "mean_attention": mean_attention,
        "attention_var": attention_var,
        "per_pass_scores": scores
    }

# ---------------------------
# Evaluation metrics
# ---------------------------

def evaluate_anomaly_detection(scores: np.ndarray, labels: np.ndarray, threshold: float = None) -> Dict:
    """
    scores: (N,) anomaly scores (higher -> more anomalous)
    labels: (N,) binary ground truth (1 anomaly, 0 normal)
    If threshold is None, compute AUROC and return best threshold by Youden or use ROC AUC only.
    """
    results = {}
    try:
        auc = roc_auc_score(labels, scores)
    except Exception:
        auc = float('nan')
    results['auroc'] = auc

    if threshold is None:
        # choose threshold by maximizing F1 on validation-like split (here we compute best threshold)
        best_f1 = -1.0
        best_thresh = None
        for t in np.linspace(scores.min(), scores.max(), 200):
            preds = (scores >= t).astype(int)
            p, r, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = t
        threshold = best_thresh
    preds = (scores >= threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
    results.update({'precision': p, 'recall': r, 'f1': f1, 'threshold': threshold})
    return results

# ---------------------------
# Example end-to-end usage
# ---------------------------

def example_pipeline(X_train: np.ndarray,
                     X_val: np.ndarray,
                     X_test: np.ndarray,
                     y_test: np.ndarray,
                     L: int = 50,
                     device_str: str = 'cpu'):
    """
    X_train, X_val, X_test: arrays of shape (T, d) for each split (continuous time series).
    y_test: binary labels aligned to windows (N_test,) where N_test = number of windows from X_test.
    """
    device = torch.device(device_str)
    # 1. Fit normalization on train
    X_train_norm, mu, sigma = zscore_fit_transform(X_train)
    X_val_norm = zscore_transform(X_val, mu, sigma)
    X_test_norm = zscore_transform(X_test, mu, sigma)

    # 2. Create windows
    stride = 1
    train_w = sliding_windows(X_train_norm, L, stride)
    val_w = sliding_windows(X_val_norm, L, stride)
    test_w = sliding_windows(X_test_norm, L, stride)

    # 3. Dataloaders
    batch_size = 64
    train_loader = DataLoader(WindowDataset(train_w), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(WindowDataset(val_w), batch_size=batch_size, shuffle=False)

    # 4. Model
    d = X_train.shape[1]
    model = LSTMAutoencoderWithAttention(input_dim=d, hidden1=256, hidden2=128, latent_dim=128, dropout=0.2)

    # 5. Train
    model = train_model(model, train_loader, val_loader, device, lr=1e-3, weight_decay=1e-5, max_epochs=100, patience=10)

    # 6. Inference on test windows with MC dropout
    K = 10  # MC passes
    scores = []
    confidences = []
    attentions = []
    latencies = []
    for w in test_w:
        t0 = time.perf_counter()
        res = mc_dropout_scores(model, w, device, K=K)
        t1 = time.perf_counter()
        scores.append(res['mean_score'])
        confidences.append(res['confidence'])
        attentions.append(res['mean_attention'])
        latencies.append((t1 - t0) * 1000.0)  # ms per window

    scores = np.array(scores)
    confidences = np.array(confidences)
    latencies = np.array(latencies)

    # 7. Evaluate
    eval_res = evaluate_anomaly_detection(scores, y_test, threshold=None)

    # 8. Latency summary
    latency_summary = {'mean_ms': float(latencies.mean()), 'std_ms': float(latencies.std())}

    return {
        'model': model,
        'scores': scores,
        'confidences': confidences,
        'attentions': attentions,
        'eval': eval_res,
        'latency': latency_summary
    }

# ---------------------------
# If run as script: small sanity test with synthetic data
# ---------------------------

if __name__ == "__main__":
    # Synthetic quick test (not real datasets)
    T = 5000
    d = 6
    L = 50
    # generate normal sine-like multivariate series
    t = np.arange(T)
    X = np.stack([np.sin(0.01 * t + phase) for phase in np.linspace(0, 2*np.pi, d)], axis=1)
    # inject anomalies in the tail
    X[T//2:T//2+100] += np.random.normal(3.0, 0.5, size=(100, d))
    # split
    X_train = X[:3000]
    X_val = X[3000:4000]
    X_test = X[4000:]
    # create labels for test windows (simple heuristic: any window overlapping anomaly region -> label 1)
    test_w = sliding_windows(zscore_transform(X_test, *zscore_fit_transform(X_train)[:2]), L)
    N_test = test_w.shape[0]
    # For synthetic, mark windows with mean absolute > threshold as anomalies
    y_test = (np.mean(np.abs(test_w), axis=(1,2)) > 0.5).astype(int)

    out = example_pipeline(X_train, X_val, X_test, y_test, L=L, device_str='cpu')
    print("Evaluation:", out['eval'])
    print("Latency (ms):", out['latency'])
