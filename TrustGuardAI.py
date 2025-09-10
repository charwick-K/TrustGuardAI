TrustGuardAI: Explainable Real‐Time Anomaly Detection Framework
Single‐file implementation combining model, training, inference, and plotting.
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------
# Model Definitions
# -------------------------

class StreamingLSTMAutoencoder(nn.Module):
    """
    Online LSTM Autoencoder for reconstruction-based anomaly scoring.
    """
    def __init__(self, input_dim, hidden_dim, latent_dim, dropout=0.1):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc_enc  = nn.Linear(hidden_dim, latent_dim)
        self.fc_dec  = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.LSTM(hidden_dim, input_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        enc_out, (h_n, _) = self.encoder(x)
        z = self.fc_enc(self.dropout(h_n[-1]))              # (batch, latent_dim)
        dec_in = self.fc_dec(z).unsqueeze(1) \
                 .repeat(1, x.size(1), 1)                  # (batch, seq_len, hidden_dim)
        dec_out, _ = self.decoder(dec_in)
        return dec_out, z

class AttentionExplain(nn.Module):
    """
    Learnable attention weights for feature‐level attribution.
    """
    def __init__(self, seq_len, input_dim):
        super().__init__()
        self.attn_weights = nn.Parameter(torch.randn(seq_len, input_dim))

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        weights = torch.sigmoid(self.attn_weights)            # (seq_len, input_dim)
        weights = weights.unsqueeze(0)                        # (1, seq_len, input_dim)
        attn_applied = x * weights                            # elementwise
        feature_imp = attn_applied.mean(dim=1)                # (batch, input_dim)
        return feature_imp, weights

class TrustGuardAI:
    """
    Wrapper combining streaming AE, attention explainability, and
    Bayesian dropout uncertainty estimation.
    """
    def __init__(self,
                 input_dim: int,
                 seq_len: int,
                 hidden_dim: int = 64,
                 latent_dim: int = 20,
                 dropout: float = 0.1,
                 threshold: float = 0.1,
                 device: str = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = StreamingLSTMAutoencoder(input_dim, hidden_dim, latent_dim, dropout)\
                        .to(self.device)
        self.attn  = AttentionExplain(seq_len, input_dim).to(self.device)
        self.threshold = threshold

    def anomaly_score(self, x: torch.Tensor):
        """
        Returns
          score: (batch,)    – mean reconstruction error
          loss_map: (batch, seq_len) – per‐step MSE
        """
        recon, _ = self.model(x)
        loss = F.mse_loss(recon, x, reduction='none')         # (batch, seq_len, feat)
        loss = loss.mean(dim=2)                               # (batch, seq_len)
        score = loss.mean(dim=1)                              # (batch,)
        return score, loss

    def interpret(self, x: torch.Tensor):
        """
        Returns feature importance and raw attention weights.
        """
        feat_imp, weights = self.attn(x)
        return feat_imp, weights

    def predict(self,
                x: torch.Tensor,
                mc_passes: int = 10):
        """
        Streaming prediction with Bayesian dropout.
        """
        x = x.to(self.device)
        # Enable dropout at inference
        self.model.train()
        scores = []
        for _ in range(mc_passes):
            with torch.no_grad():
                s, _ = self.anomaly_score(x)
                scores.append(s)
        scores = torch.stack(scores, dim=0)                   # (mc, batch)
        mean_score = scores.mean(dim=0)
        var_score  = scores.var(dim=0)
        conf       = torch.exp(-var_score)
        # Restore eval mode
        self.model.eval()

        # Single forward pass for feature attribution
        score, loss_map = self.anomaly_score(x)
        feat_imp, attn_w  = self.interpret(x)

        is_anom = (mean_score > self.threshold).float()
        return {
            'flag':      is_anom.cpu().numpy(),
            'score':     mean_score.cpu().numpy(),
            'confidence':conf.cpu().numpy(),
            'feature_importance': feat_imp.cpu().numpy(),
            'attention_weights':  attn_w.cpu().numpy(),
            'loss_map':   loss_map.cpu().numpy()
        }

# -------------------------
# Dataset Definition
# -------------------------

class TimeSeriesDataset(Dataset):
    """
    Windowed dataset from a NumPy array of shape (T, d).
    """
    def __init__(self, data: np.ndarray, window: int):
        X = []
        for i in range(len(data) - window + 1):
            X.append(data[i:i+window])
        self.X = np.stack(X)  # (N, window, d)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]).float()

# -------------------------
# Training Function
# -------------------------

def train(args):
    # Load data
    arr = np.load(args.data)        # shape (T, d)
    ds  = TimeSeriesDataset(arr, args.window)
    dl  = DataLoader(ds, batch_size=args.bs, shuffle=True, drop_last=True)

    # Model
    model = StreamingLSTMAutoencoder(
        input_dim=arr.shape[1],
        hidden_dim=args.hidden,
        latent_dim=args.latent,
        dropout=args.dropout
    ).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_loss = float('inf')

    # Training loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for x in dl:
            x = x.to(args.device)
            recon, _ = model(x)
            loss = F.mse_loss(recon, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
        avg_loss = running_loss / len(ds)
        print(f"Epoch {epoch:03d}  Loss: {avg_loss:.6f}")
        # Save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), args.checkpoint)
    print("Training complete. Best loss:", best_loss)

# -------------------------
# Inference Function
# -------------------------

def infer(args):
    # Load batch of sequences
    x = np.load(args.batch)           # shape (B, window, d)
    x_tensor = torch.from_numpy(x).float()

    tg = TrustGuardAI(
        input_dim=x.shape[2],
        seq_len = x.shape[1],
        hidden_dim=args.hidden,
        latent_dim=args.latent,
        dropout=args.dropout,
        threshold=args.threshold,
        device=args.device
    )
    # Load pretrained weights
    state = torch.load(args.checkpoint, map_location=tg.device)
    tg.model.load_state_dict(state)

    # Run prediction
    out = tg.predict(x_tensor, mc_passes=args.mc_passes)

    # Print results
    for i in range(x.shape[0]):
        print(f"Seq {i:02d} | Flag: {int(out['flag'][i])} | "
              f"Score: {out['score'][i]:.4f} | "
              f"Conf: {out['confidence'][i]:.4f}")

    # Plot attention heatmap for first sample
    attn = out['attention_weights'][0]  # shape (seq_len, d)
    plt.figure(figsize=(6,3))
    sns.heatmap(attn.T, cmap="viridis", cbar=True,
                xticklabels=5, yticklabels=np.arange(1,attn.shape[1]+1))
    plt.xlabel("Time Step")
    plt.ylabel("Feature Index")
    plt.title("Attention Heatmap")
    plt.tight_layout()
    os.makedirs(args.figures_dir, exist_ok=True)
    save_path = os.path.join(args.figures_dir, "attention_heatmap.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved attention heatmap to {save_path}")

# -------------------------
# Argument Parser & Main
# -------------------------

def main():
    parser = argparse.ArgumentParser(
        description="TrustGuardAI: streaming anomaly detection with explainability"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train subcommand
    p_train = subparsers.add_parser("train", help="Train the LSTM autoencoder")
    p_train.add_argument("--data",       type=str, required=True,
                         help="Path to .npy file of shape (T, d)")
    p_train.add_argument("--window",     type=int, default=30)
    p_train.add_argument("--hidden",     type=int, default=64)
    p_train.add_argument("--latent",     type=int, default=20)
    p_train.add_argument("--dropout",    type=float, default=0.1)
    p_train.add_argument("--bs",         type=int, default=64)
    p_train.add_argument("--lr",         type=float, default=1e-3)
    p_train.add_argument("--epochs",     type=int, default=50)
    p_train.add_argument("--checkpoint", type=str, default="ae.pth")
    p_train.add_argument("--device",     type=str, default=None)

    # Infer subcommand
    p_inf = subparsers.add_parser("infer", help="Run streaming inference")
    p_inf.add_argument("--batch",       type=str, required=True,
                       help=".npy file of shape (B, window, d)")
    p_inf.add_argument("--checkpoint",  type=str, default="ae.pth")
    p_inf.add_argument("--hidden",      type=int, default=64)
    p_inf.add_argument("--latent",      type=int, default=20)
    p_inf.add_argument("--dropout",     type=float, default=0.1)
    p_inf.add_argument("--threshold",   type=float, default=0.1)
    p_inf.add_argument("--mc_passes",   type=int, default=10)
    p_inf.add_argument("--figures_dir", type=str, default="figures")
    p_inf.add_argument("--device",      type=str, default=None)

    args = parser.parse_args()
    if args.command == "train":
        train(args)
    elif args.command == "infer":
        infer(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
```
