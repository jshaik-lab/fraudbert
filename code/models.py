"""
Deep learning model definitions for the FraudBERT paper.

Models:
  1. MLPClassifier         — 3-layer MLP baseline (sklearn-compatible wrapper)
  2. TransformerClassifier — TabTransformer-style attention baseline
  3. FraudBERTMLP          — Learnable projection + fusion + MLP head

All models expose a sklearn-compatible interface:
  .fit(X, y)  → train
  .predict_proba(X) → (n, 2) probability array

Author: Juharasha Shaik (shaik.juharasha@ieee.org)
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.class_weight import compute_class_weight


# ─────────────────────────────────────────────────────────────────────────────
# Utility: Generic PyTorch trainer with sklearn interface
# ─────────────────────────────────────────────────────────────────────────────
class _TorchWrapper(BaseEstimator, ClassifierMixin):
    """Base wrapper that handles training loop, class weighting, and predict."""

    def __init__(self, lr=1e-3, epochs=50, batch_size=512, patience=7,
                 device=None, random_state=42):
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.random_state = random_state
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

    def _build_net(self, in_features):
        raise NotImplementedError

    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long)

        # Compute class weights for imbalanced data
        classes = np.unique(y)
        cw = compute_class_weight("balanced", classes=classes, y=y)
        weight = torch.tensor(cw, dtype=torch.float32).to(self.device)

        self.net_ = self._build_net(X.shape[1]).to(self.device)
        optimizer = torch.optim.AdamW(self.net_.parameters(), lr=self.lr,
                                      weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3
        )
        criterion = nn.CrossEntropyLoss(weight=weight)

        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True,
                            drop_last=False)

        best_loss, wait = float("inf"), 0
        best_state = None

        self.net_.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                out = self.net_(xb)
                loss = criterion(out, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net_.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item() * len(xb)
            epoch_loss /= len(dataset)
            scheduler.step(epoch_loss)

            if epoch_loss < best_loss - 1e-5:
                best_loss = epoch_loss
                best_state = {k: v.cpu().clone() for k, v in self.net_.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= self.patience:
                    break

        if best_state is not None:
            self.net_.load_state_dict(best_state)
        self.net_.eval()
        self.classes_ = classes
        return self

    def predict_proba(self, X):
        self.net_.eval()
        X_t = torch.tensor(X, dtype=torch.float32)
        dataset = TensorDataset(X_t)
        loader = DataLoader(dataset, batch_size=self.batch_size * 2, shuffle=False)
        probs = []
        with torch.no_grad():
            for (xb,) in loader:
                xb = xb.to(self.device)
                out = self.net_(xb)
                probs.append(torch.softmax(out, dim=1).cpu().numpy())
        return np.vstack(probs)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# Model 1: MLP Baseline
# ─────────────────────────────────────────────────────────────────────────────
class _MLPNet(nn.Module):
    def __init__(self, in_features, hidden_dims=(256, 128, 64), dropout=0.3):
        super().__init__()
        layers = []
        prev = in_features
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.GELU(),
                       nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MLPClassifier(_TorchWrapper):
    """3-layer MLP with BatchNorm + GELU + Dropout. Sklearn-compatible."""

    def __init__(self, hidden_dims=(256, 128, 64), dropout=0.3, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dims = hidden_dims
        self.dropout = dropout

    def _build_net(self, in_features):
        return _MLPNet(in_features, self.hidden_dims, self.dropout)


# ─────────────────────────────────────────────────────────────────────────────
# Model 2: TabTransformer-style Transformer Baseline
# ─────────────────────────────────────────────────────────────────────────────
class _TransformerNet(nn.Module):
    """Treats each feature as a token, applies self-attention, then classifies."""

    def __init__(self, in_features, d_model=128, nhead=4, num_layers=2,
                 dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)  # project each scalar feature
        self.pos_emb = nn.Parameter(torch.randn(1, in_features, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        # x: (batch, features) → (batch, features, 1) → (batch, features, d_model)
        tokens = self.input_proj(x.unsqueeze(-1)) + self.pos_emb
        cls = self.cls_token.expand(x.size(0), -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        encoded = self.encoder(tokens)
        cls_out = encoded[:, 0, :]
        return self.head(cls_out)


class TransformerClassifier(_TorchWrapper):
    """TabTransformer-style classifier. Each feature becomes a token with
    learned position embeddings; self-attention captures feature interactions."""

    def __init__(self, d_model=128, nhead=4, num_layers=2, dropout=0.2,
                 **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dropout_rate = dropout

    def _build_net(self, in_features):
        return _TransformerNet(in_features, self.d_model, self.nhead,
                               self.num_layers, self.dropout_rate)


# ─────────────────────────────────────────────────────────────────────────────
# Model 3: FraudBERT + MLP (Learnable Projection + Fusion)
# ─────────────────────────────────────────────────────────────────────────────
class _FraudBERTMLPNet(nn.Module):
    """
    Separate processing paths for numerical and categorical (pre-encoded)
    features, with a learnable projection for the categorical embeddings
    and a fusion MLP head.
    """

    def __init__(self, num_dim, cat_dim, proj_dim=128, hidden=(256, 128),
                 dropout=0.3):
        super().__init__()
        # Learnable projection for categorical embeddings (replaces PCA)
        self.cat_proj = nn.Sequential(
            nn.Linear(cat_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
        )
        # Numerical encoder
        self.num_enc = nn.Sequential(
            nn.Linear(num_dim, hidden[0]),
            nn.BatchNorm1d(hidden[0]),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        # Fusion head
        fused_dim = hidden[0] + proj_dim
        layers = []
        prev = fused_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.GELU(),
                       nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 2))
        self.fusion_head = nn.Sequential(*layers)

    def forward(self, x):
        # x is [numerical | categorical] concatenated
        x_num = x[:, :self._num_dim]
        x_cat = x[:, self._num_dim:]
        h_num = self.num_enc(x_num)
        h_cat = self.cat_proj(x_cat)
        h_fused = torch.cat([h_num, h_cat], dim=1)
        return self.fusion_head(h_fused)


class FraudBERTMLP(_TorchWrapper):
    """FraudBERT + MLP variant with learnable categorical projection.

    The input X is expected to be [X_num | X_cat_raw] concatenated, where
    X_cat_raw is the RAW (un-PCA'd) sentence transformer embeddings.

    Parameters
    ----------
    num_dim : int
        Number of numerical features (first num_dim columns of X).
    cat_dim : int
        Number of categorical embedding dimensions (remaining columns).
    proj_dim : int
        Target dimension for the learnable categorical projection.
    """

    def __init__(self, num_dim, cat_dim, proj_dim=128,
                 hidden=(256, 128), dropout=0.3, **kwargs):
        super().__init__(**kwargs)
        self.num_dim = num_dim
        self.cat_dim = cat_dim
        self.proj_dim = proj_dim
        self.hidden = hidden
        self.dropout_val = dropout

    def _build_net(self, in_features):
        net = _FraudBERTMLPNet(
            self.num_dim, self.cat_dim, self.proj_dim,
            self.hidden, self.dropout_val
        )
        net._num_dim = self.num_dim
        return net
