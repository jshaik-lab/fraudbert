"""
Step 2: Train all models on pre-computed features and generate results tables.
Requires step1_encode.py to have been run first.

Models trained:
  - Logistic Regression (baseline)
  - Random Forest (baseline)
  - XGBoost (baseline)
  - LightGBM (baseline)
  - MLP 3-layer (deep learning baseline)
  - Transformer/TabTransformer-style (deep learning baseline)
  - FraudBERT + XGBoost (proposed)
  - FraudBERT + LightGBM (proposed)
  - FraudBERT + MLP (proposed — learnable fusion)

Usage:
  python3 step2_train.py --dataset synth    # Synthetic smoke test
  python3 step2_train.py --dataset vehicle  # Vehicle Claim Fraud Detection
  python3 step2_train.py --dataset claims   # Insurance Claims Fraud

Author: Juharasha Shaik (shaik.juharasha@ieee.org)
"""

import argparse
import json
import time
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    average_precision_score, roc_auc_score, f1_score,
    precision_recall_curve,
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from scipy import stats

# Local deep learning models
from models import MLPClassifier, TransformerClassifier, FraudBERTMLP

warnings.filterwarnings("ignore")

BASE_DIR    = Path(__file__).parent.parent
CACHE_DIR   = BASE_DIR / "data" / "embeddings"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────────────────────────────────────
def load_features(dataset):
    def _load(name):
        p = CACHE_DIR / f"{dataset}_{name}.npy"
        if not p.exists():
            raise FileNotFoundError(f"Missing {p}. Run step1_encode.py first.")
        return np.load(p)

    X_num   = _load("X_num")
    X_fused = _load("X_fused")
    y       = _load("y")

    # Try to load raw (un-PCA'd) categorical embeddings for FraudBERT+MLP
    X_cat_raw = None
    cat_raw_path = CACHE_DIR / f"{dataset}_X_cat_raw.npy"
    if cat_raw_path.exists():
        X_cat_raw = np.load(cat_raw_path)
        print(f"  Raw categorical embeddings loaded: {X_cat_raw.shape}")

    print(f"Loaded [{dataset}]: X_num={X_num.shape} X_fused={X_fused.shape} y={y.shape}")
    print(f"  Fraud rate: {y.mean():.3%} | Imbalance ratio: {(y==0).sum()/(y==1).sum():.1f}:1")
    return X_num, X_fused, y, X_cat_raw


# ─────────────────────────────────────────────────────────────────────────────
# Cross-Validation with Proper Evaluation
# ─────────────────────────────────────────────────────────────────────────────
def cross_validate(name, model, X, y, n_splits=5):
    skf  = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    rows = []
    t0 = time.time()

    for fold, (tr, val) in enumerate(skf.split(X, y), 1):
        X_tr, X_val = X[tr], X[val]
        y_tr, y_val = y[tr], y[val]

        # Apply SMOTE only for non-deep-learning models (DL uses class weights)
        is_dl = isinstance(model, (MLPClassifier, TransformerClassifier, FraudBERTMLP))
        if not is_dl:
            try:
                sm = SMOTE(random_state=42, k_neighbors=min(5, (y_tr==1).sum()-1))
                X_tr, y_tr = sm.fit_resample(X_tr, y_tr)
            except Exception:
                pass

        model.fit(X_tr, y_tr)
        prob = model.predict_proba(X_val)[:, 1]

        # Threshold-optimized F1 (as stated in Section 5.2)
        prec, rec, thresholds = precision_recall_curve(y_val, prob)
        f1_vals = 2 * prec * rec / (prec + rec + 1e-8)
        best_idx = np.argmax(f1_vals)
        best_thr = thresholds[min(best_idx, len(thresholds) - 1)]
        pred = (prob >= best_thr).astype(int)

        # Recall at 95% precision
        p95_mask = prec >= 0.95
        recall_at_p95 = rec[p95_mask].max() if p95_mask.any() else 0.0

        rows.append({
            "auprc":        average_precision_score(y_val, prob),
            "auc":          roc_auc_score(y_val, prob),
            "f1":           f1_score(y_val, pred, zero_division=0),
            "recall_at_p95": float(recall_at_p95),
            "best_thr":     float(best_thr),
        })
        print(f"  [{name}] fold {fold}: AUPRC={rows[-1]['auprc']:.4f}  "
              f"AUC={rows[-1]['auc']:.4f}  F1={rows[-1]['f1']:.4f}  "
              f"R@P95={rows[-1]['recall_at_p95']:.4f}")

    elapsed = time.time() - t0
    a = np.array([[r["auprc"], r["auc"], r["f1"], r["recall_at_p95"]] for r in rows])
    res = {
        "model":        name,
        "AUPRC":        f"{a[:,0].mean():.4f}±{a[:,0].std():.4f}",
        "AUC":          f"{a[:,1].mean():.4f}±{a[:,1].std():.4f}",
        "F1":           f"{a[:,2].mean():.4f}±{a[:,2].std():.4f}",
        "Recall@P95":   f"{a[:,3].mean():.4f}±{a[:,3].std():.4f}",
        "_auprc_mean":  float(a[:,0].mean()),
        "_auprc_std":   float(a[:,0].std()),
        "_fold_auprcs": a[:,0].tolist(),
        "_time_sec":    round(elapsed, 1),
    }
    print(f"  ► {name}: AUPRC={res['AUPRC']}  AUC={res['AUC']}  "
          f"F1={res['F1']}  R@P95={res['Recall@P95']}  "
          f"({elapsed:.1f}s)\n")
    return res


# ─────────────────────────────────────────────────────────────────────────────
# Model Definitions
# ─────────────────────────────────────────────────────────────────────────────
def build_models(imbalance_ratio, num_dim=None, cat_raw_dim=None):
    """Build all models. Returns dict of {name: (model, feature_type)}."""
    models = {
        # ── Classical Baselines (numerical features only) ──
        "Logistic Regression": (
            LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", n_jobs=-1),
            "num"
        ),
        "Random Forest": (
            RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                   n_jobs=-1, random_state=42),
            "num"
        ),
        "XGBoost": (
            XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05,
                          scale_pos_weight=imbalance_ratio,
                          eval_metric="aucpr", random_state=42, n_jobs=-1, verbosity=0),
            "num"
        ),
        "LightGBM": (
            LGBMClassifier(n_estimators=500, num_leaves=63, learning_rate=0.05,
                           is_unbalance=True, random_state=42, n_jobs=-1, verbosity=-1),
            "num"
        ),

        # ── Deep Learning Baselines (numerical features only) ──
        "MLP (3-layer)": (
            MLPClassifier(hidden_dims=(256, 128, 64), dropout=0.3,
                          epochs=50, batch_size=512, lr=1e-3),
            "num"
        ),
        "Transformer": (
            TransformerClassifier(d_model=128, nhead=4, num_layers=2, dropout=0.2,
                                  epochs=50, batch_size=512, lr=1e-3),
            "num"
        ),

        # ── FraudBERT Proposed Models (fused features) ──
        "FraudBERT + XGBoost (ours)": (
            XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05,
                          scale_pos_weight=imbalance_ratio,
                          eval_metric="aucpr", random_state=42, n_jobs=-1, verbosity=0),
            "fused"
        ),
        "FraudBERT + LightGBM (ours)": (
            LGBMClassifier(n_estimators=500, num_leaves=63, learning_rate=0.05,
                           is_unbalance=True, random_state=42, n_jobs=-1, verbosity=-1),
            "fused"
        ),
    }

    # FraudBERT + MLP (learnable fusion) — only if raw cat embeddings available
    if num_dim is not None and cat_raw_dim is not None and cat_raw_dim > 0:
        models["FraudBERT + MLP (ours)"] = (
            FraudBERTMLP(num_dim=num_dim, cat_dim=cat_raw_dim, proj_dim=128,
                         epochs=50, batch_size=512, lr=1e-3),
            "fused_raw"  # needs X_num + X_cat_raw (not PCA'd)
        )

    return models


# ─────────────────────────────────────────────────────────────────────────────
# Formatting & Output
# ─────────────────────────────────────────────────────────────────────────────
def print_latex_table(results, caption, label):
    print(f"\n{'='*70}")
    print(f"LaTeX Table: {caption}")
    print(f"{'='*70}")
    print(r"\begin{table}[h!]")
    print(r"\centering")
    print(r"\caption{" + caption + r"}\label{" + label + r"}")
    print(r"\begin{tabular}{lcccc}")
    print(r"\hline")
    print(r"\textbf{Model} & \textbf{AUPRC} & \textbf{AUC-ROC} "
          r"& \textbf{F1-Score} & \textbf{R@P95} \\")
    print(r"\hline")
    for r in sorted(results, key=lambda x: x["_auprc_mean"], reverse=True):
        bold = "ours" in r["model"]
        s, e = (r"\textbf{", "}") if bold else ("", "")
        print(f"{s}{r['model']}{e} & {s}{r['AUPRC']}{e} & "
              f"{s}{r['AUC']}{e} & {s}{r['F1']}{e} & "
              f"{s}{r['Recall@P95']}{e} \\\\")
    print(r"\hline")
    print(r"\end{tabular}")
    print(r"\end{table}")


def print_timing_table(results):
    print(f"\n{'='*50}")
    print("Computational Cost (Training Time)")
    print(f"{'='*50}")
    for r in sorted(results, key=lambda x: x["_time_sec"]):
        print(f"  {r['model']:40s} {r['_time_sec']:8.1f}s")


def statistical_tests(results):
    """Pairwise Wilcoxon signed-rank tests between top model and baselines."""
    sorted_r = sorted(results, key=lambda x: x["_auprc_mean"], reverse=True)
    best = sorted_r[0]
    print(f"\n{'='*60}")
    print(f"Statistical Significance Tests (vs {best['model']})")
    print(f"{'='*60}")

    for other in sorted_r[1:]:
        if len(best["_fold_auprcs"]) >= 5:
            try:
                stat, p_val = stats.wilcoxon(
                    best["_fold_auprcs"], other["_fold_auprcs"]
                )
                sig = "✓ p<0.05" if p_val < 0.05 else "✗ not significant"
                print(f"  vs {other['model']:40s} W={stat:.2f}  p={p_val:.4f}  {sig}")
            except Exception as e:
                print(f"  vs {other['model']:40s} — test failed: {e}")


def plot_bar_chart(results, dataset_name):
    df_sorted = sorted(results, key=lambda r: r["_auprc_mean"])
    names  = [r["model"].replace(" (ours)", " ★") for r in df_sorted]
    auprcs = [r["_auprc_mean"] for r in df_sorted]
    stds   = [r["_auprc_std"] for r in df_sorted]
    colors = ["#EF5350" if "ours" in r["model"] or "FraudBERT" in r["model"]
              else "#42A5F5" for r in df_sorted]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(names, auprcs, xerr=stds, color=colors, alpha=0.88,
                   edgecolor="white", capsize=3)
    ax.set_xlabel("AUPRC (higher is better)", fontsize=11)
    ax.set_title(f"FraudBERT vs Baselines — {dataset_name}",
                 fontsize=13, fontweight="bold")
    ax.set_xlim(0, min(1.0, max(auprcs) * 1.15))
    for bar, val in zip(bars, auprcs):
        ax.text(val + 0.003, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", fontsize=9)

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color="#EF5350", label="FraudBERT (proposed)"),
        Patch(color="#42A5F5", label="Baseline"),
    ], loc="lower right")

    plt.tight_layout()
    out = RESULTS_DIR / f"auprc_{dataset_name.lower().replace(' ','_')}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Chart saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["synth","vehicle","claims"], default="synth")
    parser.add_argument("--folds",   type=int, default=5)
    args = parser.parse_args()

    X_num, X_fused, y, X_cat_raw = load_features(args.dataset)
    ir = float((y == 0).sum()) / float((y == 1).sum())

    # Build feature matrix for FraudBERT+MLP (raw cat if available)
    cat_raw_dim = X_cat_raw.shape[1] if X_cat_raw is not None else 0
    X_fused_raw = None
    if X_cat_raw is not None:
        X_fused_raw = np.hstack([X_num, X_cat_raw])

    models = build_models(ir, num_dim=X_num.shape[1], cat_raw_dim=cat_raw_dim)

    results = []
    for name, (model, feat_type) in models.items():
        print(f"\n{'─'*60}")
        print(f"Model: {name}  |  Features: {feat_type}")

        if feat_type == "fused_raw":
            if X_fused_raw is None:
                print(f"  ⚠ Skipping {name} — no raw categorical embeddings available")
                continue
            X = X_fused_raw
        elif feat_type == "fused":
            X = X_fused
        else:
            X = X_num

        r = cross_validate(name, model, X, y, n_splits=args.folds)
        results.append(r)

    # Save JSON
    out_json = RESULTS_DIR / f"{args.dataset}_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {out_json}")

    # Print LaTeX table
    labels = {"synth": "tab:synth", "vehicle": "tab:vehicle", "claims": "tab:claims"}
    captions = {
        "synth":   "Model Comparison on Synthetic Data (Validation)",
        "vehicle": "Model Comparison on Vehicle Claim Fraud Detection Dataset",
        "claims":  "Model Comparison on Insurance Claims Fraud Dataset",
    }
    print_latex_table(results, captions[args.dataset], labels[args.dataset])

    # Statistical tests
    statistical_tests(results)

    # Timing
    print_timing_table(results)

    # Plot
    plot_bar_chart(results, args.dataset.upper())
    print(f"\n✓ All done! Results: {out_json}")
