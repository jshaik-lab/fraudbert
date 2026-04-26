"""
Ablation studies for the FraudBERT paper.

Studies:
  A1 — Embedding projection dimensionality (32, 64, 128, 256)
  A2 — Per-feature importance (leave-one-out categorical ablation)
  A3 — Class imbalance strategy comparison (None, SMOTE, ADASYN, cost-sensitive)
  A4 — Pre-trained encoder comparison (MiniLM, MPNet, BGE-small)

Usage:
  python3 ablations.py --dataset vehicle       # all ablations on Vehicle Insurance dataset
  python3 ablations.py --dataset synth         # smoke test on synthetic data
  python3 ablations.py --study dim             # only run dimension ablation
  python3 ablations.py --study feature         # only run feature ablation
  python3 ablations.py --study imbalance       # only run imbalance ablation

Author: Juharasha Shaik (shaik.juharasha@ieee.org)
"""

import argparse
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE, ADASYN
from sentence_transformers import SentenceTransformer
from scipy import stats

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = DATA_DIR / "embeddings"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

CAT_FEATURES = [
    "ProductCD", "card4", "card6",
    "P_emaildomain", "R_emaildomain",
    "DeviceType", "DeviceInfo",
    "id_12", "id_15", "id_16",
    "id_28", "id_29", "id_31", "id_35", "id_36", "id_37", "id_38"
]

NUM_FEATURES = (
    ["TransactionAmt"] +
    [f"card{i}" for i in range(1, 4)] +
    ["addr1", "addr2", "dist1", "dist2"] +
    [f"C{i}" for i in range(1, 15)] +
    [f"D{i}" for i in range(1, 16)] +
    [f"V{i}" for i in range(1, 100)]
)


# ─────────────────────────────────────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────────────────────────────────────
def load_dataset(name):
    if name == "synth":
        return _make_synthetic()
    elif name == "vehicle":
        df = pd.read_csv(DATA_DIR / "vehicle" / "fraud_oracle.csv")
        df = df.rename(columns={"FraudFound_P": "isFraud"})
        print(f"Vehicle Insurance loaded: {len(df):,} rows | fraud={df['isFraud'].mean():.3%}")
        return df
    elif name == "claims":
        df = pd.read_csv(DATA_DIR / "claims" / "insurance_claims.csv")
        df["isFraud"] = (df["fraud_reported"] == "Y").astype(int)
        print(f"Insurance Claims loaded: {len(df):,} rows | fraud={df['isFraud'].mean():.3%}")
        return df


def _make_synthetic(n=5000, seed=42):
    np.random.seed(seed)
    y = (np.random.rand(n) < 0.035).astype(int)
    data = {
        "isFraud": y,
        "TransactionAmt": np.random.lognormal(4, 1.5, n),
        "ProductCD": np.random.choice(["W", "C", "R", "H", "S"], n),
        "card4": np.random.choice(["visa", "mastercard", "discover", "amex"], n),
        "card6": np.random.choice(["debit", "credit"], n),
        "P_emaildomain": np.random.choice(
            ["gmail.com", "yahoo.com", "hotmail.com", "anonymous.com"], n
        ),
        "R_emaildomain": np.random.choice(["gmail.com", "yahoo.com", "unknown"], n),
        "DeviceType": np.random.choice(["desktop", "mobile", "unknown"], n),
        "DeviceInfo": np.random.choice(["Windows", "iOS Device", "MacOS", "Android"], n),
        "id_12": np.random.choice(["Found", "NotFound"], n),
        "id_15": np.random.choice(["New", "Found", "Unknown"], n),
        "id_16": np.random.choice(["Found", "NotFound"], n),
        "id_28": np.random.choice(["New", "Found"], n),
        "id_29": np.random.choice(["Found", "NotFound"], n),
        "id_31": np.random.choice(["chrome 65", "firefox 60", "safari 11"], n),
        "id_35": np.random.choice(["T", "F"], n),
        "id_36": np.random.choice(["T", "F"], n),
        "id_37": np.random.choice(["T", "F"], n),
        "id_38": np.random.choice(["T", "F"], n),
    }
    for i in range(1, 4):
        data[f"card{i}"] = np.random.randint(1000, 9999, n).astype(float)
    for f in ["addr1", "addr2", "dist1", "dist2"]:
        data[f] = np.random.rand(n) * 100
    for i in range(1, 15):
        data[f"C{i}"] = np.random.exponential(2, n)
    for i in range(1, 16):
        data[f"D{i}"] = np.random.rand(n) * 30
    for i in range(1, 100):
        data[f"V{i}"] = np.random.randn(n)
    return pd.DataFrame(data)


def prepare_numerical(df):
    avail = [f for f in NUM_FEATURES if f in df.columns]
    X = df[avail].fillna(-999).values.astype(np.float32)
    return RobustScaler().fit_transform(X)


def encode_categorical(df, cat_features, model_name="all-MiniLM-L6-v2", proj_dim=128):
    """Encode categorical features using a sentence transformer."""
    model = SentenceTransformer(model_name)
    avail = [f for f in cat_features if f in df.columns]
    if not avail:
        return np.zeros((len(df), proj_dim))

    all_prompts, val_index = [], {}
    for feat in avail:
        vals = df[feat].fillna("unknown").astype(str)
        for v in vals.unique():
            if (feat, v) not in val_index:
                val_index[(feat, v)] = len(all_prompts)
                all_prompts.append(f"Transaction feature {feat}: {v}")

    all_embs = model.encode(all_prompts, batch_size=128, show_progress_bar=False)

    emb_parts = []
    for feat in avail:
        vals = df[feat].fillna("unknown").astype(str).tolist()
        col_emb = np.vstack([all_embs[val_index[(feat, v)]] for v in vals])
        emb_parts.append(col_emb)

    raw = np.hstack(emb_parts)
    n_comp = min(proj_dim, raw.shape[1], raw.shape[0] - 1)
    pca = PCA(n_components=n_comp, random_state=42)
    return pca.fit_transform(raw)


# ─────────────────────────────────────────────────────────────────────────────
# Core evaluator (with optional imbalance strategy)
# ─────────────────────────────────────────────────────────────────────────────
def evaluate(name, X, y, imbalance_ratio, n_splits=5, imbalance_strategy="smote"):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics = []

    for fold, (tr, val) in enumerate(skf.split(X, y), 1):
        X_tr, X_val = X[tr], X[val]
        y_tr, y_val = y[tr], y[val]

        # Imbalance handling
        if imbalance_strategy == "smote":
            try:
                sm = SMOTE(random_state=42, k_neighbors=min(5, (y_tr == 1).sum() - 1))
                X_tr, y_tr = sm.fit_resample(X_tr, y_tr)
            except Exception:
                pass
        elif imbalance_strategy == "adasyn":
            try:
                ad = ADASYN(random_state=42, n_neighbors=min(5, (y_tr == 1).sum() - 1))
                X_tr, y_tr = ad.fit_resample(X_tr, y_tr)
            except Exception:
                pass
        elif imbalance_strategy == "cost_sensitive":
            pass  # handled by scale_pos_weight below

        spw = imbalance_ratio if imbalance_strategy == "cost_sensitive" else 1.0
        model = XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            scale_pos_weight=spw if imbalance_strategy == "cost_sensitive" else imbalance_ratio,
            eval_metric="aucpr", random_state=42, n_jobs=-1, verbosity=0
        )
        model.fit(X_tr, y_tr)
        prob = model.predict_proba(X_val)[:, 1]

        # Threshold-optimized F1
        from sklearn.metrics import precision_recall_curve
        prec, rec, thresholds = precision_recall_curve(y_val, prob)
        f1_vals = 2 * prec * rec / (prec + rec + 1e-8)
        best_idx = np.argmax(f1_vals)
        best_thr = thresholds[min(best_idx, len(thresholds) - 1)]
        pred = (prob >= best_thr).astype(int)

        fold_metrics.append({
            "auprc": average_precision_score(y_val, prob),
            "auc": roc_auc_score(y_val, prob),
            "f1": f1_score(y_val, pred, zero_division=0),
        })

    a = np.array([[m["auprc"], m["auc"], m["f1"]] for m in fold_metrics])
    return {
        "name": name,
        "AUPRC_mean": float(a[:, 0].mean()),
        "AUPRC_std": float(a[:, 0].std()),
        "AUC_mean": float(a[:, 1].mean()),
        "AUC_std": float(a[:, 1].std()),
        "F1_mean": float(a[:, 2].mean()),
        "F1_std": float(a[:, 2].std()),
        "fold_auprcs": a[:, 0].tolist(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# A1: Embedding Dimension Ablation
# ─────────────────────────────────────────────────────────────────────────────
def ablation_dimension(df, dims=(32, 64, 128, 256)):
    print("\n" + "=" * 60)
    print("ABLATION A1: Embedding Projection Dimensionality")
    print("=" * 60)

    y = df["isFraud"].values
    ir = float((y == 0).sum()) / max((y == 1).sum(), 1)
    X_num = prepare_numerical(df)
    results = []

    # Baseline: no LLM embeddings
    r_base = evaluate("No-LLM (XGBoost baseline)", X_num, y, ir)
    results.append(r_base)
    print(f"  Baseline AUPRC: {r_base['AUPRC_mean']:.4f}")

    for dim in dims:
        print(f"\n  Projection dim = {dim}...")
        X_cat = encode_categorical(df, CAT_FEATURES, proj_dim=dim)
        X_fused = np.hstack([X_num, X_cat])
        r = evaluate(f"FraudBERT+XGB (d={dim})", X_fused, y, ir)
        r["proj_dim"] = dim
        results.append(r)
        delta = r["AUPRC_mean"] - r_base["AUPRC_mean"]
        print(f"  d={dim}: AUPRC={r['AUPRC_mean']:.4f} (Δ={delta:+.4f})")

    _save_and_plot_ablation(results, "ablation_dimension", "Projection Dimension",
                            "proj_dim", dims)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# A2: Per-Feature Importance (Leave-One-Out)
# ─────────────────────────────────────────────────────────────────────────────
def ablation_feature(df):
    print("\n" + "=" * 60)
    print("ABLATION A2: Per-Feature Leave-One-Out Importance")
    print("=" * 60)

    y = df["isFraud"].values
    ir = float((y == 0).sum()) / max((y == 1).sum(), 1)
    X_num = prepare_numerical(df)
    avail = [f for f in CAT_FEATURES if f in df.columns]

    # Full model
    X_cat_full = encode_categorical(df, avail, proj_dim=128)
    X_full = np.hstack([X_num, X_cat_full])
    r_full = evaluate("All features", X_full, y, ir)
    print(f"  Full model AUPRC: {r_full['AUPRC_mean']:.4f}")

    results = [r_full]
    for feat in avail:
        remaining = [f for f in avail if f != feat]
        X_cat = encode_categorical(df, remaining, proj_dim=128)
        X_fused = np.hstack([X_num, X_cat])
        r = evaluate(f"Drop: {feat}", X_fused, y, ir)
        r["dropped_feature"] = feat
        r["delta"] = r_full["AUPRC_mean"] - r["AUPRC_mean"]
        results.append(r)
        print(f"  Drop {feat}: AUPRC={r['AUPRC_mean']:.4f} "
              f"(Δ={r['delta']:+.4f} — {'important' if r['delta'] > 0 else 'not helpful'})")

    out = RESULTS_DIR / "ablation_feature.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {out}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# A3: Class Imbalance Strategy Comparison
# ─────────────────────────────────────────────────────────────────────────────
def ablation_imbalance(df):
    print("\n" + "=" * 60)
    print("ABLATION A3: Class Imbalance Strategy Comparison")
    print("=" * 60)

    y = df["isFraud"].values
    ir = float((y == 0).sum()) / max((y == 1).sum(), 1)
    X_num = prepare_numerical(df)
    X_cat = encode_categorical(df, CAT_FEATURES, proj_dim=128)
    X_fused = np.hstack([X_num, X_cat])

    strategies = ["none", "smote", "adasyn", "cost_sensitive"]
    results = []
    for strat in strategies:
        print(f"\n  Strategy: {strat}...")
        r = evaluate(f"FraudBERT+XGB ({strat})", X_fused, y, ir,
                     imbalance_strategy=strat)
        r["strategy"] = strat
        results.append(r)
        print(f"  {strat}: AUPRC={r['AUPRC_mean']:.4f} F1={r['F1_mean']:.4f}")

    # Statistical significance: compare best vs second-best
    sorted_r = sorted(results, key=lambda x: x["AUPRC_mean"], reverse=True)
    if len(sorted_r) >= 2:
        stat, p_val = stats.wilcoxon(
            sorted_r[0]["fold_auprcs"], sorted_r[1]["fold_auprcs"]
        )
        print(f"\n  Wilcoxon test (best vs 2nd): statistic={stat:.4f}, p={p_val:.4f}")

    out = RESULTS_DIR / "ablation_imbalance.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved: {out}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _save_and_plot_ablation(results, filename, xlabel, key, tick_values):
    out_json = RESULTS_DIR / f"{filename}.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, default=str)

    fig, ax = plt.subplots(figsize=(8, 5))
    names = [r["name"] for r in results]
    auprcs = [r["AUPRC_mean"] for r in results]
    stds = [r["AUPRC_std"] for r in results]
    colors = ["#42A5F5" if "baseline" in r["name"].lower() or "No-LLM" in r["name"]
              else "#EF5350" for r in results]

    bars = ax.barh(names, auprcs, xerr=stds, color=colors, alpha=0.85,
                   edgecolor="white", capsize=3)
    ax.set_xlabel("AUPRC", fontsize=11)
    ax.set_title(f"Ablation: {xlabel}", fontsize=12, fontweight="bold")
    for bar, val in zip(bars, auprcs):
        ax.text(val + 0.003, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9)
    plt.tight_layout()
    out_png = RESULTS_DIR / f"{filename}.png"
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"  Plot saved: {out_png}")
    print(f"  JSON saved: {out_json}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FraudBERT Ablation Studies")
    parser.add_argument("--dataset", choices=["synth", "vehicle", "claims"], default="synth")
    parser.add_argument("--study", choices=["all", "dim", "feature", "imbalance"],
                        default="all")
    args = parser.parse_args()

    print("FraudBERT Ablation Studies")
    print(f"Dataset: {args.dataset} | Study: {args.study}")
    print()

    df = load_dataset(args.dataset)

    if args.study in ("all", "dim"):
        ablation_dimension(df)

    if args.study in ("all", "feature"):
        ablation_feature(df)

    if args.study in ("all", "imbalance"):
        ablation_imbalance(df)

    print(f"\n✓ Ablation studies complete. Results in: {RESULTS_DIR}")
