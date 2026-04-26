"""
Step 1: Generate LLM embeddings for categorical features.
Run this FIRST. Saves embeddings to disk so step2_train.py can load them.

Usage:
  python3 step1_encode.py --dataset vehicle  # Vehicle Claim Fraud Detection (Kaggle: shivamb)
  python3 step1_encode.py --dataset claims   # Insurance Claims Fraud (Kaggle: mastmustu)
  python3 step1_encode.py --dataset synth    # Synthetic test data (no download needed)

Author: Juharasha Shaik (shaik.juharasha@ieee.org)
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler

BASE_DIR  = Path(__file__).parent.parent
DATA_DIR  = BASE_DIR / "data"
CACHE_DIR = BASE_DIR / "data" / "embeddings"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ── Vehicle Insurance Fraud Dataset (Kaggle: shivamb/vehicle-claim-fraud-detection)
VEHICLE_CAT = [
    "Make", "AccidentArea", "Sex", "MaritalStatus", "Fault",
    "PolicyType", "VehicleCategory", "VehiclePrice", "Days_Policy_Accident",
    "Days_Policy_Claim", "PastNumberOfClaims", "AgeOfVehicle",
    "AgeOfPolicyHolder", "PoliceReportFiled", "WitnessPresent",
    "AgentType", "NumberOfSuppliments", "AddressChange_Claim",
    "NumberOfCars", "BasePolicy"
]
VEHICLE_NUM = [
    "Month", "WeekOfMonth", "DayOfWeek", "DayOfWeekClaimed",
    "MonthClaimed", "WeekOfMonthClaimed", "Age", "Deductible",
    "DriverRating", "Year", "RepNumber"
]

# ── Insurance Claims Fraud Dataset (Kaggle: mastmustu/insurance-claims-fraud-data)
CLAIMS_CAT = [
    "policy_csl", "insured_sex", "insured_education_level", "insured_occupation",
    "insured_hobbies", "insured_relationship", "incident_type", "collision_type",
    "incident_severity", "authorities_contacted", "incident_state", "incident_city",
    "property_damage", "bodily_injuries", "witnesses",
    "police_report_available", "auto_make", "auto_model"
]
CLAIMS_NUM = [
    "months_as_customer", "age", "policy_deductable", "policy_annual_premium",
    "umbrella_limit", "capital-gains", "capital-loss", "incident_hour_of_the_day",
    "number_of_vehicles_involved", "injury_claim", "property_claim",
    "vehicle_claim", "total_claim_amount"
]

# Default to vehicle insurance dataset
CAT_FEATURES = VEHICLE_CAT
NUM_FEATURES  = VEHICLE_NUM


def make_synthetic(n=5000, seed=42):
    np.random.seed(seed)
    y = (np.random.rand(n) < 0.035).astype(int)
    data = {
        "isFraud": y,
        "TransactionAmt": np.random.lognormal(4, 1.5, n),
        "ProductCD":   np.random.choice(["W","C","R","H","S"], n),
        "card4":       np.random.choice(["visa","mastercard","discover","amex"], n),
        "card6":       np.random.choice(["debit","credit"], n),
        "P_emaildomain": np.random.choice(["gmail.com","yahoo.com","hotmail.com","anonymous.com"], n),
        "R_emaildomain": np.random.choice(["gmail.com","yahoo.com","unknown"], n),
        "DeviceType":  np.random.choice(["desktop","mobile","unknown"], n),
        "DeviceInfo":  np.random.choice(["Windows","iOS Device","MacOS","Android"], n),
        "id_12": np.random.choice(["Found","NotFound"], n),
        "id_15": np.random.choice(["New","Found","Unknown"], n),
        "id_16": np.random.choice(["Found","NotFound"], n),
        "id_28": np.random.choice(["New","Found"], n),
        "id_29": np.random.choice(["Found","NotFound"], n),
        "id_31": np.random.choice(["chrome 65","firefox 60","safari 11"], n),
        "id_35": np.random.choice(["T","F"], n),
        "id_36": np.random.choice(["T","F"], n),
        "id_37": np.random.choice(["T","F"], n),
        "id_38": np.random.choice(["T","F"], n),
    }
    for i in range(1, 4):   data[f"card{i}"] = np.random.randint(1000, 9999, n).astype(float)
    for f in ["addr1","addr2","dist1","dist2"]: data[f] = np.random.rand(n) * 100
    for i in range(1, 15):  data[f"C{i}"] = np.random.exponential(2, n)
    for i in range(1, 16):  data[f"D{i}"] = np.random.rand(n) * 30
    for i in range(1, 100): data[f"V{i}"] = np.random.randn(n)
    return pd.DataFrame(data)


def load_vehicle():
    """Vehicle Claim Fraud Detection — Kaggle: shivamb/vehicle-claim-fraud-detection"""
    df = pd.read_csv(DATA_DIR / "vehicle" / "fraud_oracle.csv")
    df = df.rename(columns={"FraudFound_P": "isFraud"})
    print(f"Loaded Vehicle Insurance: {len(df):,} rows | fraud={df['isFraud'].mean():.3%}")
    return df, VEHICLE_CAT, VEHICLE_NUM


def load_claims():
    """Insurance Claims Fraud — Kaggle: mastmustu/insurance-claims-fraud-data"""
    df = pd.read_csv(DATA_DIR / "claims" / "insurance_claims.csv")
    df["isFraud"] = (df["fraud_reported"] == "Y").astype(int)
    print(f"Loaded Insurance Claims: {len(df):,} rows | fraud={df['isFraud'].mean():.3%}")
    return df, CLAIMS_CAT, CLAIMS_NUM


def encode_and_save(df, dataset_name, proj_dim=128):
    TARGET = "isFraud"
    y = df[TARGET].values

    # ── Numerical ────────────────────────────────────────────────
    avail_num = [f for f in NUM_FEATURES if f in df.columns]
    X_num = RobustScaler().fit_transform(df[avail_num].fillna(-999))
    np.save(CACHE_DIR / f"{dataset_name}_X_num.npy",  X_num)
    np.save(CACHE_DIR / f"{dataset_name}_y.npy",      y)
    print(f"Numerical features: {X_num.shape}")

    # ── LLM Categorical ──────────────────────────────────────────
    avail_cat = [f for f in CAT_FEATURES if f in df.columns]
    if not avail_cat:
        X_fused = X_num
        np.save(CACHE_DIR / f"{dataset_name}_X_fused.npy", X_fused)
        print("No categorical features — fused = numerical only")
        return

    print(f"\nBuilding prompt list for {len(avail_cat)} categorical features...")
    all_prompts, val_index = [], {}
    for feat in avail_cat:
        vals = df[feat].fillna("unknown").astype(str)
        for v in vals.unique():
            if (feat, v) not in val_index:
                val_index[(feat, v)] = len(all_prompts)
                all_prompts.append(f"Transaction feature {feat}: {v}")
    print(f"Total unique (feature, value) prompts: {len(all_prompts)}")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("Encoding with LLM (single batch call)...")
    all_embs = model.encode(all_prompts, batch_size=128, show_progress_bar=True)
    print(f"Embeddings shape: {all_embs.shape}")

    emb_parts = []
    for feat in avail_cat:
        vals = df[feat].fillna("unknown").astype(str).tolist()
        col_emb = np.vstack([all_embs[val_index[(feat, v)]] for v in vals])
        emb_parts.append(col_emb)

    raw_cat = np.hstack(emb_parts)
    print(f"Raw categorical embeddings: {raw_cat.shape}")

    n_comp = min(proj_dim, raw_cat.shape[1], raw_cat.shape[0] - 1)
    pca = PCA(n_components=n_comp, random_state=42)
    X_cat_proj = pca.fit_transform(raw_cat)
    explained = pca.explained_variance_ratio_.sum()
    print(f"PCA: {raw_cat.shape[1]}d → {X_cat_proj.shape[1]}d ({explained:.1%} variance)")

    X_fused = np.hstack([X_num, X_cat_proj])
    np.save(CACHE_DIR / f"{dataset_name}_X_fused.npy", X_fused)
    np.save(CACHE_DIR / f"{dataset_name}_X_cat.npy",   X_cat_proj)
    np.save(CACHE_DIR / f"{dataset_name}_X_cat_raw.npy", raw_cat)  # for learnable projection
    print(f"\nSaved:")
    print(f"  {dataset_name}_X_num.npy     → {X_num.shape}")
    print(f"  {dataset_name}_X_cat.npy     → {X_cat_proj.shape}")
    print(f"  {dataset_name}_X_cat_raw.npy → {raw_cat.shape} (pre-PCA, for FraudBERT+MLP)")
    print(f"  {dataset_name}_X_fused.npy   → {X_fused.shape}")
    print(f"  {dataset_name}_y.npy         → {y.shape}")
    print(f"\nRun next: python3 step2_train.py --dataset {dataset_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["vehicle", "claims", "synth"], default="synth")
    parser.add_argument("--proj_dim", type=int, default=128)
    args = parser.parse_args()

    if args.dataset == "synth":
        df = make_synthetic()
        cat_feats, num_feats = VEHICLE_CAT, VEHICLE_NUM
    elif args.dataset == "vehicle":
        df, cat_feats, num_feats = load_vehicle()
    else:
        df, cat_feats, num_feats = load_claims()

    encode_and_save(df, args.dataset, proj_dim=args.proj_dim)
