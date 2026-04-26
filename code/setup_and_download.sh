#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Paper 1 — FraudBERT: Dataset Setup Script
# Author: Juharasha Shaik (shaik.juharasha@ieee.org)
# ─────────────────────────────────────────────────────────────────────────────

echo "=== FraudBERT Dataset Setup ==="

# Step 1: Get Kaggle API credentials
# 1. Go to https://www.kaggle.com → Account → API → Create New Token
# 2. This downloads kaggle.json — move it here:
mkdir -p ~/.kaggle
# cp /path/to/your/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json 2>/dev/null

# Step 2: Download Vehicle Insurance Fraud dataset
echo "Downloading Vehicle Claim Fraud Detection Dataset..."
mkdir -p ../data/vehicle
kaggle datasets download -d shivamb/vehicle-claim-fraud-detection -p ../data/vehicle
cd ../data/vehicle && unzip -q vehicle-claim-fraud-detection.zip && cd -
echo "Vehicle dataset ready at data/vehicle/"

# Step 3: Download Insurance Claims Fraud dataset
echo "Downloading Insurance Claims Fraud Dataset..."
mkdir -p ../data/claims
kaggle datasets download -d mastmustu/insurance-claims-fraud-data -p ../data/claims
cd ../data/claims && unzip -q insurance-claims-fraud-data.zip && cd -
echo "Claims dataset ready at data/claims/"

echo ""
echo "=== Setup Complete ==="
echo "Run: python3 step1_encode.py --dataset vehicle"
echo "     python3 step2_train.py --dataset vehicle"
echo "     python3 step1_encode.py --dataset claims"
echo "     python3 step2_train.py --dataset claims"
