#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Paper 1 — FraudBERT: Dataset Setup Script
# Author: Juharasha Shaik (shaik.juharasha@ieee.org)
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VEHICLE_DIR="$ROOT_DIR/data/vehicle"
CLAIMS_DIR="$ROOT_DIR/data/claims"

echo "=== FraudBERT Dataset Setup ==="

# Step 1: Get Kaggle API credentials
# 1. Go to https://www.kaggle.com → Account → API → Create New Token
# 2. This downloads kaggle.json — move it here:
mkdir -p ~/.kaggle
# cp /path/to/your/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json 2>/dev/null

if ! command -v kaggle >/dev/null 2>&1; then
  echo "Error: kaggle CLI not found in PATH."
  echo "Install it with: pip install kaggle  (or use your virtualenv pip)"
  exit 1
fi

# Step 2: Download Vehicle Insurance Fraud dataset
echo "Downloading Vehicle Claim Fraud Detection Dataset..."
mkdir -p "$VEHICLE_DIR"
kaggle datasets download -d shivamb/vehicle-claim-fraud-detection -p "$VEHICLE_DIR"
unzip -o -q "$VEHICLE_DIR/vehicle-claim-fraud-detection.zip" -d "$VEHICLE_DIR"
echo "Vehicle dataset ready at data/vehicle/"

# Step 3: Download Insurance Claims Fraud dataset
echo "Downloading Insurance Claims Fraud Dataset..."
mkdir -p "$CLAIMS_DIR"
kaggle datasets download -d mastmustu/insurance-claims-fraud-data -p "$CLAIMS_DIR"
unzip -o -q "$CLAIMS_DIR/insurance-claims-fraud-data.zip" -d "$CLAIMS_DIR"
echo "Claims dataset ready at data/claims/"

echo ""
echo "=== Setup Complete ==="
echo "Run: python3 step1_encode.py --dataset vehicle"
echo "     python3 step2_train.py --dataset vehicle"
echo "     python3 step1_encode.py --dataset claims"
echo "     python3 step2_train.py --dataset claims"
