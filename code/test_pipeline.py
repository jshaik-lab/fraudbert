"""
Pipeline smoke-test using synthetic insurance fraud data.
Run this BEFORE downloading real data to verify the full pipeline works.

Usage:
  python3 test_pipeline.py

Author: Juharasha Shaik (shaik.juharasha@ieee.org)
"""
import subprocess, sys
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent

if __name__ == "__main__":
    print("=" * 60)
    print("SMOKE TEST — Synthetic insurance fraud data")
    print("=" * 60)
    print("Step 1: Encoding...")
    r1 = subprocess.run(
        [sys.executable, "step1_encode.py", "--dataset", "synth"],
        cwd=Path(__file__).parent, capture_output=False
    )
    if r1.returncode != 0:
        print("STEP 1 FAILED"); sys.exit(1)

    print("\nStep 2: Training (2 folds)...")
    r2 = subprocess.run(
        [sys.executable, "step2_train.py", "--dataset", "synth", "--folds", "2"],
        cwd=Path(__file__).parent, capture_output=False
    )
    if r2.returncode != 0:
        print("STEP 2 FAILED"); sys.exit(1)

    print("\n" + "=" * 60)
    print("SMOKE TEST PASSED — Pipeline is working correctly")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Get Kaggle API key → place at ~/.kaggle/kaggle.json")
    print("  2. bash setup_and_download.sh")
    print("  3. python3 step1_encode.py --dataset vehicle")
    print("  4. python3 step2_train.py  --dataset vehicle")
    print("  5. python3 step1_encode.py --dataset claims")
    print("  6. python3 step2_train.py  --dataset claims")
