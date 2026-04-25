#!/usr/bin/env python3
"""
Upload retrained YOLO weights to GCS so the ML API picks them up on next restart.

Usage:
    python ml/stamp_detector/upload_model.py

Requires:
    - gcloud CLI authenticated (run: gcloud auth application-default login)
    - GCS_MODEL_BUCKET env var (default: obelisk-stamps-models)
    - Trained weights at ml/stamp_detector/weights/stamp_detector.pt
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
MODULE_DIR = Path(__file__).parent
WEIGHTS_PATH = MODULE_DIR / "weights" / "stamp_detector.pt"
GCS_BUCKET = os.getenv("GCS_MODEL_BUCKET", "obelisk-stamps-models")
GCS_DEST = f"gs://{GCS_BUCKET}/stamp_detector.pt"


def main():
    if not WEIGHTS_PATH.exists():
        print(f"ERROR: Weights not found at {WEIGHTS_PATH}")
        print("Train the model first:")
        print("  python ml/stamp_detector/train.py")
        sys.exit(1)

    size_mb = WEIGHTS_PATH.stat().st_size / 1024 / 1024
    print(f"Found weights: {WEIGHTS_PATH} ({size_mb:.1f} MB)")
    print(f"Uploading to: {GCS_DEST}")

    ret = os.system(f'gsutil cp "{WEIGHTS_PATH}" {GCS_DEST}')
    if ret != 0:
        print("\nERROR: Upload failed.")
        print("Make sure gcloud is authenticated:")
        print("  gcloud auth application-default login")
        sys.exit(1)

    print(f"\nDone! Model uploaded to {GCS_DEST}")
    print("\nNext: trigger ML API redeploy so it downloads the new weights.")
    print("Either push a change to ml_api/ (triggers GitHub Actions),")
    print("or manually restart the Cloud Run service:")
    print("  gcloud run services update obelisk-stamps-ml-api --region europe-west1 --no-traffic")
    print("  gcloud run services update-traffic obelisk-stamps-ml-api --region europe-west1 --to-latest")


if __name__ == "__main__":
    main()
