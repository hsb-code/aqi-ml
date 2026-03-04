#!/usr/bin/env python
"""
Preprocessing entry-point.

Run the full preprocessing pipeline from the project root:

    conda activate aqi-ml
    python scripts/preprocess.py

Optional overrides:
    python scripts/preprocess.py --ground-station path/to/file.csv
    python scripts/preprocess.py --no2 path/to/no2.csv --aod path/to/aod.csv --era5 path/to/era5.csv
    python scripts/preprocess.py --output-dir data/processed --model-dir models

Outputs:
    data/processed/training_data_full.parquet   — full cleaned dataset
    data/processed/train.parquet                — training split
    data/processed/val.parquet                  — validation split
    data/processed/test.parquet                 — test split (2024)
    models/feature_scaler.pkl                   — fitted RobustScaler
"""

import argparse
import logging
import sys
from pathlib import Path

# ── Make sure project root is on sys.path ─────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.preprocessing.pipeline import run_pipeline

# ── Logging ───────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(ROOT / "logs" / "preprocess.log", mode="w"),
    ],
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="AQI ML Preprocessing Pipeline")
    p.add_argument("--ground-station", type=Path, default=None,
                   help="Path to deduplicated ground station CSV")
    p.add_argument("--no2",  type=Path, default=None, help="S5P NO2 at-stations CSV")
    p.add_argument("--aod",  type=Path, default=None, help="MODIS AOD at-stations CSV")
    p.add_argument("--era5", type=Path, default=None, help="ERA5 at-stations CSV")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Directory to save parquet files (default: data/processed)")
    p.add_argument("--model-dir", type=Path, default=None,
                   help="Directory to save scaler (default: models)")
    return p.parse_args()


def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info("AQI ML — PREPROCESSING PIPELINE")
    logger.info("=" * 60)

    result = run_pipeline(
        ground_station_path=args.ground_station,
        no2_path=args.no2,
        aod_path=args.aod,
        era5_path=args.era5,
        output_dir=args.output_dir,
        model_dir=args.model_dir,
    )

    logger.info("Pipeline completed successfully.")
    return result


if __name__ == "__main__":
    main()
