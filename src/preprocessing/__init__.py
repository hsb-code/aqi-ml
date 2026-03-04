"""
Preprocessing package for AQI ML pipeline.

Modules:
    ground_station  — load, filter, aggregate EAD hourly data
    satellite       — S5P NO2 unit conversion, MODIS AOD scaling, QA filters
    era5            — ERA5 unit conversions and derived met variables
    features        — merge all sources, build 21-feature matrix
    pipeline        — top-level orchestrator: run all steps end-to-end
"""

from .ground_station import load_ground_station, aggregate_daily
from .satellite import process_s5p, process_modis, convert_no2_to_ugm3, scale_aod
from .era5 import process_era5
from .features import merge_sources, build_features, final_qc, FEATURE_COLS, TARGET_COLS
from .pipeline import run_pipeline

__all__ = [
    "load_ground_station",
    "aggregate_daily",
    "process_s5p",
    "process_modis",
    "convert_no2_to_ugm3",
    "scale_aod",
    "process_era5",
    "merge_sources",
    "build_features",
    "final_qc",
    "run_pipeline",
    "FEATURE_COLS",
    "TARGET_COLS",
]
