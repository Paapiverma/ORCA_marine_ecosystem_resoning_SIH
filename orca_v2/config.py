"""Path configuration for ORCA v2 data pipelines, rules, and forecast scripts."""

import os
from pathlib import Path

ORCA_V2_DIR = Path(os.environ.get("ORCA_V2_DIR", Path(__file__).resolve().parent))
ORCA_PROJECT_ROOT = Path(os.environ.get("ORCA_PROJECT_ROOT", r"C:\ORCA REMASTERED"))

# Long-record parquet store for 2010-2022 (ocean_long_YYYY.parquet, rules_v3c_YYYY.parquet).
ORCA_LONG_DIR = Path(os.environ.get("ORCA_LONG_DIR", r"G:\My Drive\ORCA_LONG"))

# Near-real-time operational dataset store (2022-2025).
MERGED_V3_DIR = Path(os.environ.get("MERGED_V3_DIR", ORCA_PROJECT_ROOT / "MERGED_V3"))

# Day-of-year rolling climatology and anomalies store.
MERGED_V4_DIR = Path(os.environ.get("MERGED_V4_DIR", ORCA_PROJECT_ROOT / "MERGED_V4"))

RULES_V2_DIR = Path(os.environ.get("RULES_V2_DIR", ORCA_PROJECT_ROOT / "RULES_V2"))
RULES_V3C_DIR = Path(os.environ.get("RULES_V3C_DIR", ORCA_PROJECT_ROOT / "RULES_V3C"))

# Raw staging directory for Copernicus NetCDF downloads.
RAW_TEMP_DIR = Path(os.environ.get("RAW_TEMP_DIR", r"C:\ORCA_TEMP\raw"))

LOGS_DIR = Path(os.environ.get("LOGS_DIR", ORCA_PROJECT_ROOT / "logs"))

MODELS_DIR = Path(os.environ.get("MODELS_DIR", ORCA_V2_DIR / "models"))
SAMPLE_DATA_DIR = Path(os.environ.get("SAMPLE_DATA_DIR", ORCA_V2_DIR / "sample_data"))
FIGURES_DIR = Path(os.environ.get("FIGURES_DIR", ORCA_V2_DIR / "figures"))
