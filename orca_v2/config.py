"""
orca_v2/config.py
Central configuration module for ORCA v2 data pipelines, rule builds, and forecast models.
All paths are configurable via environment variables with sensible defaults.
No secrets, usernames, passwords, or API keys are stored here.
"""

import os
from pathlib import Path

# Base directories
ORCA_V2_DIR = Path(os.environ.get("ORCA_V2_DIR", Path(__file__).resolve().parent))
ORCA_PROJECT_ROOT = Path(os.environ.get("ORCA_PROJECT_ROOT", r"C:\ORCA REMASTERED"))

# Data directories
# ORCA_LONG_DIR: Long-record parquet store for 2010-2022 (ocean_long_YYYY.parquet and rules_v3c_YYYY.parquet)
ORCA_LONG_DIR = Path(os.environ.get("ORCA_LONG_DIR", r"G:\My Drive\ORCA_LONG"))

# MERGED_V3_DIR: Near-real-time operational dataset store (2022-2025)
MERGED_V3_DIR = Path(os.environ.get("MERGED_V3_DIR", ORCA_PROJECT_ROOT / "MERGED_V3"))

# MERGED_V4_DIR: Day-of-year rolling climatology and anomalies store
MERGED_V4_DIR = Path(os.environ.get("MERGED_V4_DIR", ORCA_PROJECT_ROOT / "MERGED_V4"))

# RULES directories
RULES_V2_DIR = Path(os.environ.get("RULES_V2_DIR", ORCA_PROJECT_ROOT / "RULES_V2"))
RULES_V3C_DIR = Path(os.environ.get("RULES_V3C_DIR", ORCA_PROJECT_ROOT / "RULES_V3C"))

# Temporary raw staging directory for NetCDF downloads
RAW_TEMP_DIR = Path(os.environ.get("RAW_TEMP_DIR", r"C:\ORCA_TEMP\raw"))

# Logs directory
LOGS_DIR = Path(os.environ.get("LOGS_DIR", ORCA_PROJECT_ROOT / "logs"))

# Model artifacts and sample data inside orca_v2
MODELS_DIR = Path(os.environ.get("MODELS_DIR", ORCA_V2_DIR / "models"))
SAMPLE_DATA_DIR = Path(os.environ.get("SAMPLE_DATA_DIR", ORCA_V2_DIR / "sample_data"))
FIGURES_DIR = Path(os.environ.get("FIGURES_DIR", ORCA_V2_DIR / "figures"))
