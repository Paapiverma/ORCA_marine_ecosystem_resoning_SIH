"""
load_sample.py
Demonstration script loading the July 2022 sample dataset and printing upwelling and hypoxia stats.
"""

from pathlib import Path
import pandas as pd

SAMPLE_DIR = Path(__file__).resolve().parent / "july_2022"


def main():
    print(f"Loading sample data from {SAMPLE_DIR}...")
    df = pd.read_parquet(SAMPLE_DIR)
    print(f"Successfully loaded {len(df):,} cell-days across {df['time'].nunique()} days.")
    print(f"Date range: {df['time'].min().strftime('%Y-%m-%d')} to {df['time'].max().strftime('%Y-%m-%d')}")
    print(f"Total ocean cells per day: {len(df) // df['time'].nunique():,}")

    upw_count = int(df["upwelling"].sum())
    upw_pct = 100.0 * upw_count / len(df)
    low_o2_count = int(df["low_oxygen_risk"].sum())
    low_o2_pct = 100.0 * low_o2_count / len(df)

    print("\n=== Summary Statistics (July 2022) ===")
    print(f"Upwelling flagged cell-days: {upw_count:,} ({upw_pct:.2f}% of cell-days)")
    print(f"Low oxygen risk cell-days:   {low_o2_count:,} ({low_o2_pct:.2f}% of cell-days)")
    print(f"Mean SST (thetao):           {df['thetao'].mean():.2f} °C")
    print(f"Mean Chlorophyll (chl):      {df['chl'].mean():.2f} mg/m³")
    print(f"Mean Nitrate (no3):          {df['no3'].mean():.2f} mmol/m³")
    print(f"Mean Deep Oxygen (50-150m):  {df['o2_50_150m'].mean():.2f} mmol/m³")

    # Sample rows near Kerala coast (e.g. 9.5 N, 76.0 E on July 15)
    sample_pt = df[(df["time"] == "2022-07-15") & (df["latitude"].between(9.4, 9.6)) & (df["longitude"].between(75.8, 76.2))]
    if not sample_pt.empty:
        print("\nSample row near Kerala coast (2022-07-15):")
        cols = ["time", "latitude", "longitude", "thetao", "ref_thetao", "no3", "chl", "o2_50_150m", "upwelling", "low_oxygen_risk"]
        print(sample_pt[cols].to_string(index=False))


if __name__ == "__main__":
    main()
