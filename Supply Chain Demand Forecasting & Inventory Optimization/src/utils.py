"""
utils.py
--------
Shared utilities: formatting, export, summary tables.
Author: Mousumi Paul | Feb 2025
"""

import pandas as pd
import os


def fmt_inr(val: float) -> str:
    return f"₹{val:,.2f}"

def fmt_units(val: float) -> str:
    return f"{int(val):,} units"

def fmt_pct(val: float) -> str:
    return f"{val:.1f}%"

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_csv(df: pd.DataFrame, path: str):
    ensure_dir(os.path.dirname(path))
    df.to_csv(path, index=False)
    print(f"✅ Saved: {path}")

def combined_report(forecast_df: pd.DataFrame, inv_df: pd.DataFrame,
                    out_path: str = "outputs/reports/combined_summary.csv"):
    """Join forecast output with inventory status for full supply chain view."""
    inv_sub = inv_df[["Category","EOQ_Units","Safety_Stock_Units",
                       "Reorder_Point_Units","Current_Stock","Days_of_Supply",
                       "Total_Inventory_Cost_INR","Alert_Status"]].copy()
    fc_sub = forecast_df.groupby("Category").agg(
        Best_Model=("Best_Model","first"),
        Forecast_Accuracy_Pct=("Forecast_Accuracy_Pct","first")
    ).reset_index()
    merged = inv_sub.merge(fc_sub, on="Category", how="left")
    save_csv(merged, out_path)
    return merged
