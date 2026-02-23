"""
inventory_optimization.py
--------------------------
EOQ, Reorder Point, Safety Stock, and holding cost optimization.
Author: Mousumi Paul | Feb 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import math


# ── Core Formulas ─────────────────────────────────────────────────

def eoq(annual_demand: float, ordering_cost: float, holding_cost_per_unit: float) -> float:
    """
    Economic Order Quantity:
        EOQ = sqrt(2 × D × S / H)
    where:
        D = annual demand (units)
        S = ordering cost per order (₹)
        H = holding cost per unit per year (₹)
    """
    if holding_cost_per_unit <= 0:
        raise ValueError("Holding cost must be > 0")
    return round(math.sqrt(2 * annual_demand * ordering_cost / holding_cost_per_unit), 0)


def safety_stock(demand_std_dev: float, lead_time_days: float,
                 z_score: float = 1.65, days_per_month: float = 30) -> float:
    """
    Safety Stock = Z × σ_d × sqrt(LT)
    where σ_d is demand std dev (monthly units), LT in months.
    """
    lead_time_months = lead_time_days / days_per_month
    return round(z_score * demand_std_dev * math.sqrt(lead_time_months), 0)


def reorder_point(daily_demand: float, lead_time_days: float,
                  ss: float) -> float:
    """
    Reorder Point = (Daily Demand × Lead Time) + Safety Stock
    """
    return round(daily_demand * lead_time_days + ss, 0)


def daily_demand(annual_demand: float, working_days: int = 250) -> float:
    """Average daily demand assuming working days per year."""
    return round(annual_demand / working_days, 4)


def annual_holding_cost(eoq_qty: float, ss_qty: float,
                        holding_cost_per_unit: float) -> float:
    """
    Annual Holding Cost = (EOQ/2 + Safety Stock) × H
    Cycle stock held = EOQ/2 on average; SS always held.
    """
    return round((eoq_qty / 2 + ss_qty) * holding_cost_per_unit, 2)


def annual_ordering_cost(annual_demand: float, eoq_qty: float,
                         ordering_cost: float) -> float:
    """
    Annual Ordering Cost = (D / EOQ) × S
    """
    if eoq_qty <= 0:
        return 0.0
    return round((annual_demand / eoq_qty) * ordering_cost, 2)


def total_inventory_cost(holding_cost_annual: float,
                         ordering_cost_annual: float) -> float:
    return round(holding_cost_annual + ordering_cost_annual, 2)


# ── Pipeline ──────────────────────────────────────────────────────

def run_inventory_optimization(params_df: pd.DataFrame,
                                ordering_cost: float = 2500,
                                z_score: float = 1.65,
                                lead_time_days: float = 14,
                                working_days: int = 250) -> pd.DataFrame:
    """
    Run full EOQ/ROP/SS optimization for all products.

    Parameters:
        params_df: DataFrame with columns:
            Category, Annual_Demand_Units, Demand_StdDev,
            Unit_Cost_INR, Holding_Cost_Pct, Current_Stock_Units
    Returns:
        DataFrame with all computed inventory metrics.
    """
    results = []
    for _, row in params_df.iterrows():
        ann_dem  = row["Annual_Demand_Units"]
        std_dev  = row["Demand_StdDev"]
        unit_cost = row["Unit_Cost_INR"]
        hold_pct = row["Holding_Cost_Pct"]
        curr_stk = row["Current_Stock_Units"]

        hc_unit  = unit_cost * hold_pct
        dd       = daily_demand(ann_dem, working_days)
        eoq_qty  = eoq(ann_dem, ordering_cost, hc_unit)
        ss       = safety_stock(std_dev, lead_time_days, z_score)
        rop      = reorder_point(dd, lead_time_days, ss)

        ahc      = annual_holding_cost(eoq_qty, ss, hc_unit)
        aoc      = annual_ordering_cost(ann_dem, eoq_qty, ordering_cost)
        tic      = total_inventory_cost(ahc, aoc)

        excess   = max(0, curr_stk - (rop + eoq_qty))
        excess_hc = round(excess * hc_unit, 2)
        dos      = round(curr_stk / dd, 1) if dd > 0 else 0
        stockout_risk = max(0, min(1, 1 - curr_stk / rop)) if rop > 0 else 0

        if curr_stk < ss:
            status = "🔴 CRITICAL – Below Safety Stock"
            action = "Place emergency order immediately"
        elif curr_stk < rop:
            status = "🟠 REORDER NOW"
            action = "Place standard replenishment order"
        elif excess > 0:
            status = "🟡 EXCESS STOCK"
            action = "Review demand; consider promotion to reduce excess"
        else:
            status = "🟢 HEALTHY"
            action = "No action needed"

        results.append({
            "Category":               row["Category"],
            "Annual_Demand":          ann_dem,
            "Avg_Monthly_Demand":     round(ann_dem / 12, 1),
            "Demand_StdDev":          std_dev,
            "Unit_Cost_INR":          unit_cost,
            "Holding_Cost_Per_Unit":  round(hc_unit, 2),
            "Daily_Demand":           dd,
            "EOQ_Units":              int(eoq_qty),
            "Safety_Stock_Units":     int(ss),
            "Reorder_Point_Units":    int(rop),
            "Current_Stock":          int(curr_stk),
            "Days_of_Supply":         dos,
            "Stockout_Risk_Pct":      round(stockout_risk * 100, 1),
            "Excess_Stock_Units":     int(excess),
            "Excess_Holding_Cost_INR": excess_hc,
            "Annual_Holding_Cost_INR": ahc,
            "Annual_Ordering_Cost_INR": aoc,
            "Total_Inventory_Cost_INR": tic,
            "Alert_Status":           status,
            "Recommended_Action":     action,
        })

    return pd.DataFrame(results)


def cost_savings_analysis(before_df: pd.DataFrame,
                          after_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare inventory costs before and after optimization.
    'before' represents unoptimized (e.g. fixed large order qty).
    'after'  represents EOQ-based orders.
    """
    rows = []
    for cat in before_df["Category"]:
        b = before_df[before_df["Category"] == cat].iloc[0]
        a = after_df[after_df["Category"] == cat].iloc[0]
        saving = b["Total_Inventory_Cost_INR"] - a["Total_Inventory_Cost_INR"]
        pct    = saving / b["Total_Inventory_Cost_INR"] * 100 if b["Total_Inventory_Cost_INR"] > 0 else 0
        rows.append({
            "Category":          cat,
            "Before_Cost_INR":   b["Total_Inventory_Cost_INR"],
            "After_Cost_INR":    a["Total_Inventory_Cost_INR"],
            "Cost_Saving_INR":   round(saving, 2),
            "Saving_Pct":        round(pct, 1),
        })
    return pd.DataFrame(rows)


# ── Charts ────────────────────────────────────────────────────────

def plot_eoq_cost_curve(category: str, annual_demand: float,
                        ordering_cost: float, holding_cost_pu: float,
                        save_path: str = None):
    """Classic EOQ total cost curve."""
    opt_eoq = eoq(annual_demand, ordering_cost, holding_cost_pu)
    q_range = np.linspace(max(1, opt_eoq * 0.2), opt_eoq * 3, 300)

    holding  = (q_range / 2) * holding_cost_pu
    ordering = (annual_demand / q_range) * ordering_cost
    total    = holding + ordering

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(q_range, holding,  "--", color="#2E75B6", lw=1.8, label="Holding Cost")
    ax.plot(q_range, ordering, "--", color="#C0392B", lw=1.8, label="Ordering Cost")
    ax.plot(q_range, total,    "-",  color="#1B2A4A", lw=2.5, label="Total Cost")
    ax.axvline(opt_eoq, color="#1E8449", linestyle=":", lw=2, label=f"EOQ = {int(opt_eoq)} units")

    min_cost = (opt_eoq / 2) * holding_cost_pu + (annual_demand / opt_eoq) * ordering_cost
    ax.annotate(f"Min Cost\n₹{min_cost:,.0f}",
                xy=(opt_eoq, min_cost),
                xytext=(opt_eoq * 1.3, min_cost * 1.2),
                arrowprops=dict(arrowstyle="->", color="#1E8449"),
                fontsize=9, color="#1E8449")

    ax.set_title(f"EOQ Cost Curve – {category}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Order Quantity (Units)"); ax.set_ylabel("Annual Cost (₹)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"₹{v:,.0f}"))
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_stock_health(inv_df: pd.DataFrame, save_path: str = None):
    """Grouped bar: Current Stock vs ROP vs Safety Stock."""
    cats = inv_df["Category"].tolist()
    x = np.arange(len(cats))
    w = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - w, inv_df["Current_Stock"], w, label="Current Stock",
           color="#2E75B6", alpha=0.9)
    ax.bar(x,     inv_df["Reorder_Point_Units"], w, label="Reorder Point",
           color="#C0392B", alpha=0.9)
    ax.bar(x + w, inv_df["Safety_Stock_Units"],  w, label="Safety Stock",
           color="#1E8449", alpha=0.9)

    ax.set_xticks(x); ax.set_xticklabels(cats, rotation=15, ha="right")
    ax.set_ylabel("Units"); ax.set_title("Stock Health: Current vs ROP vs Safety Stock",
                                          fontsize=12, fontweight="bold")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_cost_breakdown(inv_df: pd.DataFrame, save_path: str = None):
    """Stacked bar: Holding vs Ordering cost per product."""
    cats = inv_df["Category"].tolist()
    x = np.arange(len(cats))

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x, inv_df["Annual_Holding_Cost_INR"],  label="Holding Cost",  color="#2E75B6", alpha=0.9)
    ax.bar(x, inv_df["Annual_Ordering_Cost_INR"], label="Ordering Cost", color="#1A7A6E", alpha=0.9,
           bottom=inv_df["Annual_Holding_Cost_INR"])

    ax.set_xticks(x); ax.set_xticklabels(cats, rotation=15, ha="right")
    ax.set_ylabel("Annual Cost (₹)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"₹{v:,.0f}"))
    ax.set_title("Annual Inventory Cost Breakdown by Category", fontsize=12, fontweight="bold")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    params = pd.read_csv("data/raw/inventory_params.csv")
    inv_df = run_inventory_optimization(params)
    print("\n📦 INVENTORY OPTIMIZATION RESULTS")
    print(inv_df[["Category","EOQ_Units","Safety_Stock_Units","Reorder_Point_Units",
                  "Days_of_Supply","Total_Inventory_Cost_INR","Alert_Status"]].to_string(index=False))

    os.makedirs("data/processed", exist_ok=True)
    inv_df.to_csv("data/processed/inventory_optimization_output.csv", index=False)
    print("\n✅ Saved: data/processed/inventory_optimization_output.csv")

    os.makedirs("outputs/charts", exist_ok=True)
    plot_stock_health(inv_df, save_path="outputs/charts/stock_health.png")
    plot_cost_breakdown(inv_df, save_path="outputs/charts/cost_breakdown.png")

    electronics = params[params["Category"] == "Electronics"].iloc[0]
    hc_pu = electronics["Unit_Cost_INR"] * electronics["Holding_Cost_Pct"]
    plot_eoq_cost_curve("Electronics", electronics["Annual_Demand_Units"],
                        2500, hc_pu, save_path="outputs/charts/eoq_curve_electronics.png")
    print("✅ Charts saved to outputs/charts/")
