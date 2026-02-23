"""
demand_forecasting.py
---------------------
Demand forecasting engine: Moving Averages & Exponential Smoothing.
Author: Mousumi Paul | Feb 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os


CATEGORIES = ["Electronics", "Apparel", "Home & Kitchen", "Sports & Outdoors", "Beauty & Health"]


# ── Data Loading ──────────────────────────────────────────────────

def load_sales(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, parse_dates=["Date"])
    df["Month_Num"] = df["Date"].dt.month
    return df


def pivot_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot to wide format: rows=Category, cols=Month."""
    pivot = df.pivot_table(
        index="Category", columns="Month", values="Units_Sold", aggfunc="sum"
    )
    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    pivot = pivot[[m for m in month_order if m in pivot.columns]]
    return pivot


# ── Moving Average ────────────────────────────────────────────────

def moving_average(series: list, window: int) -> list:
    """
    Compute centered/trailing moving average.
    Returns list of same length; first (window-1) values are NaN.
    """
    result = [np.nan] * len(series)
    for i in range(window - 1, len(series)):
        result[i] = round(np.mean(series[i - window + 1: i + 1]), 2)
    return result


def forecast_ma(series: list, window: int, horizon: int = 3) -> list:
    """
    Extend series with MA-based forecast for `horizon` steps.
    Returns only the forecast values (length = horizon).
    """
    extended = list(series)
    forecasts = []
    for _ in range(horizon):
        next_val = round(np.mean(extended[-window:]), 2)
        forecasts.append(next_val)
        extended.append(next_val)
    return forecasts


# ── Exponential Smoothing ─────────────────────────────────────────

def exponential_smoothing(series: list, alpha: float = 0.3) -> list:
    """
    Simple exponential smoothing: F_t = α*A_t + (1-α)*F_{t-1}
    Seed = first actual value.
    """
    result = [series[0]]
    for t in range(1, len(series)):
        result.append(round(alpha * series[t] + (1 - alpha) * result[-1], 2))
    return result


def forecast_es(series: list, alpha: float = 0.3, horizon: int = 3) -> list:
    """Extend ES with flat forecast (last smoothed value carried forward)."""
    smoothed = exponential_smoothing(series, alpha)
    last = smoothed[-1]
    return [round(last, 2)] * horizon


# ── Accuracy Metrics ──────────────────────────────────────────────

def mape(actual: list, forecast: list) -> float:
    """Mean Absolute Percentage Error (ignores NaN values)."""
    errors = []
    for a, f in zip(actual, forecast):
        if a and a != 0 and not np.isnan(a) and f and not np.isnan(f):
            errors.append(abs(a - f) / abs(a))
    return round(np.mean(errors) * 100, 2) if errors else np.nan


def mae(actual: list, forecast: list) -> float:
    """Mean Absolute Error."""
    errors = [abs(a - f) for a, f in zip(actual, forecast)
              if a is not None and f is not None and not np.isnan(f)]
    return round(np.mean(errors), 2) if errors else np.nan


def rmse(actual: list, forecast: list) -> float:
    """Root Mean Square Error."""
    errors = [(a - f) ** 2 for a, f in zip(actual, forecast)
              if a is not None and f is not None and not np.isnan(f)]
    return round(np.sqrt(np.mean(errors)), 2) if errors else np.nan


# ── Full Forecast Pipeline ────────────────────────────────────────

def run_forecast_pipeline(df: pd.DataFrame, alpha: float = 0.3,
                           ma_short: int = 3, ma_long: int = 6,
                           horizon: int = 3) -> dict:
    """
    Run all forecasting models for each product category.

    Returns:
        dict keyed by category, each containing:
            - actuals, ma3, ma6, es_smoothed, ma3_forecast, ma6_forecast, es_forecast
            - accuracy: mape_ma3, mape_ma6, mape_es, best_model, forecast_accuracy
    """
    pivot = pivot_monthly(df)
    results = {}

    for cat in CATEGORIES:
        if cat not in pivot.index:
            continue
        actuals = pivot.loc[cat].tolist()

        ma3_vals  = moving_average(actuals, ma_short)
        ma6_vals  = moving_average(actuals, ma_long)
        es_vals   = exponential_smoothing(actuals, alpha)

        ma3_fc = forecast_ma(actuals, ma_short, horizon)
        ma6_fc = forecast_ma(actuals, ma_long,  horizon)
        es_fc  = forecast_es(actuals, alpha,    horizon)

        # MAPE on in-sample fit (skip NaN warmup)
        valid_start_3 = ma_short - 1
        valid_start_6 = ma_long  - 1

        mape_ma3 = mape(actuals[valid_start_3:], ma3_vals[valid_start_3:])
        mape_ma6 = mape(actuals[valid_start_6:], ma6_vals[valid_start_6:])
        mape_es  = mape(actuals[1:],             es_vals[1:])

        best_mape  = min(mape_ma3, mape_ma6, mape_es)
        best_model = {mape_ma3: "3-Month MA", mape_ma6: "6-Month MA", mape_es: "Exp Smoothing"}[best_mape]
        accuracy   = round((1 - best_mape / 100) * 100, 2)

        results[cat] = {
            "actuals":      actuals,
            "ma3":          ma3_vals,
            "ma6":          ma6_vals,
            "es_smoothed":  es_vals,
            "ma3_forecast": ma3_fc,
            "ma6_forecast": ma6_fc,
            "es_forecast":  es_fc,
            "accuracy": {
                "mape_ma3":         mape_ma3,
                "mape_ma6":         mape_ma6,
                "mape_es":          mape_es,
                "mae_es":           mae(actuals[1:], es_vals[1:]),
                "rmse_es":          rmse(actuals[1:], es_vals[1:]),
                "best_model":       best_model,
                "forecast_accuracy":accuracy,
                "meets_88pct":      accuracy >= 88.0,
            }
        }

    return results


# ── Reporting ─────────────────────────────────────────────────────

def print_accuracy_report(results: dict):
    print("\n" + "="*70)
    print("FORECAST ACCURACY REPORT")
    print("="*70)
    print(f"{'Category':<25} {'MA3 MAPE':>10} {'MA6 MAPE':>10} {'ES MAPE':>10} "
          f"{'Accuracy':>10} {'Best Model':<18} {'≥88%?':>6}")
    print("-"*70)
    mapes_all = []
    for cat, res in results.items():
        acc = res["accuracy"]
        check = "✅" if acc["meets_88pct"] else "❌"
        print(f"{cat:<25} {acc['mape_ma3']:>9.1f}% {acc['mape_ma6']:>9.1f}% "
              f"{acc['mape_es']:>9.1f}% {acc['forecast_accuracy']:>9.1f}% "
              f"{acc['best_model']:<18} {check:>6}")
        mapes_all.append(acc["forecast_accuracy"])
    avg_acc = round(np.mean(mapes_all), 2)
    print("-"*70)
    print(f"{'AVERAGE ACCURACY':<25} {'':>32} {avg_acc:>9.1f}%")
    print("="*70)


def forecast_summary_df(results: dict, horizon_months: list = None) -> pd.DataFrame:
    """Export forecast summary as DataFrame."""
    if horizon_months is None:
        horizon_months = ["Jan-25", "Feb-25", "Mar-25"]
    rows = []
    for cat, res in results.items():
        for i, m in enumerate(horizon_months):
            rows.append({
                "Category":       cat,
                "Forecast_Month": m,
                "MA3_Forecast":   res["ma3_forecast"][i],
                "MA6_Forecast":   res["ma6_forecast"][i],
                "ES_Forecast":    res["es_forecast"][i],
                "Best_Model":     res["accuracy"]["best_model"],
                "Forecast_Accuracy_Pct": res["accuracy"]["forecast_accuracy"],
            })
    return pd.DataFrame(rows)


# ── Charts ────────────────────────────────────────────────────────

def plot_forecast(results: dict, category: str, save_path: str = None):
    """Plot actual vs all forecast models for one category."""
    res = results[category]
    months = [f"M{i+1}" for i in range(12)]
    months_ext = months + ["J25", "F25", "M25"]

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(months, res["actuals"], "o-", color="#1B2A4A", lw=2.5, label="Actual", zorder=5)

    ma3_full = res["ma3"] + res["ma3_forecast"]
    ma6_full = res["ma6"] + res["ma6_forecast"]
    es_full  = res["es_smoothed"] + res["es_forecast"]

    ax.plot(months_ext, ma3_full,  "--", color="#2E75B6", lw=1.8, label="3-Month MA")
    ax.plot(months_ext, ma6_full,  "--", color="#1A7A6E", lw=1.8, label="6-Month MA")
    ax.plot(months_ext, es_full,   "--", color="#1E8449", lw=1.8, label="Exp Smoothing")

    ax.axvline(x=11.5, color="grey", linestyle=":", lw=1.2)
    ax.text(11.6, ax.get_ylim()[0] + 10, "Forecast →", fontsize=9, color="grey")
    ax.set_title(f"{category} – Demand Forecast (FY2024 + Jan–Mar 2025)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Month"); ax.set_ylabel("Units Sold")
    ax.legend(loc="upper left"); ax.grid(axis="y", alpha=0.3)

    acc = res["accuracy"]
    ax.text(0.98, 0.97,
            f"Best: {acc['best_model']}  |  Accuracy: {acc['forecast_accuracy']:.1f}%",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#D5F5E3", alpha=0.8))

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_all_actuals(df: pd.DataFrame, save_path: str = None):
    """Overlay line chart of all 5 categories."""
    pivot = pivot_monthly(df)
    months = list(pivot.columns)
    colors = ["#1B2A4A", "#2E75B6", "#1A7A6E", "#1E8449", "#D4A017"]

    fig, ax = plt.subplots(figsize=(13, 6))
    for i, cat in enumerate(CATEGORIES):
        if cat in pivot.index:
            ax.plot(months, pivot.loc[cat].tolist(), "o-",
                    color=colors[i], lw=2, label=cat)

    ax.set_title("Monthly Sales Trend – All Categories (FY2024)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Month"); ax.set_ylabel("Units Sold")
    ax.legend(loc="upper left"); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    df = load_sales("data/raw/sales_data_2024.csv")
    results = run_forecast_pipeline(df, alpha=0.3, ma_short=3, ma_long=6, horizon=3)
    print_accuracy_report(results)

    fc_df = forecast_summary_df(results)
    os.makedirs("data/processed", exist_ok=True)
    fc_df.to_csv("data/processed/forecast_output.csv", index=False)
    print("\n✅ Forecast saved: data/processed/forecast_output.csv")

    os.makedirs("outputs/charts", exist_ok=True)
    plot_all_actuals(df, save_path="outputs/charts/all_categories_trend.png")
    for cat in CATEGORIES:
        safe = cat.replace(" & ", "_").replace(" ", "_")
        plot_forecast(results, cat, save_path=f"outputs/charts/forecast_{safe}.png")
    print("✅ Charts saved to outputs/charts/")
