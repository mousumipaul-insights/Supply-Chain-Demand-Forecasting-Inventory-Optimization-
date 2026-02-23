# Methodology & Model Documentation
**Author:** Mousumi Paul | Feb 2025

---

## 1. Demand Forecasting Models

### 1.1 Moving Averages
**3-Month MA:** Simple average of last 3 months of actuals.  
**6-Month MA:** Simple average of last 6 months of actuals.

```
MA_t(n) = (A_{t} + A_{t-1} + ... + A_{t-n+1}) / n
```

Best for: Stable demand with no strong trend. 6-month MA smooths more noise but reacts slower.

### 1.2 Exponential Smoothing (Simple)
```
F_t = α × A_{t-1} + (1 - α) × F_{t-1}
```
Where:
- `F_t` = Forecast for period t
- `A_{t-1}` = Actual demand in previous period
- `α` = Smoothing parameter (default: **0.3**)
- Seed value = first actual observation

**Alpha selection:** α = 0.3 provides good balance between responsiveness and stability. Higher α (0.5–0.7) suits volatile demand; lower α (0.1–0.2) suits stable demand.

### 1.3 Accuracy Metrics
| Metric | Formula | Interpretation |
|--------|---------|----------------|
| MAPE | Mean(\|Actual-Forecast\|/Actual) × 100 | % error; lower is better |
| MAE  | Mean(\|Actual-Forecast\|)              | Absolute units error |
| RMSE | √Mean((Actual-Forecast)²)             | Penalizes large errors |

**Target Forecast Accuracy:** ≥ 88%  
**Overall Achieved:** ~88%+ across best models per category

---

## 2. Inventory Optimization

### 2.1 Economic Order Quantity (EOQ)
```
EOQ = √(2 × D × S / H)
```
Where:
- `D` = Annual demand (units)
- `S` = Ordering cost per order (₹2,500)
- `H` = Holding cost per unit per year (= Unit Cost × Holding %)

EOQ minimizes total inventory cost by balancing ordering frequency vs holding cost.

### 2.2 Safety Stock
```
SS = Z × σ_d × √(LT_months)
```
Where:
- `Z` = Service level z-score (1.65 for 95% service level)
- `σ_d` = Monthly demand standard deviation
- `LT_months` = Lead time in months (14 days ÷ 30 = 0.467 months)

### 2.3 Reorder Point
```
ROP = (Daily Demand × Lead Time in Days) + Safety Stock
```
When on-hand inventory reaches ROP, place a new order of EOQ units.

### 2.4 Annual Inventory Costs
```
Annual Holding Cost  = (EOQ/2 + SS) × H
Annual Ordering Cost = (D / EOQ) × S
Total Inventory Cost = Holding + Ordering
```

### 2.5 Excess Stock
```
Excess Stock = max(0, Current Stock - (ROP + EOQ))
Excess Holding Cost = Excess Stock × Holding Cost per Unit
```

---

## 3. Model Assumptions

| Parameter | Value | Source |
|-----------|-------|--------|
| Working days/year | 250 | Company calendar |
| Ordering cost | ₹2,500/order | Procurement team estimate |
| Lead time | 14 days | Supplier SLA |
| Service level | 95% (Z = 1.65) | Business requirement |
| Alpha (ES) | 0.3 | Empirically tuned |
| Forecast horizon | 3 months | Rolling quarterly |

---

## 4. Power BI Dashboard

### Low-Stock Alert Logic
```
IF Current_Stock < Safety_Stock    → 🔴 CRITICAL (emergency reorder)
IF Current_Stock < Reorder_Point   → 🟠 REORDER NOW
IF Current_Stock > (ROP + EOQ)     → 🟡 EXCESS STOCK (review demand)
ELSE                                → 🟢 HEALTHY
```

All alerts are driven by dynamic DAX measures — see `powerbi/dax_measures.md`.

---

## 5. Results Summary

| Category | Best Model | Forecast Accuracy | EOQ (units) | Safety Stock | Days of Supply |
|----------|-----------|-------------------|-------------|--------------|----------------|
| Electronics | Exp Smoothing | ~90% | 183 | 90 | ~35 |
| Apparel | Exp Smoothing | ~88% | 242 | 74 | ~21 |
| Home & Kitchen | 3-Month MA | ~89% | 200 | 56 | ~19 |
| Sports & Outdoors | 6-Month MA | ~88% | 204 | 103 | ~17 |
| Beauty & Health | Exp Smoothing | ~91% | 330 | 49 | ~26 |

Estimated reduction in excess inventory holding costs: **~22%** vs unoptimized fixed-order-quantity baseline.
