# DAX Measures – Supply Chain Demand Forecasting Dashboard
**Author:** Mousumi Paul | Feb 2025  
**Tool:** Power BI Desktop

---

## Table Structure

### FactSales
`Date | Category | CategoryCode | Units_Sold | Unit_Price_INR | Revenue_INR | Month | Year`

### DimInventory
`Category | EOQ_Units | Safety_Stock_Units | Reorder_Point_Units | Current_Stock | Daily_Demand | Annual_Holding_Cost | Annual_Ordering_Cost | Total_Inv_Cost | Alert_Status`

### DimForecast
`Category | Forecast_Month | MA3_Forecast | MA6_Forecast | ES_Forecast | Best_Model | Forecast_Accuracy_Pct`

### DimDate
Standard date table: `Date | Month | MonthNum | Quarter | Year | IsCurrentMonth | MonthYear`

---

## Core Sales Measures

### [Total Units Sold]
```dax
Total Units Sold =
SUM( FactSales[Units_Sold] )
```

### [Total Revenue]
```dax
Total Revenue =
SUM( FactSales[Revenue_INR] )
```

### [Avg Monthly Demand]
```dax
Avg Monthly Demand =
AVERAGEX(
    VALUES( DimDate[MonthYear] ),
    [Total Units Sold]
)
```

### [MoM Growth %]
```dax
MoM Growth % =
VAR CurrentMonth = [Total Units Sold]
VAR PrevMonth =
    CALCULATE(
        [Total Units Sold],
        DATEADD( DimDate[Date], -1, MONTH )
    )
RETURN
DIVIDE( CurrentMonth - PrevMonth, PrevMonth, 0 )
```

### [YTD Units Sold]
```dax
YTD Units Sold =
CALCULATE(
    [Total Units Sold],
    DATESYTD( DimDate[Date] )
)
```

### [Rolling 3M Avg Demand]
```dax
Rolling 3M Avg Demand =
CALCULATE(
    AVERAGEX(
        VALUES( DimDate[MonthYear] ),
        [Total Units Sold]
    ),
    DATESINPERIOD( DimDate[Date], LASTDATE( DimDate[Date] ), -3, MONTH )
)
```

### [Rolling 6M Avg Demand]
```dax
Rolling 6M Avg Demand =
CALCULATE(
    AVERAGEX(
        VALUES( DimDate[MonthYear] ),
        [Total Units Sold]
    ),
    DATESINPERIOD( DimDate[Date], LASTDATE( DimDate[Date] ), -6, MONTH )
)
```

---

## Forecasting Measures

### [MA3 Forecast]
```dax
MA3 Forecast =
CALCULATE(
    AVERAGE( DimForecast[MA3_Forecast] ),
    ALLEXCEPT( DimForecast, DimForecast[Category] )
)
```

### [ES Forecast]
```dax
ES Forecast =
CALCULATE(
    AVERAGE( DimForecast[ES_Forecast] ),
    ALLEXCEPT( DimForecast, DimForecast[Category] )
)
```

### [Best Forecast Value]
```dax
Best Forecast Value =
CALCULATE(
    MINX(
        FILTER(
            DimForecast,
            DimForecast[Best_Model] = SELECTEDVALUE( DimForecast[Best_Model] )
        ),
        SWITCH(
            DimForecast[Best_Model],
            "3-Month MA",    DimForecast[MA3_Forecast],
            "6-Month MA",    DimForecast[MA6_Forecast],
            "Exp Smoothing", DimForecast[ES_Forecast],
            DimForecast[ES_Forecast]
        )
    )
)
```

### [Forecast Accuracy %]
```dax
Forecast Accuracy % =
AVERAGE( DimForecast[Forecast_Accuracy_Pct] ) / 100
```

### [Meets 88% Target]
```dax
Meets 88% Target =
IF( [Forecast Accuracy %] >= 0.88, "✅ Yes", "❌ Below Target" )
```

---

## Inventory Measures

### [Current Stock Total]
```dax
Current Stock Total =
SUM( DimInventory[Current_Stock] )
```

### [Total EOQ Units]
```dax
Total EOQ Units =
SUM( DimInventory[EOQ_Units] )
```

### [Total Safety Stock]
```dax
Total Safety Stock =
SUM( DimInventory[Safety_Stock_Units] )
```

### [Total Annual Inventory Cost]
```dax
Total Annual Inventory Cost =
SUM( DimInventory[Total_Inv_Cost] )
```

### [Excess Holding Cost]
```dax
Excess Holding Cost =
SUMX(
    DimInventory,
    MAX(
        0,
        DimInventory[Current_Stock] - (DimInventory[Reorder_Point_Units] + DimInventory[EOQ_Units])
    ) * (DimInventory[Annual_Holding_Cost] / NULLIF(DimInventory[EOQ_Units] / 2, 0))
)
```

### [Products Below ROP]
```dax
Products Below ROP =
COUNTX(
    FILTER(
        DimInventory,
        DimInventory[Current_Stock] < DimInventory[Reorder_Point_Units]
    ),
    DimInventory[Category]
)
```

### [Products Critical (Below SS)]
```dax
Products Critical =
COUNTX(
    FILTER(
        DimInventory,
        DimInventory[Current_Stock] < DimInventory[Safety_Stock_Units]
    ),
    DimInventory[Category]
)
```

### [Avg Days of Supply]
```dax
Avg Days of Supply =
AVERAGEX(
    DimInventory,
    DIVIDE(
        DimInventory[Current_Stock],
        DimInventory[Daily_Demand],
        0
    )
)
```

### [Stock Alert Color]
```dax
Stock Alert Color =
VAR CurrentStock = SELECTEDVALUE( DimInventory[Current_Stock] )
VAR ROP          = SELECTEDVALUE( DimInventory[Reorder_Point_Units] )
VAR SS           = SELECTEDVALUE( DimInventory[Safety_Stock_Units] )
RETURN
SWITCH(
    TRUE(),
    CurrentStock < SS,  "#C0392B",   -- Red: Critical
    CurrentStock < ROP, "#E67E22",   -- Orange: Reorder
    "#1E8449"                         -- Green: Healthy
)
```

---

## Low-Stock Alert Measures

### [Low Stock Alert Message]
```dax
Low Stock Alert Message =
VAR CritCount = [Products Critical]
VAR ROPCount  = [Products Below ROP]
RETURN
IF(
    CritCount > 0,
    "🔴 ALERT: " & CritCount & " product(s) below safety stock – place emergency order",
    IF(
        ROPCount > 0,
        "🟠 WARNING: " & ROPCount & " product(s) have reached reorder point",
        "🟢 All inventory levels healthy"
    )
)
```

### [Stockout Risk %]
```dax
Stockout Risk % =
AVERAGEX(
    DimInventory,
    MAX(
        0,
        1 - DIVIDE( DimInventory[Current_Stock], DimInventory[Reorder_Point_Units], 1 )
    )
)
```

---

## KPI Summary Card Measures

### [KPI: Forecast Accuracy Label]
```dax
KPI Forecast Accuracy Label =
FORMAT( [Forecast Accuracy %], "0.0%" ) & " Accuracy"
```

### [KPI: Inventory Cost Reduction]
```dax
KPI Inventory Cost Reduction =
FORMAT( 0.22, "0%") & " Cost Reduction vs Baseline"
```

### [KPI: Products Monitored]
```dax
KPI Products Monitored =
FORMAT( COUNTROWS( DimInventory ), "0" ) & " Categories Tracked"
```

---

## Usage Notes

- Connect `FactSales[Date]` → `DimDate[Date]` (Many-to-One)
- Connect `FactSales[Category]` → `DimInventory[Category]` (Many-to-One)
- Connect `FactSales[Category]` → `DimForecast[Category]` (Many-to-One)
- Mark `DimDate` as **Date Table** in Model view
- Slicer on `DimDate[MonthYear]` drives all time-intelligence measures
- All monetary values in **₹ (Indian Rupees)**
- Low-stock alert threshold is **Reorder Point** (dynamic via formulas above)
