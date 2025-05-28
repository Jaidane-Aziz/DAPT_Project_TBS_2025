# 🧭 Project Architecture – USD/TND Interbank Rate Estimation

This document outlines the full architecture of the DAPT_Project, from data input to final interbank rate output. It is designed to support transparency, reproducibility, and academic review.

---

## 📈 Overview

The model estimates the **real-time intrinsic interbank rate** of USD/TND by combining:
1. A macroeconomic basket-based pricing model (Part I)
2. Two modeling approaches for the actual traded rate (Part II):
   - **LSTM model** (deep learning)
   - **OU-Jump Diffusion model** (stochastic simulation)
3. A hybrid **decision engine** (Integrated) that selects the appropriate model output based on seasonality and pip-tolerance filters.

---

## 🔁 Full Forecast Flow

                         ┌────────────────────────────┐
                         │     Global FX Data Inputs  │
                         │  (EUR/USD, GBP/USD, JPY)   │
                         └──────────────┬─────────────┘
                                        │
                         ┌──────────────▼─────────────┐
                         │  Basket Weight Estimation  │
                         │                            │
                         └──────────────┬─────────────┘
                                        │
                        (Latest Fixing × Basket Output)
                                        ↓
                         ┌────────────────────────────┐
                         │  Intrinsic Value (USD/TND) │
                         └──────────────┬─────────────┘
                                        │
        ┌───────────────────────────────┴───────────────────────────────┐
        │                                                               │
 ┌──────▼──────┐                                                 ┌──────▼────────────────┐
 │  LSTM Model │                                                 │ OU-Jump Diffusion     │
 │ (Stable Mo.)│                                                 │ Model (Volatile Mo.)  │
 └──────┬──────┘                                                 └────────┬──────────────┘
        │                                                                 │
        └────────────────────┬────────────────────────────────────────────┘
                             ▼
               ┌────────────────────────────────────┐
               │     Hybrid Decision Engine         │
               │ (month rule + pip-based safety)    │
               └────────────────────┬───────────────┘
                                    │
                            Final Forecast Output
                                    ↓
               ┌────────────────────────────────────┐
               │            Result.xlsx             │
               └────────────────────────────────────┘


---

## 🧠 Decision Logic

The hybrid engine works as follows:

### 📅 Model Selection by Month:
- **LSTM** is used when the month ∈ `[3, 4, 7, 8, 10, 11, 12]`
- **Parametric + OU Spread** is used when month ∈ `[1, 2, 5, 6, 9]`, *only if*:
  - `|Forecast_Parametric - Forecast_LSTM| ≤ 0.03`
  - `|Final[t] - Final[t-1]| ≤ 0.03`

If conditions fail, fallback to **LSTM**.

---

## 📤 Output Columns

Final `.xlsx` output includes:
- `Date`
- `Forecast_LSTM`
- `Forecast_Parametric`
- `Final_Forecast`

Used to assess market quotes against fair value in real time.

---

## 🧱 Components Summary

| Component         | Role                                                       |
|------------------|-------------------------------------------------------------|
| Basket Model      | Computes intrinsic FX value from global macro movements    |
| LSTM              | Captures memory and autocorrelation in interbank data      |
| OU-Jump Simulator | Models liquidity-driven jump behavior in spread            |
| Hybrid Engine     | Dynamically switches between models based on month & logic |

---

## 📎 Associated Diagram

![Forecast Architecture](../docs/A_flowchart_depicts_the_architecture_of_a_USD/TND_.png)

---

## 👤 Author

Aziz Jaidane  
FIN460 – Dynamic Asset Pricing Theory  @ Tunis Business School | Université de Tunis
