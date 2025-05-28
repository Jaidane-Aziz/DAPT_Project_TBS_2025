# 🤖 Integrated – Hybrid Forecast Decision Engine

This module is the culmination of the entire DAPT_Project. It integrates both the macroeconomic basket model (from Part I) and the time series forecasting models (from Part II) into a unified decision engine that outputs a consistent, real-time estimation of the USD/TND interbank rate.

---

## 🎯 Goal

To produce a **final forecast** of the USD/TND interbank rate that:
- Reacts to global FX movements via a basket model
- Adjusts for Tunisian market liquidity via a stochastic spread model
- Leverages deep learning (LSTM) during stable periods
- Uses statistical simulation (OU jump diffusion) in high-volatility months
- Enforces pip-based safety rules for realism

---

## 🧠 Core Forecast Logic

For each date in `Data.xlsx`:

### ✅ Use **LSTM** forecast when:
- The date falls in a **stable month**: [March, April, July, August, October, November, December]

### ✅ Use **Parametric + OU Spread** forecast when:
- The date falls in a **volatile month**: [January, February, May, June, September]

- AND two safety checks pass:
1. Difference from LSTM output ≤ **300 pips (0.03)**
2. Difference from previous forecast ≤ **300 pips**

If either rule fails, fallback to **LSTM**.

---

## 📁 File Contents

Integrated/
├── forecast_model_runner.py # ✅ Final forecast script
├── lstm_ib_model.h5 # Trained LSTM model
├── Result.xlsx # 🔚 Final result (exported)
└── README.md # 📄 This file

## 👤 Author

Aziz Jaidane  
FIN460 – Dynamic Asset Pricing Theory  @ Tunis Business School | Université de Tunis