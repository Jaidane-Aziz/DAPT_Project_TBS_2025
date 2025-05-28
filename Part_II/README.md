# 📈 Part II – Spread Modeling & Interbank Forecasting

This part of the project builds the **core forecasting engines** that simulate the real-time USD/TND interbank exchange rate. It integrates two complementary approaches:

1. A memory-based **LSTM model** trained on historical interbank data.
2. A stochastic **Ornstein-Uhlenbeck (OU) Jump Diffusion model** that simulates market spread behavior during volatile months.

---

## 🎯 Objective

To model and forecast the **spread** between the fundamental (basket-based) USD/TND rate and the **actual interbank rate**, reflecting local liquidity pressures and market frictions in the Tunisian FX market.

---

## 🗃️ Dataset

- Main data file: [`Book2.xlsx`](./Book2.xlsx)
- Contains:
  - `Date`
  - `IB_USD`: Historical interbank rates (target for LSTM)
  - `Spot USD/TND`
  - `Spread`: Manually computed as `IB_USD - Spot`

---

## 🔍 Spread Seasonality Analysis

- Spreads were grouped by month (2020–2025)
- Monthly boxplots revealed **outlier behavior** in:


- These months correspond to high economic activity, foreign transfers, and FX shocks (e.g., tourism seasons)

---

## 🧠 Model 1: Long Short-Term Memory (LSTM)

### Description:
- A type of Recurrent Neural Network (RNN)
- Ideal for capturing sequential dependencies in interbank rate movements

### Configuration:
- Input: Last 30 days of `IB_USD` (sliding window)
- Output: Next day's interbank rate
- Layers:
- LSTM (64 → LeakyReLU → LSTM (32))
- Dense(16 → relu → 1)
- Training:
- Early stopping
- Mean Squared Error loss
- Output stored as: `lstm_ib_model.h5`

---

## 📉 Model 2: OU-Jump Diffusion Spread Simulator

### Description:
- Models spread as a **mean-reverting process with random jumps**

\[
dX_t = \theta(\mu - X_t)\,dt + \sigma\,dW_t + J_t\,dN_t
\]

Where:
- \( X_t \): Spread level
- \( \theta \): Reversion speed
- \( \mu \): Long-term mean
- \( \sigma \): Volatility
- \( dW_t \): Brownian motion
- \( J_t \): Laplace-distributed jump size
- \( dN_t \): Poisson-distributed jump frequency

### Estimation:
- Parameters estimated via **Maximum Likelihood Estimation (MLE)**
- Best result found for:
- \( \theta = 21 \)
- Custom-tuned \( \mu, \sigma \) and jump parameters

---

## 🔧 Output Use

- **LSTM predictions** used for stable months:

- **OU simulation outputs** used for volatile months (above)
- Spread added to Part I model output
- Captures market frictions and liquidity-driven pricing

---

## 📁 Contents

Part_II/
├── Book2.xlsx # Spread and interbank data
├── LSTM_Model.py # Code to train and save the LSTM model
├── OU_Jump_Model.py # Simulation and parameter tuning for stochastic model
├── Spread_Monthly_Boxplots/ # Visual analysis of spread seasonality
├── Notes.txt # Model notes and testing logs
└── README.md # This file


---

## 📤 Output

- Trained LSTM model: `lstm_ib_model.h5`
- Simulated spread series → integrated in final pipeline

---

## 👤 Author

Aziz Jaidane  
FIN460 – Dynamic Asset Pricing Theory  @ Tunis Business School | Université de Tunis
