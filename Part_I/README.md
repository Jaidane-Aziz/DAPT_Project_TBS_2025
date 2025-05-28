# 📊 Part I – Basket Weight Estimation

This module focuses on estimating the intrinsic value of the USD/TND exchange rate based on global currency movements — specifically EUR/USD, GBP/USD, and USD/JPY. The goal is to construct a **basket-based valuation model** for USD/TND using historical data and interpretable machine learning.

---

## 🎯 Objective

Estimate the weights \( w_1, w_2, w_3 \) in the following equation:

\[
\text{USD/TND} = \text{Latest Fixing} \times (w_1 \cdot \%\Delta\text{EUR/USD} + w_2 \cdot \%\Delta\text{GBP/USD} + w_3 \cdot \%\Delta\text{USD/JPY})
\]

This basket defines the **baseline fundamental value** of the USD/TND rate, assuming no market distortions in Tunisia.

---

## 🧾 Dataset

- Final dataset: [`Data.xlsx`](./Data.xlsx)
- Columns include:
  - `Latest Fixing` (official USD/TND rate)
  - `%Δ EUR/USD`, `%Δ GBP/USD`, `%Δ USD/JPY` (returns)
  - `Spot USD/TND`, `Spot/Fixing` ratio
- Created by compiling and cleaning historical FX data from the three `Project_Description` folders

---

## 🔍 Exploratory Analysis

### ✅ Correlation Insights:
- Pearson analysis showed:
  - **EUR/USD** and **GBP/USD**: negative correlation with USD/TND
  - **USD/JPY**: positive correlation
- No clear linear relationship between `Spot USD/TND` and `%Δ` features → regression not ideal
- Feature engineering (e.g., EUR/GBP) yielded no additional value

---

## 🚀 Modeling Approach

### 📌 XGBoost Regressor
- Trained on:
  - `Latest Fixing`
  - `%Δ EUR/USD`, `%Δ GBP/USD`, `%Δ USD/JPY`
- Achieved **R² ≈ 0.98**
- Captured non-linear interactions

### 📈 SHAP Analysis
- Used to interpret feature importance
- Results (absolute SHAP values):
  - \( w_1 = -0.15 \)
  - \( w_2 = -0.12 \)
  - \( w_3 = +0.13 \)

These were adopted as the **weights** for the basket-based valuation.

---

## 🧪 Alternative Methods Considered

- **Macroeconomic Weighting**:
  - Based on Tunisia’s FX reserves (EUR, USD, GBP, JPY proportions)
- **PCA (Principal Component Analysis)**:
  - Explained variance ratios: [.64, .28, .08]
  - Rejected due to lack of interpretability for actual basket components

---

## 📁 Contents

Part_I/
├── Data.xlsx # Cleaned and feature-engineered dataset
├── SHAP_Values.png # (Optional) Visual plot of SHAP results
├── XGBoost_Model.py # (Optional) script for training and interpreting the model
└── README.md # This file


---

## 📤 Output

- Estimated weights: saved and used in `Integrated/hybrid_forecast.py`
- Foundation for generating the **intrinsic value** in Part II

---

## 👤 Author

Aziz Jaidane  
FIN460 – Dynamic Asset Pricing Theory  @ Tunis Business School | Université de Tunis
