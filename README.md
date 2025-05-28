# DAPT_Project — Real-Time Intrinsic USD/TND Interbank Rate Forecasting 🇹🇳💱

This project estimates the **real-time interbank rate of the USD/TND currency pair** in Tunisia by combining global currency fundamentals with local market liquidity effects. It was developed as part of the **FIN460 – Dynamic Asset Pricing Theory** course under the supervision of **Dr. Eymen Errais**.

---

## 📌 Objective

To compute a continuously updating **intrinsic value** of USD/TND that reflects:
- 🌍 Global economic fundamentals (EUR/USD, GBP/USD, USD/JPY)
- 💧 Local interbank market conditions (spread modeling)

The final model helps market participants evaluate whether offered quotes are fair or distorted due to short-term frictions.

---

## 🧠 Methodology Overview

The project is structured in **three parts**, each building toward a hybrid, real-time forecast engine.

---

### 📊 PART I – Basket Weight Estimation

#### Goal:
Estimate weights \( w_1, w_2, w_3 \) in:

\[
\text{USD/TND} = \text{Latest Fixing} \times (w_1 \cdot \%\Delta\text{EUR/USD} + w_2 \cdot \%\Delta\text{GBP/USD} + w_3 \cdot \%\Delta\text{USD/JPY})
\]

#### Steps:
- Data compilation from raw exchange sources
- Correlation analysis (Pearson): revealed expected signs (EUR & GBP: negative, JPY: positive)
- Feature engineering trials (e.g., EUR/GBP) – inconclusive
- Applied **XGBoost** regression (R² = 0.98)
- Used **SHAP** to extract feature impact:
  - \( w_1 = -0.15 \), \( w_2 = -0.12 \), \( w_3 = +0.13 \)

#### Alternatives Explored:
- FX reserve-based macroeconomic weighting
- PCA (explained variance: 64%, 28%, 8%) – informative but not interpretable enough

---

### 📈 PART II – Spread Modeling & Forecasting

#### Goal:
Model the **spread** between the theoretical value and the interbank rate — driven by liquidity shocks and seasonality.

#### Insights:
- Monthly spread boxplots revealed volatility clusters in months:
  `[January, February, May, June, September]`

#### Dual Modeling Approach:

- **LSTM Model**:
  - Trained on rolling windows of 30 days of interbank rates
  - Used for stable months: `[March, April, July, August, October, November, December]`
  - Captures memory and autocorrelation in the FX market

- **OU-Jump Diffusion**:
  - Stochastic model with:
    - Mean-reverting base
    - Laplace-distributed jumps
  - Parameters estimated using **Maximum Likelihood Estimation (MLE)**
  - Models real-time spread in volatile periods

---

### 🧩 INTEGRATED – Decision Engine

#### Final Assembly:
For each date:
- If month is stable → use **LSTM**
- If month is volatile → use **Parametric model**, only if:
  - `|Parametric - LSTM| ≤ 300 pips (0.03)`
  - AND `|Forecast[t] - Forecast[t-1]| ≤ 300 pips`

#### Output:
- `Forecast_LSTM`
- `Forecast_Parametric`
- `Final_Forecast` → based on decision logic

Saved to: `Forecast_By_Month_Resolved.xlsx`

---

## 📚 Key Concepts

### 🧠 Long Short-Term Memory (LSTM)

**LSTM** is a special type of Recurrent Neural Network (RNN) designed to capture **long-term dependencies** in sequential data. It's ideal for financial time series because it "remembers" patterns over time — such as cycles or delayed effects.

#### Why LSTM?
- Traditional RNNs suffer from **vanishing gradients** → they forget early time steps.
- LSTMs use **gates** (input, forget, output) to control memory flow and maintain information across time.

#### Core Equations:

Let:
- \( x_t \): input at time \( t \)  
- \( h_t \): hidden state  
- \( C_t \): cell state  

The update equations:

\[
\begin{aligned}
f_t &= \sigma(W_f \cdot [h_{t-1}, x_t] + b_f) &\quad \text{(Forget gate)} \\
i_t &= \sigma(W_i \cdot [h_{t-1}, x_t] + b_i) &\quad \text{(Input gate)} \\
\tilde{C}_t &= \tanh(W_C \cdot [h_{t-1}, x_t] + b_C) &\quad \text{(Candidate)} \\
C_t &= f_t \cdot C_{t-1} + i_t \cdot \tilde{C}_t &\quad \text{(Cell update)} \\
o_t &= \sigma(W_o \cdot [h_{t-1}, x_t] + b_o) &\quad \text{(Output gate)} \\
h_t &= o_t \cdot \tanh(C_t) &\quad \text{(New hidden state)} \\
\end{aligned}
\]

---

### 📈 Ornstein-Uhlenbeck (OU) Jump Diffusion

The **OU Jump Diffusion** model is a **mean-reverting stochastic process** that includes **occasional jumps**. It’s well-suited to FX markets where the rate typically reverts to a fundamental level — but can "jump" due to liquidity shocks, policy events, or sudden demand shifts.

#### Core Equation (SDE):

\[
dX_t = \theta(\mu - X_t)\,dt + \sigma\,dW_t + J_t\,dN_t
\]

Where:
- \( X_t \): the spread or price level  
- \( \theta \): speed of mean reversion  
- \( \mu \): long-term mean  
- \( \sigma \): volatility  
- \( dW_t \): Brownian motion (random noise)  
- \( J_t \): jump size (Laplace-distributed)  
- \( dN_t \): jump process (Poisson arrivals)

#### Features:
- Captures **mean-reversion**: price wants to return to \( \mu \)  
- Allows **jumps**: sudden, realistic shifts  
- Ideal for modeling FX spreads with seasonality and shock behavior

---

### 🔍 SHAP (SHapley Additive exPlanations)

**SHAP** explains predictions by computing the **marginal contribution** of each feature using a concept borrowed from cooperative game theory: **Shapley values**.

#### Why SHAP?
- Works with any black-box model (e.g., XGBoost)
- Assigns a fair contribution to each feature
- Visual, interpretable, and consistent

#### Equation:

The SHAP value \( \phi_i \) for feature \( i \) is:

\[
\phi_i = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|! (|F| - |S| - 1)!}{|F|!} \left[ f(S \cup \{i\}) - f(S) \right]
\]

Where:
- \( F \): full set of features  
- \( S \): subset of features excluding \( i \)  
- \( f(S) \): model output using subset \( S \)

#### In This Project:
SHAP was used to determine the relative impact of:
- \( \%\Delta \text{EUR/USD} \)
- \( \%\Delta \text{GBP/USD} \)
- \( \%\Delta \text{USD/JPY} \)

Resulting in:
- \( w_1 = -0.15 \), \( w_2 = -0.12 \), \( w_3 = +0.13 \)

---

## ⚙️ How to Run

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/DAPT_Project.git
cd DAPT_Project/Integrated
```

### 2. Clone the repository
```bash
pip install numpy pandas tensorflow scikit-learn scipy matplotlib
```

### 3. Clone the repository
```bash
python forecast_model_runner.py
```

## 🗂 Folder Structure

DAPT_Project/
├── Project_Description/   # Data sources, PDF brief
├── Part_I/                # Basket weight estimation
├── Part_II/               # LSTM + stochastic spread modeling
├── Integrated/            # Final hybrid decision engine
├── docs/                  # Architecture, diagrams
├── LICENSE.txt
└── README.md              # This file

## 👤 Author

Aziz Jaidane  
FIN460 – Dynamic Asset Pricing Theory  @ Tunis Business School | Université de Tunis