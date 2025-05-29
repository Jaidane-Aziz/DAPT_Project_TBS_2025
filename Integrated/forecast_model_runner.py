import numpy as np
import pandas as pd
from pathlib import Path
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import minimize
from scipy.stats import norm, laplace

# === FILES ===
book2 = 'Book2.xlsx'
datafile = 'Data.xlsx'
model_file = 'lstm_ib_model.h5'

# === PARAMETERS ===
theta = 21
dt = 1 / 252
window_size = 30
lstm_months = [3, 4, 7, 8, 10, 11, 12]
PIP_THRESHOLD = 0.03


# === SET BASE PATH RELATIVE TO SCRIPT ===
BASE_DIR = Path(__file__).resolve().parent.parent  # points to DAPT_Project/
PART_I = BASE_DIR / "Part_I"
PART_II = BASE_DIR / "Part_II"

# === LOAD DATA ===
df_book = pd.read_excel(PART_II / "Book2.xlsx")
df_book['Date'] = pd.to_datetime(df_book['Date'])
df_book.sort_values('Date', inplace=True)

df_data = pd.read_excel(PART_I / "Data.xlsx", sheet_name='Data_Var')
df_data['Date'] = pd.to_datetime(df_data['Date'])
ref_dates = df_data['Date']

# === LSTM PREDICTION ===
ib_series = df_book[['Date', 'IB_USD']].set_index('Date')
scaler = MinMaxScaler()
ib_scaled = scaler.fit_transform(ib_series)
ib_df_scaled = pd.DataFrame(ib_scaled, index=ib_series.index, columns=['IB_Scaled'])

model = load_model(model_file)

lstm_preds = []
for date in ref_dates:
    history_start = date - pd.Timedelta(days=60)
    window_data = ib_df_scaled.loc[history_start:date - pd.Timedelta(days=1)]
    window_data = window_data.tail(window_size)

    if len(window_data) == window_size:
        input_seq = window_data.values.reshape(1, window_size, 1)
        pred_scaled = model.predict(input_seq, verbose=0)
        pred_actual = scaler.inverse_transform(pred_scaled)[0, 0]
    else:
        pred_actual = 0.0
    lstm_preds.append(pred_actual)

df_lstm_out = pd.DataFrame({'Date': ref_dates, 'Forecast_LSTM': lstm_preds})

# === PARAMETRIC MODEL ===
spread = df_book['Spread'].dropna().values
T = len(ref_dates)

def ou_neg_log_likelihood_fixed_theta(mu_sigma, x, dt, theta):
    mu, sigma = mu_sigma
    X_t, X_t1 = x[:-1], x[1:]
    mu_t = X_t + theta * (mu - X_t) * dt
    var_t = sigma ** 2 * dt
    return -np.sum(norm.logpdf(X_t1, loc=mu_t, scale=np.sqrt(var_t)))

def estimate_jump_params(spread, theta, mu, dt):
    residuals = pd.Series(spread).diff().dropna() - theta * (mu - pd.Series(spread).shift(1).dropna()) * dt
    jump_threshold = residuals.abs().quantile(0.975)
    jumps = residuals[np.abs(residuals) > jump_threshold]
    lambda_hat = len(jumps) / len(residuals)
    mu_j_hat = np.median(jumps)
    b_j_hat = np.mean(np.abs(jumps - mu_j_hat))
    return lambda_hat, mu_j_hat, b_j_hat

def simulate_jump_diffusion_laplace(theta, mu, sigma, lamb, mu_j, b_j, X0, T, dt):
    X = np.zeros(T)
    X[0] = X0
    for t in range(1, T):
        dW = np.random.normal(0, np.sqrt(dt))
        jump = laplace.rvs(loc=mu_j, scale=b_j) if np.random.rand() < lamb * dt else 0
        X[t] = X[t-1] + theta * (mu - X[t-1]) * dt + sigma * dW + jump
    return X

res = minimize(ou_neg_log_likelihood_fixed_theta, [np.mean(spread), np.std(spread)],
               args=(spread, dt, theta), bounds=[(None, None), (1e-4, 10)])
mu_hat, sigma_hat = res.x
lambda_hat, mu_j_hat, b_j_hat = estimate_jump_params(spread, theta, mu_hat, dt)
sim_spread = simulate_jump_diffusion_laplace(theta, mu_hat, sigma_hat,
                                             lambda_hat, mu_j_hat, b_j_hat,
                                             spread[0], T, dt)

# === FINAL TABLE CONSTRUCTION ===
df_final = df_data[['Date']].copy()
df_final['Forecast_LSTM'] = lstm_preds
df_final['Forecast_Parametric'] = (
    df_data['Latest Fixing'] * (
        (
        - 0.15 * df_data['Var EUR/USD'] +
        - 0.12 * df_data['Var GBP/USD'] +
        0.13 * df_data['Var USD/JPY']
    ) + 1 ) + sim_spread
)
df_final['Month'] = df_final['Date'].dt.month

# === RULE 1: Use parametric if pip difference is small in parametric months ===
def choose_model(row):
    if row['Month'] in lstm_months:
        diff = abs(row['Forecast_LSTM'] - row['Forecast_Parametric'])
        if diff <= PIP_THRESHOLD:
            return row['Forecast_Parametric']
    return row['Forecast_LSTM']

df_final['Final_Forecast'] = df_final.apply(choose_model, axis=1)

# === RULE 2: Ensure max jump <= 300 pips in final forecast ===
final_series = df_final['Final_Forecast'].copy()
for i in range(1, len(final_series)):
    jump = abs(final_series.iloc[i] - final_series.iloc[i - 1])
    if jump > PIP_THRESHOLD:
        final_series.iloc[i] = df_final['Forecast_LSTM'].iloc[i]
df_final['Final_Forecast'] = final_series

# === SAVE OUTPUT ===
df_final[['Date', 'Forecast_LSTM', 'Forecast_Parametric', 'Final_Forecast']].to_excel(
    'Result.xlsx', index=False
)
print("✅ Forecast saved as 'Result.xlsx'")
