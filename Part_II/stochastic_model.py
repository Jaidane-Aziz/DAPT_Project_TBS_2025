import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm, laplace, ks_2samp
import statsmodels.api as sm

# Load and preprocess data
df = pd.read_excel("Book2.xlsx", engine="openpyxl")
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')
df.set_index('Date', inplace=True)
spread = df['Spread'].dropna()
x_data = spread.values
dt = 1 / 252
print(df.head())
print(len(df))

# Estimate OU parameters
def ou_neg_log_likelihood(params, x, dt):
    theta, mu, sigma = params
    X_t, X_t1 = x[:-1], x[1:]
    mu_t = X_t + theta * (mu - X_t) * dt
    var_t = sigma ** 2 * dt
    return -np.sum(norm.logpdf(X_t1, loc=mu_t, scale=np.sqrt(var_t)))

res = minimize(ou_neg_log_likelihood, [1, np.mean(x_data), np.std(x_data)],
               args=(x_data, dt), bounds=[(1e-4, 10), (None, None), (1e-4, 10)])
theta_hat, mu_hat, sigma_hat = res.x

# Estimate jump parameters using top residuals
residuals = spread.diff().dropna() - theta_hat * (mu_hat - spread.shift(1).dropna()) * dt
jump_threshold = residuals.abs().quantile(0.975)  # top 2.5%
jumps = residuals[np.abs(residuals) > jump_threshold]
lambda_hat = len(jumps) / len(residuals)
mu_j_hat = np.median(jumps)  # robust central tendency
b_j_hat = np.mean(np.abs(jumps - mu_j_hat))  # Laplace scale parameter

print(f"Theta = {theta_hat:.4f}, Mu = {mu_hat:.4f}, Sigma = {sigma_hat:.4f}")
print(f"Lambda = {lambda_hat:.4f}, Jump Location = {mu_j_hat:.4f}, Jump Scale (b) = {b_j_hat:.4f}")

# Simulate Jump-Diffusion OU with Laplace jumps
def simulate_jump_diffusion_laplace(theta, mu, sigma, lamb, mu_j, b_j, X0, T, dt):
    X = np.zeros(T)
    X[0] = X0
    for t in range(1, T):
        dW = np.random.normal(0, np.sqrt(dt))
        jump = laplace.rvs(loc=mu_j, scale=b_j) if np.random.rand() < lamb * dt else 0
        X[t] = X[t-1] + theta * (mu - X[t-1]) * dt + sigma * dW + jump
    return X

T = len(spread)
jd_laplace = simulate_jump_diffusion_laplace(theta_hat, mu_hat, sigma_hat,
                                             lambda_hat, mu_j_hat, b_j_hat,
                                             spread.iloc[0], T, dt)

# Plot actual vs simulated
plt.figure(figsize=(12, 6))
plt.plot(spread.index, spread.values, label="Actual Spread", linewidth=1.2)
plt.plot(spread.index, jd_laplace, label="Jump-Diffusion OU (Laplace Jumps)", alpha=0.7)
plt.title("Improved Jump-Diffusion OU Model vs Actual Spread")
plt.xlabel("Date")
plt.ylabel("Spread")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# -----------------------------------------
# ✅ EVALUATION: KS Test + QQ Plot
# -----------------------------------------

# Kolmogorov–Smirnov (KS) test
ks_stat, ks_pval = ks_2samp(spread.values, jd_laplace)
print(f"\nKS Test Statistic: {ks_stat:.4f}, p-value: {ks_pval:.4f}")

# QQ plot (quantile-quantile plot)
sm.qqplot_2samples(jd_laplace, spread.values, line='45')
plt.title("QQ Plot: Simulated vs Actual Spread")
plt.grid(True)
plt.tight_layout()
plt.show()
