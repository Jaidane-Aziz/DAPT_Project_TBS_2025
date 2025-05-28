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
T = len(spread)

# Define functions
def ou_neg_log_likelihood_fixed_theta(mu_sigma, x, dt, theta):
    mu, sigma = mu_sigma
    X_t, X_t1 = x[:-1], x[1:]
    mu_t = X_t + theta * (mu - X_t) * dt
    var_t = sigma ** 2 * dt
    return -np.sum(norm.logpdf(X_t1, loc=mu_t, scale=np.sqrt(var_t)))

def estimate_jump_params(spread, theta, mu, dt):
    residuals = spread.diff().dropna() - theta * (mu - spread.shift(1).dropna()) * dt
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

# Try different thetas (including large ones)
theta_list = [10,13,15,17, 20, 21, 22, 23, 24, 25]
results = []

for theta in theta_list:
    # Estimate mu and sigma for fixed theta
    res = minimize(ou_neg_log_likelihood_fixed_theta, [np.mean(x_data), np.std(x_data)],
                   args=(x_data, dt, theta), bounds=[(None, None), (1e-4, 10)])
    mu_hat, sigma_hat = res.x

    # Estimate jump parameters
    lambda_hat, mu_j_hat, b_j_hat = estimate_jump_params(spread, theta, mu_hat, dt)

    # Simulate model
    jd_sim = simulate_jump_diffusion_laplace(theta, mu_hat, sigma_hat,
                                             lambda_hat, mu_j_hat, b_j_hat,
                                             spread.iloc[0], T, dt)

    # KS test
    ks_stat, ks_pval = ks_2samp(spread.values, jd_sim)

    # Store results
    results.append({
        "Theta": theta,
        "KS Statistic": round(ks_stat, 4),
        "p-value": round(ks_pval, 4)
    })

    # Plot: Simulated vs Actual Spread
    plt.figure(figsize=(12, 5))
    plt.plot(spread.index, spread.values, label="Actual Spread", linewidth=1.2)
    plt.plot(spread.index, jd_sim, label=f"Simulated (theta={theta})", alpha=0.8)
    plt.title(f"Simulated vs Actual Spread (theta={theta})")
    plt.xlabel("Date")
    plt.ylabel("Spread")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # QQ Plot
    sm.qqplot_2samples(jd_sim, spread.values, line='45')
    plt.title(f"QQ Plot: Simulated vs Actual (theta={theta})")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Display results table
results_df = pd.DataFrame(results)
print("\nComparison of KS Test Results:")
print(results_df.to_string(index=False))
