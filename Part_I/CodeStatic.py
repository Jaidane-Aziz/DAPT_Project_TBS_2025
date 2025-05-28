import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import shap
import xgboost as xgb

# --- Load and clean data ---
file_path = "Data.xlsx"
df = pd.read_excel(file_path, sheet_name="Data")
df.dropna(subset=["Latest Fixing", "EUR/USD", "GBP/USD", "USD/JPY", "Spot USD/TND"], inplace=True)

# --- PCA: Understanding variance ---
features = ["EUR/USD", "GBP/USD", "USD/JPY"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])
pca = PCA(n_components=3)
pca.fit(X_scaled)

print("PCA Explained Variance:")
for i, var in enumerate(pca.explained_variance_ratio_, start=1):
    print(f"  PC{i}: {var:.4f}")

# --- Regression with Constraints ---
target = df["Spot USD/TND"]

# Prepare features
X_restricted = df[["EUR/USD", "GBP/USD", "USD/JPY"]]

# OLS without intercept
model_restricted = sm.OLS(target, X_restricted).fit()

print("\nRegression Summary (No Intercept, Latest Fixing Coefficient = 1):")
print(model_restricted.summary())

print("\nAdjusted Coefficients:")
for var, coef in model_restricted.params.items():
    print(f"  {var}: {coef:.6f}")

# --- Predicted Spot USD/TND and R² Evaluation ---
df["Predicted Spot USD/TND"] = (
    df["Latest Fixing"] +
    (df["EUR/USD"] * model_restricted.params["EUR/USD"]) +
    (df["GBP/USD"] * model_restricted.params["GBP/USD"]) +
    (df["USD/JPY"] * model_restricted.params["USD/JPY"])
)

r2 = r2_score(df["Spot USD/TND"], df["Predicted Spot USD/TND"])
print(f"\nManual R-squared of constrained model: {r2:.4f}")

# --- Feature Importance: Permutation ---
print("\nPermutation Importance (Linear Regression):")
X_train, X_test, y_train, y_test = train_test_split(X_restricted, target, test_size=0.2, random_state=0)
lin_model = LinearRegression().fit(X_train, y_train)
perm_result = permutation_importance(lin_model, X_test, y_test, n_repeats=30, random_state=0)
for f, imp in zip(X_restricted.columns, perm_result.importances_mean):
    print(f"  {f}: {imp:.6f}")

# --- SHAP with XGBoost ---
print("\nSHAP Values (XGBoost):")
xgb_model = xgb.XGBRegressor(random_state=0).fit(X_restricted, target)
explainer = shap.Explainer(xgb_model, X_restricted)
shap_values = explainer(X_restricted)

# Visual summary of SHAP feature importance
shap.summary_plot(shap_values, X_restricted, plot_type="bar")
