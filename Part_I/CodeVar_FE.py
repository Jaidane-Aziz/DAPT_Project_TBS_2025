import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import shap
import xgboost as xgb
import matplotlib.pyplot as plt

# --- Load and clean data ---
file_path = "Data.xlsx"
df = pd.read_excel(file_path, sheet_name="Data_Var_FE")
df.dropna(subset=["Latest Fixing","Var EUR/GBP", "Var USD/JPY", "Spot USD/TND"], inplace=True)

# --- PCA Step (Optional) ---
features = ["Var EUR/GBP", "Var USD/JPY"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])
pca = PCA(n_components=2)
pca.fit(X_scaled)

print("PCA Explained Variance:")
for i, var in enumerate(pca.explained_variance_ratio_, start=1):
    print(f"  PC{i}: {var:.4f}")

# --- Regression with Constraints ---
target = df["Spot USD/TND"]
X_restricted = df[["Var EUR/GBP", "Var USD/JPY"]]

model_restricted = sm.OLS(target, X_restricted).fit()

print("\nRegression Summary (No Intercept, Latest Fixing Coefficient = 1):")
print(model_restricted.summary())

print("\nAdjusted Weights (Coefficients):")
for var, coef in model_restricted.params.items():
    print(f"  {var}: {coef:.6f}")

df["Predicted Spot USD/TND"] = (
    df["Latest Fixing"] *
    df["Var EUR/GBP"] * model_restricted.params["Var EUR/GBP"] +
    df["Var USD/JPY"] * model_restricted.params["Var USD/JPY"]
)

r2_ols = r2_score(df["Spot USD/TND"], df["Predicted Spot USD/TND"])
print(f"\nManual R-squared of constrained model: {r2_ols:.4f}")

# --- Prepare for ML Models ---
X = df[["Latest Fixing","Var EUR/GBP", "Var USD/JPY"]]
y = df["Spot USD/TND"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- XGBoost Regression ---
xgb_model = xgb.XGBRegressor(random_state=42)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
r2_xgb = r2_score(y_test, y_pred_xgb)
print(f"\nXGBoost R² on Test Set: {r2_xgb:.4f}")

# --- Random Forest Regression (Depth=10) ---
rf_model = RandomForestRegressor(max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
r2_rf = r2_score(y_test, y_pred_rf)
print(f"Random Forest R² (max_depth=10) on Test Set: {r2_rf:.4f}")

# --- SHAP for XGBoost ---
print("\nSHAP Values for XGBoost:")
explainer_xgb = shap.Explainer(xgb_model, X)
shap_values_xgb = explainer_xgb(X)
shap.summary_plot(shap_values_xgb, X, plot_type="bar")

# --- SHAP for Random Forest ---
print("\nSHAP Values for Random Forest:")
explainer_rf = shap.Explainer(rf_model, X)
shap_values_rf = explainer_rf(X)
shap.summary_plot(shap_values_rf, X, plot_type="bar")
