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
df.dropna(subset=["Var EUR/GBP", "Var USD/JPY", "Spot/Fixing"], inplace=True)

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
target = df["Spot/Fixing"]
X_restricted = df[["Var EUR/GBP", "Var USD/JPY"]]

# --- Prepare for ML Models ---
X = df[["Var EUR/GBP", "Var USD/JPY"]]
y = df["Spot/Fixing"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- XGBoost Regression ---
xgb_model = xgb.XGBRegressor(random_state=42)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

# --- Random Forest Regression (Depth=10) ---
rf_model = RandomForestRegressor(max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# --- SHAP for XGBoost ---
print("\nSHAP Values for XGBoost:")
explainer_xgb = shap.Explainer(xgb_model, X)
shap_values_xgb = explainer_xgb(X) * 100
shap.summary_plot(shap_values_xgb, X, plot_type="bar")

# --- SHAP for Random Forest ---
print("\nSHAP Values for Random Forest:")
explainer_rf = shap.Explainer(rf_model, X)
shap_values_rf = explainer_rf(X) * 1000
shap.summary_plot(shap_values_rf, X, plot_type="bar")
