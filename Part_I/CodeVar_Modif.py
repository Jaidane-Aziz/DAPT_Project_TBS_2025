import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import shap
import xgboost as xgb
import matplotlib.pyplot as plt

# --- Load Data ---
file_path = "Data.xlsx"
df = pd.read_excel(file_path, sheet_name="Data_Var")
df.dropna(subset=[ "Var EUR/USD", "Var GBP/USD", "Var USD/JPY", "Spot/Fixing"], inplace=True)

# --- PCA ---
features = ["Var EUR/USD", "Var GBP/USD", "Var USD/JPY"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])
pca = PCA(n_components=3)
pca.fit(X_scaled)

print("PCA Explained Variance:")
for i, var in enumerate(pca.explained_variance_ratio_, start=1):
    print(f"  PC{i}: {var:.4f}")

# --- Target and Feature Definition for ML ---
X = df[["Var EUR/USD", "Var GBP/USD", "Var USD/JPY"]]
y = df["Spot/Fixing"]

# --- Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Train XGBoost Regressor ---
xgb_model = xgb.XGBRegressor(random_state=42)
xgb_model.fit(X_train, y_train)

# --- Evaluate Model ---
y_pred = xgb_model.predict(X_test)

# --- SHAP Analysis ---
print("\nComputing SHAP values...")
explainer = shap.Explainer(xgb_model, X)
shap_values = explainer(X) * 100

# --- SHAP Summary Plot ---
shap.summary_plot(shap_values, X, plot_type="bar")
