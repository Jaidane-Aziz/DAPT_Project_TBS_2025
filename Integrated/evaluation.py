import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# === Load the Excel File ===
file_path = "test.xlsx"  # Make sure this is in your current working directory
df = pd.read_excel(file_path)

# === Extract True and Predicted Values ===
y_true = df['IB_Rate']
y_pred = df['Final_Forecast']

# === Compute Accuracy Metrics ===
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

# === Print Metrics ===
print("📊 Accuracy Metrics")
print(f"Mean Squared Error (MSE): {mse:.8f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.8f}")
print(f"Mean Absolute Error (MAE): {mae:.8f}")
print(f"R-squared (R²): {r2:.6f}")

# === Visualization: Actual vs Predicted ===
plt.figure(figsize=(14, 6))
plt.plot(df['Date'], y_true, label='Actual IB Rate', linewidth=2)
plt.plot(df['Date'], y_pred, label='Predicted IB Rate', linestyle='--', linewidth=2)
plt.title('Actual vs Predicted USD/TND Interbank Rate (Backtesting)')
plt.xlabel('Date')
plt.ylabel('USD/TND Rate')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

