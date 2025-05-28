import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# === Load Model ===
model = load_model('lstm_ib_model.h5')

# === Load and Prepare Data ===
file_path = 'Book2.xlsx'
df = pd.read_excel(file_path)
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# === Normalize IB Rate ===
ib_series = df['IB_USD'].values.reshape(-1, 1)
scaler = MinMaxScaler()
ib_scaled = scaler.fit_transform(ib_series)

# === Full Prediction Using Sliding Window ===
window_size = 30
all_inputs = ib_scaled
full_predictions = []

for i in range(window_size, len(all_inputs)):
    input_seq = all_inputs[i - window_size:i]
    pred = model.predict(input_seq.reshape(1, window_size, 1), verbose=0)
    full_predictions.append(pred[0][0])

# Inverse scale
full_predictions_rescaled = scaler.inverse_transform(np.array(full_predictions).reshape(-1, 1))

# Corresponding Dates and Spot USD/TND
dates = df['Date'].iloc[window_size:].reset_index(drop=True)
spot_values = df['Spot USD/TND'].iloc[window_size:].reset_index(drop=True)

# === Final DataFrame ===
result_df = pd.DataFrame({
    'Date': dates,
    'Predicted_IB_Rate': full_predictions_rescaled.flatten(),
    'Spot_USD_TND': spot_values
})

# === Calculate Spread and Summary Statistics ===
result_df['Spread'] = result_df['Predicted_IB_Rate'] - result_df['Spot_USD_TND']

# Summary statistics
avg_spread = result_df['Spread'].mean()
min_spread = result_df['Spread'].min()
max_spread = result_df['Spread'].max()

# Proportions
negative_spreads = result_df[result_df['Spread'] < 0]
positive_spreads = result_df[result_df['Spread'] > 0]

count_negative = len(negative_spreads)
count_positive = len(positive_spreads)
total = len(result_df)

prop_negative = count_negative / total
prop_positive = count_positive / total

# === Output Results ===
print("Average Spread:", avg_spread)
print("Min Spread:", min_spread)
print("Max Spread:", max_spread)
print(f"Negative Spreads: {count_negative} ({prop_negative:.2%})")
print(f"Positive Spreads: {count_positive} ({prop_positive:.2%})")

# Optional: Save Results
# result_df.to_excel("IB_vs_Spot_Analysis.xlsx", index=False)
