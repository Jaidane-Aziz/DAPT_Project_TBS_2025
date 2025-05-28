import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError


# === Step 1: Load and Prepare Data ===
file_path = 'Book2.xlsx'
df = pd.read_excel(file_path)
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# === Step 2: Focus on IB Rate ===
ib_series = df['IB_USD'].values.reshape(-1, 1)

# Normalize
scaler = MinMaxScaler()
ib_scaled = scaler.fit_transform(ib_series)

# === Step 3: Create Sequences ===
window_size = 30
X, y = [], []

for i in range(window_size, len(ib_scaled)):
    X.append(ib_scaled[i - window_size:i])
    y.append(ib_scaled[i])

X = np.array(X)
y = np.array(y)

# === Step 4: Train/Test Split ===
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# === Step 5: Build Enhanced LSTM Model ===
model = Sequential()

# First LSTM layer
model.add(LSTM(64, return_sequences=True, input_shape=(window_size, 1)))
model.add(LeakyReLU(alpha=0.1))

# Second LSTM layer
model.add(LSTM(32))
model.add(LeakyReLU(alpha=0.1))

# Dense layers
model.add(Dense(16, activation='relu'))
model.add(Dense(1))  # Output layer

# Compile
model.compile(optimizer='adam', loss=MeanSquaredError())

# Train with early stopping
early_stop = EarlyStopping(patience=10, restore_best_weights=True)
model.fit(X_train, y_train, epochs=100, batch_size=16, validation_split=0.1,
          callbacks=[early_stop], verbose=1)

# === Step 6: Save the model ===
model.save('lstm_ib_model.h5')
print("✅ Model saved as 'lstm_ib_model.h5'")

# === Step 7: One-Step Forecast Backtesting ===
predictions = []
input_seq = X_test[0]

for i in range(len(X_test)):
    pred = model.predict(input_seq.reshape(1, window_size, 1), verbose=0)
    predictions.append(pred[0][0])
    if i + 1 < len(X_test):
        input_seq = X_test[i + 1]

# Inverse scale
predictions_rescaled = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

# === Step 8: Plotting Results ===
plt.figure(figsize=(12, 6))
plt.plot(y_test_rescaled, label='Actual IB Rate', color='blue')
plt.plot(predictions_rescaled, label='Predicted IB Rate', color='orange')
plt.title('IB Rate One-Step Ahead Forecast (Backtesting)')
plt.xlabel('Time Step')
plt.ylabel('IB Rate')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()