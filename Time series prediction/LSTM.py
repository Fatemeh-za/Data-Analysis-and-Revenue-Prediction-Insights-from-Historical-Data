
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings("ignore")

# Ensure your DataFrame 'movies_df' is defined before this code

# Convert 'release_date' to datetime and extract year
movies_df["release_date"] = pd.to_datetime(movies_df["release_date"], errors='coerce')
movies_df["release_year"] = movies_df["release_date"].dt.year

# Aggregate yearly revenue
yearly_revenue = movies_df.groupby("release_year")["revenue"].sum().reset_index()

# Log transform revenue to reduce skewness
yearly_revenue["log_revenue"] = np.log1p(yearly_revenue["revenue"])

# Set 'release_year' as index
yearly_revenue.set_index("release_year", inplace=True)

# Standardize the revenue data
scaler = StandardScaler()
yearly_revenue_scaled = scaler.fit_transform(yearly_revenue["log_revenue"].values.reshape(-1, 1))

# Plot the scaled yearly revenue data
plt.figure(figsize=(12, 6))
plt.plot(yearly_revenue.index, yearly_revenue_scaled, marker='o')
plt.title("Standardized Yearly Revenue (Log-Scaled)")
plt.xlabel("Year")
plt.ylabel("Standardized Revenue")
plt.show()

# Create sequences for LSTM
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i + seq_length]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# Use a longer sequence length
seq_length = 10  # Increased from 5
X, y = create_sequences(yearly_revenue_scaled, seq_length)

# Split into training and testing sets
train_size = int(0.9 * len(X))  # Increased training size to 90%
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]

# Reshape data for LSTM
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Build improved LSTM model
model = Sequential()
model.add(LSTM(100, return_sequences=True, input_shape=(seq_length, 1)))  # Increased units
model.add(Dropout(0.1))  # Reduced dropout
model.add(LSTM(100, return_sequences=False))
model.add(Dropout(0.1))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model with early stopping
history = model.fit(X_train, y_train, epochs=100, batch_size=4, validation_split=0.2, callbacks=[early_stopping])

# Make predictions
predictions_scaled = model.predict(X_test)

# Reverse scaling for predictions and actual test data
predictions = scaler.inverse_transform(predictions_scaled)
y_test_rescaled = scaler.inverse_transform(y_test)

# Evaluate the model
mse = mean_squared_error(y_test_rescaled, predictions)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_rescaled, predictions)

print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(yearly_revenue.index[seq_length:train_size+seq_length], scaler.inverse_transform(y_train), label="Training Data")
plt.plot(yearly_revenue.index[train_size+seq_length:], y_test_rescaled, label="Actual Revenue")
plt.plot(yearly_revenue.index[train_size+seq_length:], predictions, label="Predicted Revenue", linestyle='--')
plt.title("Time Series Revenue Prediction with Improved LSTM")
plt.xlabel("Year")
plt.ylabel("Revenue (Log-Scaled)")
plt.legend()
plt.show()

# Forecast future revenue for the next 5 years
future_forecast_scaled = model.predict(np.array([yearly_revenue_scaled[-seq_length:]]))
future_forecast = scaler.inverse_transform(future_forecast_scaled)

print("Future Revenue Forecast for Next Year:")
print(future_forecast)




    
# ![png](output_18_0.png)
    


#     Epoch 1/100
#     23/23 [==============================] - 7s 88ms/step - loss: 0.2650 - val_loss: 0.0047
#     Epoch 2/100
#     23/23 [==============================] - 0s 18ms/step - loss: 0.1679 - val_loss: 0.0021
#     Epoch 3/100
#     23/23 [==============================] - 0s 18ms/step - loss: 0.1532 - val_loss: 0.0064
#     Epoch 4/100
#     23/23 [==============================] - 0s 19ms/step - loss: 0.1408 - val_loss: 0.0038
#     Epoch 5/100
#     23/23 [==============================] - 0s 18ms/step - loss: 0.1186 - val_loss: 5.2907e-04
#     Epoch 6/100
#     23/23 [==============================] - 0s 18ms/step - loss: 0.1405 - val_loss: 0.0646
#     Epoch 7/100
#     23/23 [==============================] - 0s 17ms/step - loss: 0.1976 - val_loss: 0.0166
#     Epoch 8/100
#     23/23 [==============================] - 0s 18ms/step - loss: 0.2411 - val_loss: 5.4193e-04
#     Epoch 9/100
#     23/23 [==============================] - 0s 19ms/step - loss: 0.1535 - val_loss: 0.0187
#     Epoch 10/100
#     23/23 [==============================] - 0s 18ms/step - loss: 0.1332 - val_loss: 0.0060
#     Epoch 11/100
#     23/23 [==============================] - 0s 18ms/step - loss: 0.1680 - val_loss: 0.0024
#     Epoch 12/100
#     23/23 [==============================] - 0s 19ms/step - loss: 0.1346 - val_loss: 0.0267
#     Epoch 13/100
#     23/23 [==============================] - 0s 19ms/step - loss: 0.1363 - val_loss: 0.0172
#     Epoch 14/100
#     23/23 [==============================] - 0s 18ms/step - loss: 0.1303 - val_loss: 0.0117
#     Epoch 15/100
#     23/23 [==============================] - 0s 19ms/step - loss: 0.1448 - val_loss: 0.0240
#     1/1 [==============================] - 1s 1s/step
#     Mean Squared Error (MSE): 83.37
#     Root Mean Squared Error (RMSE): 9.13
#     Mean Absolute Error (MAE): 3.87
    


    
# #![png](output_18_2.png)
    


#     1/1 [==============================] - 0s 45ms/step
#     Future Revenue Forecast for Next Year:
#     [[20.887613]]
    
