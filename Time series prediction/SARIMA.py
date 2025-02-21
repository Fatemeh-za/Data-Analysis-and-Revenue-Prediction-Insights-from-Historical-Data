```python
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import warnings
import itertools
import numpy as np

warnings.filterwarnings("ignore")

# Ensure your DataFrame 'movies_df' is defined before this code

# Convert 'release_date' to datetime format and extract year
movies_df["release_date"] = pd.to_datetime(movies_df["release_date"], errors='coerce')
movies_df["release_year"] = movies_df["release_date"].dt.year

# Aggregate yearly revenue
yearly_revenue = movies_df.groupby("release_year")["revenue"].sum().reset_index()

# Set 'release_year' as the index
yearly_revenue.set_index("release_year", inplace=True)

# Scale the revenue data using MinMaxScaler
scaler = MinMaxScaler()
yearly_revenue_scaled = scaler.fit_transform(yearly_revenue.values.reshape(-1, 1))

# Plot the scaled yearly revenue data
plt.figure(figsize=(12, 6))
plt.plot(yearly_revenue.index, yearly_revenue_scaled, marker='o')
plt.title("Scaled Yearly Revenue")
plt.xlabel("Year")
plt.ylabel("Scaled Revenue")
plt.show()

# Split the data into training and testing sets
train_size = int(0.8 * len(yearly_revenue_scaled))
train_scaled = yearly_revenue_scaled[:train_size]
test_scaled = yearly_revenue_scaled[train_size:]

# Function to perform grid search for the best SARIMA parameters
def sarima_grid_search(train_data, p_values, d_values, q_values, P_values, D_values, Q_values, S_values):
    best_score, best_cfg = float("inf"), None
    for p, d, q in itertools.product(p_values, d_values, q_values):
        for P, D, Q, S in itertools.product(P_values, D_values, Q_values, S_values):
            order = (p, d, q)
            seasonal_order = (P, D, Q, S)
            try:
                model = SARIMAX(train_data, order=order, seasonal_order=seasonal_order)
                sarima_fit = model.fit(disp=False)
                predictions_scaled = sarima_fit.forecast(steps=len(test_scaled))
                predictions = scaler.inverse_transform(predictions_scaled.reshape(-1, 1))
                test = scaler.inverse_transform(test_scaled.reshape(-1, 1))
                mse = mean_squared_error(test, predictions)
                if mse < best_score:
                    best_score, best_cfg = mse, (order, seasonal_order)
                    print(f"SARIMA{order}x{seasonal_order} - MSE: {mse:.3f}")
            except:
                continue
    return best_cfg

# Grid search parameters
p_values = [0, 1, 2]
d_values = [0, 1]
q_values = [0, 1, 2]
P_values = [0, 1, 2]
D_values = [0, 1]
Q_values = [0, 1, 2]
S_values = [12]

# Perform grid search
best_cfg = sarima_grid_search(train_scaled, p_values, d_values, q_values, P_values, D_values, Q_values, S_values)
print(f"Best SARIMA configuration: {best_cfg}")

# Fit the best SARIMA model
best_order, best_seasonal_order = best_cfg
model = SARIMAX(train_scaled, order=best_order, seasonal_order=best_seasonal_order)
sarima_fit = model.fit(disp=False)

# Make predictions
predictions_scaled = sarima_fit.forecast(steps=len(test_scaled))

# Reverse scaling for predictions and actual test data
predictions = scaler.inverse_transform(predictions_scaled.reshape(-1, 1))
test = scaler.inverse_transform(test_scaled.reshape(-1, 1))

# Evaluate the model
mse = mean_squared_error(test, predictions)
rmse = np.sqrt(mse)
mae = mean_absolute_error(test, predictions)

print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(yearly_revenue.index[:train_size], scaler.inverse_transform(train_scaled), label="Training Data")
plt.plot(yearly_revenue.index[train_size:], test, label="Actual Revenue")
plt.plot(yearly_revenue.index[train_size:], predictions, label="Predicted Revenue", linestyle='--')
plt.title("Time Series Revenue Prediction with SARIMA")
plt.xlabel("Year")
plt.ylabel("Revenue")
plt.legend()
plt.show()

# Forecast future revenue for the next 5 years
future_forecast_scaled = sarima_fit.forecast(steps=5)
future_forecast = scaler.inverse_transform(future_forecast_scaled.reshape(-1, 1))

print("Future Revenue Forecast for Next 5 Years:")
print(future_forecast)



    
# ![png](output_17_0.png)
    


#     SARIMA(0, 0, 0)x(0, 0, 0, 12) 
#     SARIMA(0, 0, 0)x(0, 0, 1, 12) 
#     SARIMA(0, 0, 0)x(0, 0, 2, 12) 
#     SARIMA(0, 0, 0)x(0, 1, 0, 12) 
#     SARIMA(0, 0, 0)x(0, 1, 1, 12) 
#     SARIMA(0, 0, 0)x(0, 1, 2, 12) 
#     SARIMA(0, 0, 0)x(1, 1, 0, 12) 
#     SARIMA(0, 0, 0)x(1, 1, 1, 12) 
#     SARIMA(0, 0, 0)x(1, 1, 2, 12) 
#     SARIMA(0, 0, 0)x(2, 1, 1, 12) 
#     SARIMA(0, 0, 0)x(2, 1, 2, 12) 
#     SARIMA(0, 1, 0)x(0, 1, 0, 12) 
#     SARIMA(1, 1, 2)x(0, 0, 0, 12) 
#     Best SARIMA configuration: ((1, 1, 2), (0, 0, 0, 12))
   
    


    
# ![png](output_17_2.png)
    


#     Future Revenue Forecast for Next 5 Years:
#     [[6.57776934e+09]
#      [7.01961114e+09]
#      [7.45725640e+09]
#      [7.89074499e+09]
#      [8.32011638e+09]]
    
