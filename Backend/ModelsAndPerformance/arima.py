import pandas as pd
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# Load your dataset
df = pd.read_csv('./Backend/Data/augmented_economic_data.csv', parse_dates=['DATE'])
ts = df['INFLRATE'] 
ts_diff = ts.diff().dropna()
# Convert to numeric and drop NaN values
ts = pd.to_numeric(ts, errors='coerce').dropna()

lag_acf = acf(ts_diff, nlags=20)
lag_pacf = pacf(ts_diff, nlags=20, method='ols')

# Check for stationarity with the Dickey-Fuller test
adf_result = adfuller(ts)
print(f'ADF Statistic: {adf_result[0]}')
print(f'p-value: {adf_result[1]}')

#calculate ACF and PACF for lag value selection
acf_values = acf(ts, nlags=20)
pacf_values = pacf(ts, nlags=20)

# Plot ACF
plt.figure()
plt.plot(acf_values)
plt.title('Autocorrelation Function')

# Plot PACF
plt.figure()
plt.plot(pacf_values)
plt.title('Partial Autocorrelation Function')

plt.show()

# Fit ARIMA model based on determined p, d, q values after analyzing ACF and PACF
# Replace these placeholder values with the actual values you determined
p, d, q = 1, 1, 1
model = ARIMA(ts, order=(p, d, q))
results = model.fit()

# Forecast
forecast = results.forecast(steps=5)
print(forecast)

# Plot diagnostics
results.plot_diagnostics(figsize=(15, 12))
plt.show()
