import joblib
import pandas as pd
from pmdarima import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

df = pd.read_csv('./Backend/Data/complete_data.csv', parse_dates=['DATE'])
df.set_index('DATE', inplace=True)
ts = df['INFLRATE']

ts = pd.to_numeric(ts, errors='coerce').dropna()

model = ARIMA(ts, order=(2, 1, 2))
results = model.fit()

# model_filepath = './Backend/SavedModels/arima_model_212.pkl'
# joblib.dump(results, model_filepath)

forecast_steps = 5  
forecast = results.forecast(steps=forecast_steps)
print(forecast)

results.plot_diagnostics(figsize=(15, 12))
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
plot_acf(ts.diff().dropna(), ax=axes[0])
axes[0].set_title('Autocorrelation Function (ACF)')
plot_pacf(ts.diff().dropna(), ax=axes[1])
axes[1].set_title('Partial Autocorrelation Function (PACF)')
plt.show()
