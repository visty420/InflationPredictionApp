import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

df = pd.read_csv('./Backend/Data/complete_data.csv', parse_dates=['DATE'])
df.set_index('DATE', inplace=True) 
ts = df['INFLRATE']

ts = pd.to_numeric(ts, errors='coerce').dropna()

adf_result = adfuller(ts)
print(f'ADF Statistic: {adf_result[0]}')
print(f'p-value: {adf_result[1]}')

if adf_result[1] > 0.05:
    print("Seria este non-staționară, diferențiere necesară.")
    ts_diff = ts.diff().dropna()
    ts_to_model = ts_diff
else:
    ts_to_model = ts

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 4))
plot_acf(ts_to_model, ax=ax1, lags=20)
plot_pacf(ts_to_model, ax=ax2, lags=20, method='ols')
plt.show()


p, d, q = 1, 1, 1  
model = ARIMA(ts, order=(p, d, q))
results = model.fit()

model_filepath = './Backend/SavedModels/arima_model.pkl'
results.save(model_filepath)


forecast = results.forecast(steps=5)
print(forecast)

results.plot_diagnostics(figsize=(15, 12))
plt.show()
