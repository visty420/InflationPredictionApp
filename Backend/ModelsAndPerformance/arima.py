import joblib
import pandas as pd
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
