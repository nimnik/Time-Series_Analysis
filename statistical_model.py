import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import SimpleExpSmoothing
from sklearn.metrics import mean_squared_error

# Length of train dataset
TRAIN_LEN = 1000

# Loading the closed price of the dataset as the main time-series data
df = pd.read_csv("../Microsoft_Stock.csv")

close_price_list = df["Close"].tolist()
close_series = pd.Series(close_price_list)

# Partitioning the data to train and test data
train_series = close_series[:TRAIN_LEN]
test_series = close_series[TRAIN_LEN:]

# Training ARIMA on the train data
arima = ARIMA(train_series, order=(1,1,1))
arima_trained_model = arima.fit()

# Training Exponential Smoothing on the train data
exp_smoothing = SimpleExpSmoothing(train_series)
exp_trained_model = exp_smoothing.fit(smoothing_level=0.4, optimized=True)

# Predicting the future prices, using trained ARIMA model
pred_arima = arima_trained_model.predict(start=TRAIN_LEN, end=len(close_series)-1)

# Predicting the future prices, using trained Exponential Smoothing model
pred_exp = exp_trained_model.forecast(len(test_series))  #.predict(start=TRAIN_LEN, end=len(close_series)-1)

# Calculating the Error of ARIMA's prediction
pred_err_arima = mean_squared_error(test_series, pred_arima)
print(f"ARIMA's prediction error is: {pred_err_arima}")

# Calculating the Error of Exponential Smoothing's prediction
pred_err_exp = mean_squared_error(test_series, pred_exp)
print(f"Exponential Smoothing's prediction error is: {pred_err_exp}")