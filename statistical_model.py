import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# Length of train dataset
TRAIN_LEN = 1000

# Loading the closed price of the dataset as the main time-series data
df = pd.read_csv("Microsoft_Stock.csv")

close_price_list = df["Close"].tolist()
close_series = pd.Series(close_price_list)

# Partitioning the data to train and test data
train_series = close_series[:TRAIN_LEN]
test_series = close_series[TRAIN_LEN:]

# Training ARIMA on the train data
arima = ARIMA(train_series, order=(1,1,1))
trained_model = arima.fit()

# Predicting the future prices, using trained ARIMA model
pred = trained_model.predict(start=TRAIN_LEN, end=len(close_series)-1)

# Calculating the Error of ARIMA's prediction
pred_err = mean_squared_error(test_series, pred)
print(f"ARIMA's prediction error is: {pred_err}")