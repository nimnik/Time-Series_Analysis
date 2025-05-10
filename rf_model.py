import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Length of train dataset
TRAIN_LEN = 1000

# Creating a simple integer sequence to represent the dates in our train time series
train_set = np.array([i for i in range(1, TRAIN_LEN+1)])

# Loading the closed price of the dataset as the main time-series data
df = pd.read_csv("../Microsoft_Stock.csv")

close_price_list = df["Close"].tolist()
close_series = pd.Series(close_price_list)

# Partitioning the data to train and test data
train_series = close_series[:TRAIN_LEN]
test_series = close_series[TRAIN_LEN:]

# Defining and Running Random Forest Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
print(train_set.shape, np.array(train_series).shape)

model.fit(train_set.reshape(-1, 1), np.array(train_series).reshape(-1, 1))


# Creating a simple integer sequence to represent the dates in our test time series
test_set = np.array([i for i in range(TRAIN_LEN, len(close_price_list))])

# Predicting the test values
predictions = model.predict(test_set.reshape(-1, 1))

pred_rf = mean_squared_error(np.array(test_series), predictions)
print(f"Random Forest's prediction error is: {pred_rf}")