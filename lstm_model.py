from keras.models import Sequential
from keras.layers import LSTM, Dense
import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np

# Length of train dataset
TRAIN_LEN = 1000

# Creating a simple integer sequence to represent the dates in our train time series
train_set = np.array([i for i in range(1, TRAIN_LEN+1)]).reshape((1,TRAIN_LEN,1))

# Loading the closed price of the dataset as the main time-series data
df = pd.read_csv("../Microsoft_Stock.csv")

close_price_list = df["Close"].tolist()
close_series = pd.Series(close_price_list)

# Partitioning the data to train and test data
train_series = close_series[:TRAIN_LEN]
test_series = close_series[TRAIN_LEN:]

# Creating a simple integer sequence to represent the dates in our test time series
test_set = [i for i in range(TRAIN_LEN+1, TRAIN_LEN+1+len(test_series))]

model = Sequential()
model.add(LSTM(units=8, input_shape=(len(train_set),1), return_sequences=True))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
history = model.fit(train_set, np.array(train_series).reshape((1,TRAIN_LEN)), epochs=2, batch_size=1)
prediction = model.predict(np.array(test_set).reshape((1,len(test_series),1)))
# print((prediction))

pred_lstm = mean_squared_error(test_series, prediction.reshape((len(test_series),)))
print(f"LSTM's prediction error is: {pred_lstm}")