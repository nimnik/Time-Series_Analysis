import pandas as pd
import warnings

from statistical_models import Stat_Models
from rf_model import RF_Model

warnings.filterwarnings('ignore')

# Length of train data
TRAIN_LEN = 7

# Number of values to be predicted
PRED_SIZE = 1

# Loading the closed price of the dataset as the main time-series data
df = pd.read_csv("../Microsoft_Stock.csv")

if ("Close" in df.columns):
    close_price_series = df["Close"]
elif ("close" in df.columns):
    close_price_series = df["close"]
else:
    print("Close prices are not available as a column in the dataset.")
    exit()

# Receiving data point input
i = int(input(f"Please enter a number from {TRAIN_LEN} to {len(close_price_series)-1}, as the Close price to predict:\n"))

# Extract train data
train_data = close_price_series[(i-TRAIN_LEN):i]

# Receiving Model Input
model_case = input(f"Please enter 1 for using the best model for prediction, and 2 for providing the predictions of all models:\n")

# Calculating predictions of the statistical models
stat_model = Stat_Models(train_data)
arima_pred, exp_pred = stat_model.process()
arima_pred = float(arima_pred)
exp_pred = float(exp_pred)

if (model_case == "1"):
    print(f"The price predicted by Exponential Smoothing is: {exp_pred}")

elif (model_case == "2"):
    from lstm_model import LSTM_Model
    from tensorflow import get_logger
    get_logger().setLevel('ERROR')


    # Calculating predictions of Random Forest model
    rf = RF_Model(train_data)
    rf_pred = float(rf.process())

    # Calculating predictions of LSTM model
    target_data = close_price_series[i:(i+PRED_SIZE)]
    lstm = LSTM_Model(train_data, target_data)
    lstm_pred = float(lstm.process())


    print(f"The price predicted by ARIMA is: {arima_pred}")
    print(f"The price predicted by Exponential Smoothing is: {exp_pred}")
    print(f"The price predicted by Random Forest is: {rf_pred}")
    print(f"The price predicted by LSTM is: {lstm_pred}")

else:
    print("Invalid Input: The input should either be 1 or 2.")
