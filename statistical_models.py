from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import SimpleExpSmoothing

class Stat_Models:
    def __init__(self, train_data, order=(0,1,0), pred_size=1):
        # Creating a variable for train data, to later use in process method
        self.train = train_data

        # Number of future values to be predicted
        self.pred_size = pred_size

        # Initializing ARIMA on the train data
        self.arima = ARIMA(endog=train_data, order=order)

        # Initializing Exponential Smoothing on the train data
        self.exp_smooth = SimpleExpSmoothing(train_data)
        
    def process(self, smoothing_level=0.8):
        # Training ARIMA
        arima_trained_model = self.arima.fit()

        # Training Exponential Smoothing
        exp_trained_model = self.exp_smooth.fit(smoothing_level=smoothing_level, optimized=False)

        # Predicting the future prices, using trained ARIMA model
        pred_arima = arima_trained_model.predict(start=len(self.train), end=len(self.train)+self.pred_size-1)

        # Predicting the future prices, using trained Exponential Smoothing model
        pred_exp = exp_trained_model.forecast(self.pred_size)
        
        return pred_arima, pred_exp

