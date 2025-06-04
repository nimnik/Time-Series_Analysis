from keras.models import Sequential # type: ignore
from keras.layers import LSTM, Dense # type: ignore

class LSTM_Model:
    def __init__(self, train_data, target_data, pred_size=1):
        # Creating a variable for train data, to later use in process method
        self.train = train_data.to_numpy().reshape(1, len(train_data), 1)

        # Creating a variable for target data, to later use in process method
        self.target = target_data.to_numpy().reshape(1, len(target_data), 1)

        # Number of future values to be predicted
        self.pred_size = pred_size
        
    def process(self):
        # Training LSTM on the train data
        lstm_model = Sequential()
        lstm_model.add(LSTM(units=8, input_shape=(len(self.train), 1)))
        lstm_model.add(Dense(self.pred_size))
        lstm_model.compile(loss='mean_squared_error', optimizer='adam')

        # Predicting the future prices, using trained LSTM model
        lstm_model.fit(self.train, self.target, epochs=2, batch_size=1, verbose=0)
        pred_lstm = lstm_model.predict(self.train, verbose=0)
        
        return pred_lstm
    
