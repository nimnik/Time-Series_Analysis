from sklearn.ensemble import RandomForestRegressor
import numpy as np

class RF_Model:
    def __init__(self, train_data, order=(0,1,0), pred_size=1):
        # Creating a variable for train data, to later use in process method
        self.train = train_data

        # Number of future values to be predicted
        self.pred_size = pred_size
        
    def process(self, n_est=150, random_st=42):
        # Specifying Train and Test Inputs
        train_input = np.arange(len(self.train))
        test_input = np.arange(len(self.train),(len(self.train)+self.pred_size))

        # Training Random Forest on the train data
        rf = RandomForestRegressor(n_estimators=n_est, random_state=random_st)
        rf.fit(train_input.reshape(-1,1), self.train)

        # Predicting the future prices, using trained Random Forest model
        pred_rf = rf.predict(test_input.reshape(-1, 1))
        
        return pred_rf

