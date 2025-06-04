A brief description for each file:

* **statistical_models.py**: Includes a class that trains ARIMA and Exponential Smoothing models and a method that returns the predicted values

* **rf_model.py**: Includes a class that trains Random Forest model and a method that returns the predicted value

* **lstm_model.py**: Includes a class that trains LSTM model and a method that returns the predicted value

* **fine_tuning_models.ipynb**: Finds the error values corresponding to different hyperparameter values of ARIMA, Exponential Smoothing, and Random Forest

* **windowing_eval.ipynb**: For each candidate strategy finds the corresponding error values for ARIMA, Exponential Smoothing, and Random Forest models

* **lstm_eval.ipynb**: For each candidate strategy finds the corresponding error value for LSTM

* **visualization.ipynb**: Visualizes the Time Series data as well as Exponential Smoothing predictions

* **main.py**: The main code for using the models to predict the price of a given data point