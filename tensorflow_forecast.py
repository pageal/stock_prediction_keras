######################################
# Based on the code from this video: "Predicting Stock Prices in Python"
# https://www.youtube.com/watch?v=PuZY9q-aKLw
# NeuralLine Youtube channel: https://www.youtube.com/channel/UC8wZnXYK_CGKlBcZp-GxYPA
#
# pageal's comments and possible changes are based on:
# 1) Stanford University. Machine Learning course (https://www.coursera.org/learn/machine-learning/home/welcome)
######################################

######################################
#INSTALLING
######################################

######################################
## clear python environment (PyCharm Virtual Environment)
## in command-line prompt (cmd.exe) executed as Admin
## run these commands:
#pip install numpy
#pip install matplotlib
#pip install pandas
#pip install pandas-datareader
#pip install tensorflow
#pip install scikit-learn
#for candlestick
#pip install mplfinance
######################################

######################################
## Anaconda3/Conda environment
#Install Anaconda with python x64 ver 3.6-3.8
#https://www.anaconda.com/products/individual

#in Anaconda command prompt running as Admin install tensorflow package
#https://docs.anaconda.com/anaconda/user-guide/tasks/tensorflow/
#conda create -n tf tensorflow
#conda activate tf
#conda install -c conda-forge keras
#conda install -c conda-forge scikit-learn
#conda install -c conda-forge google-api-python-client
## nstall whatever package PyCharm doesn't see from within the environment (File/Settings/Project <name>/Python Interpreter)


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

company = "INTC"
training_years_back = 5
prediction_days = 7

class shape_table:
    SHAPE_ROWS = 0
    SHAPE_COLUMNS = 1

the_year = dt.datetime.now().year
start = dt.datetime(the_year - training_years_back - 1,1,1)
end = dt.datetime(the_year - 1,1,1)


######################################
# BEGIN: INPUT DATA EXTRACTION, SCALING AND FORMATTING
#
# read stock historical prices from yahoo finance
data = web.DataReader(company, 'yahoo', start, end)
# use "High" price data COLUMN for training data
historical_stock_prices = data['High']

# data[].values is represented by 1xlen(data[]) array
# reshape(-1,1) reshapes data[] from 1xlen(data[]) to len(data[])x1 vector (array with only 1 column)
# this will allow to use this a a parameter for linear algebra
historical_stock_prices_reshaped = historical_stock_prices.values.reshape(-1, 1)  #see the note below for meaning of -1
# NOTE: https://numpy.org/doc/stable/reference/generated/numpy.reshape.html
# One shape dimension can be -1. In this case, the value is inferred from the length of the array and remaining dimensions.
#
# @pageal (based on ): MODEL INPUTS RESCALING PHASE
# for algorithm to converge QUICKER, function parameters are better to be on SIMILAR scale
# so for rescaling to the range of 0...1, each parameter is divided by its MAX value
#   X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
#   X_scaled = X_std * (max - min) + min
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html#sklearn.preprocessing.MinMaxScaler
# https://scikit-learn.org/stable/modules/preprocessing.html
scaler = MinMaxScaler(feature_range=(0,1))
historical_stock_prices_scaled = scaler.fit_transform(historical_stock_prices_reshaped)
#
# END: INPUT DATA EXTRACTION, SCALING AND FORMATTING
######################################


######################################
# BEGIN: PREPARING OF TRAINING DATA FOR LTSM-NEURAL-NETWORK (Long-Short Term Memory)
# split scaled_data into 2 arrays:
#  - training sets of prices (prediction_days long) - by sliding window of prediction_days
#  - next day prices
training__input_prices = []    #array of input data arrays
training__next_day_price = []  # array of scalars representing input data starting from prediction_days'th index
# training_results[i] values represent a stock price next day after stock prices were as at training__input_prices[i]'s data set
# ,where for training__input_prices[i] is an array of the stock prices for previous prediction_days days
#               EXAMPLE for scaled_data=[0.5, 0.2, 0.8, 0.6, 0.1, 0.3, 0.4]; prediction_days=4:
#                   training__input_prices = [
#                               [0.5, 0.2, 0.8, 0.6],
#                               [0.2, 0.8, 0.6, 0.1],
#                               [0.8, 0.6, 0.1, 0.3]]
#                   training__next_day_price = [
#                                0.2,
#                                0.8,
#                                0.4
#                                ]
#training data set length should be SAME as the lengt of data set we are going to use to predict folllowing day
training_data_set_len = prediction_days
training_samples_num = len(historical_stock_prices_scaled)-prediction_days-1
for i in range(0, training_samples_num):
    training__input_prices.append(historical_stock_prices_scaled[i:i+training_data_set_len, 0])
    training__next_day_price.append(historical_stock_prices_scaled[i+training_data_set_len, 0])

#convert into numpy arrays
training__input_prices = np.array(training__input_prices)
training__next_day_price = np.array(training__next_day_price)

#reshape for neural network
#training__input_prices.shape[shape_table.SHAPE_ROWS] - num of rows
#training__input_prices.shape[shape_table.SHAPE_COLUMNS] - num of columns
training__input_prices = np.reshape(training__input_prices,
                                    (training__input_prices.shape[shape_table.SHAPE_ROWS],
                                     training__input_prices.shape[shape_table.SHAPE_COLUMNS],
                                     1))

###########################################
#Build the neural model
model = Sequential()

# LTSM neural network- https://en.wikipedia.org/wiki/Long_short-term_memory
#model.add(LSTM(units=60,return_sequences=True, input_shape=(training__input_prices.shape[shape_table.SHAPE_COLUMNS], 1)))
model.add(LSTM(units=7,return_sequences=True, input_shape=(training__input_prices.shape[shape_table.SHAPE_COLUMNS], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=7,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=7))
model.add(Dropout(0.2))
#model.add(LSTM(units=58,return_sequences=True))
#model.add(Dropout(0.2))
#model.add(LSTM(units=67))
#model.add(Dropout(0.2))
model.add(Dense(units=1))  # Prediction of the next closing value

model.compile(optimizer='adam', loss='mean_squared_error')
#perform actual training
model.fit(training__input_prices, training__next_day_price, epochs=5, shuffle=True, batch_size=prediction_days)


######################################################
# Model accuracy TEST on existing data
#Prepare test data
test_start = dt.datetime(the_year-1,1,1)
test_end = dt.datetime.now()
test_days = 30

test_data = web.DataReader(company, "yahoo", test_start, test_end)
actual_prices = test_data['High'].values
actual_prices_reshaped = actual_prices.reshape(-1, 1)
actual_prices_scaled = scaler.fit_transform(actual_prices_reshaped)


# Make predictions on Test data
test__input_prices = []
training_data_set_len = prediction_days
training_samples_num = len(actual_prices)-prediction_days-1
for i in range(0, training_samples_num):
    test__input_prices.append(actual_prices_scaled[i:i+training_data_set_len, 0])

test__input_prices = np.array(test__input_prices)
test__input_prices = np.reshape(test__input_prices,
                                (test__input_prices.shape[shape_table.SHAPE_ROWS],
                                 test__input_prices.shape[shape_table.SHAPE_COLUMNS],
                                 1))

predicted_prices = model.predict(test__input_prices)
predicted_prices = scaler.inverse_transform(predicted_prices)

# Plot the test predictions
plt.plot(actual_prices, color="black", label=f"Actual {company} price")
plt.plot(predicted_prices, color='green', label=f"Predicted {company} price")
plt.title(f"{company} Share Price")
plt.xlabel('Time')
plt.ylabel(f"{company} Share Price")
plt.legend()
plt.show()


######################################################
#Predict Next Day Price
real_data = [actual_prices_scaled[len(actual_prices_scaled) - prediction_days:, 0]]
real_data = np.array(real_data)
real_data_reshaped = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

prediction_scaled = model.predict(real_data_reshaped)
prediction = scaler.inverse_transform(prediction_scaled)
print(f"Prediction: {prediction}")

print("end")





