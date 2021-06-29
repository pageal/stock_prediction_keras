######################################
# Based on the code at:
# https://www.analyticsvidhya.com/blog/2020/10/reinforcement-learning-stock-price-prediction/
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

import sys
import math
import random
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

def sigmoid(x):
    return 1/(1+math.exp(-x))

def formatPrice(n):
    return("-Rs." if n<0 else "Rs.")+"{0:.2f}".format(abs(n))

class ReinforcentModel():
    def __init__(self, stock_ticker, state_size):
        self.model_name = "{}_model_{}".format(stock_ticker, dt.datetime.now().strftime("%m_%d_%Y__%H_%M_%S"))
        self.stock_ticker = stock_ticker
        self.state_size = state_size  # training set size

        self.batch_size = 32
        self.episode_count = 5
        self.training_years_back = 1
        the_year = dt.datetime.now().year
        self.training_end_date = dt.datetime(the_year, 1, 1)
        # end = dt.datetime.now()
        #training_span = dt.timedelta(days=365 * self.training_years_back)
        training_span = dt.timedelta(days=60)
        self.training_start_date = self.training_end_date - training_span

        self.current_element_idx = self.state_size
        self.action_size = 3  # sit, buy, sell

        self.inventory = []
        self.is_eval = True
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

    def getState(self, historical_stock_prices, current_element_idx):
        data_set_start = current_element_idx - (self.state_size+1) + 1
        data_set = historical_stock_prices[data_set_start:current_element_idx + 1]
        res = []
        for i in range(self.state_size):
            res.append(sigmoid(data_set[i + 1] - data_set[i]))
        return np.array([res])

    # States – Data values for training
    def getNextState(self):
        res = self.getState(self.historical_stock_prices,self.current_element_idx)
        self.current_element_idx = self.current_element_idx +1
        return res

    def _model(self):
        # Build the neural model
        model = Sequential()
        model.add(Dense(units=64, input_dim=self.state_size, activation="relu"))
        model.add(Dense(units=32, activation="relu"))
        model.add(Dense(units=8, activation="relu"))
        model.add(Dense(self.action_size, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(learning_rate=0.001))
        return model

    def act(self, state):
        if not self.is_eval and random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        options = self.model.predict(state)
        return np.argmax(options[0])

    def expReplay(self, batch_size):
        mini_batch = []
        memory_len = len(self.memory)
        for i in range(memory_len - batch_size + 1, memory_len):
            mini_batch.append(self.memory[i])

        for state, action, reward, next_state, done in mini_batch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def TrainAgent(self):
        self.is_eval = False
        self.model = load_model(self.model_name) if self.is_eval else self._model()
        # read stock historical prices from yahoo finance
        raw_data = web.DataReader(self.stock_ticker, 'yahoo', self.training_start_date, self.training_end_date)
        # use "High" price data COLUMN for training data
        self.historical_stock_prices = raw_data['High']
        self.memory = deque(maxlen=len(self.historical_stock_prices)+1)

        max_data_index = len(self.historical_stock_prices) - 2
        for episode in range(self.episode_count):
            print("Episode " + str(episode) + "/" + str(self.episode_count))
            state = self.getNextState()
            total_profit = 0
            self.inventory = []
            self.current_element_idx = self.state_size
            while self.current_element_idx < max_data_index:
                next_state = self.getNextState()
                reward = 0
                action = self.act(state)
                if action == 1:  # buy
                    self.inventory.append(self.historical_stock_prices[self.current_element_idx])
                    print("Training Buy: " + formatPrice(self.historical_stock_prices[self.current_element_idx]))
                elif action == 2 and len(self.inventory) > 0:  # sell
                    bought_price = window_size_price = self.inventory.pop(0)
                    reward = max(self.historical_stock_prices[self.current_element_idx] - bought_price, 0)
                    total_profit += self.historical_stock_prices[self.current_element_idx] - bought_price
                    print("Training Sell: " + formatPrice(self.historical_stock_prices[self.current_element_idx]) + " | Profit: " + formatPrice(self.historical_stock_prices[self.current_element_idx] - bought_price))
                done = True if (max_data_index - self.current_element_idx)==1 else False
                self.memory.append((state, action, reward, next_state, done))
                state = next_state
                if len(self.memory) > self.batch_size:
                    self.expReplay(self.batch_size)
            print("--------------------------------")
            print("Total Profit: " + formatPrice(total_profit))
            print("--------------------------------")
            self.model.save(self.model_name)
        self.is_eval = True

company = "INTC"
prediction_days = 7
the_model = ReinforcentModel(company, state_size=prediction_days)
the_model.TrainAgent()


######################################################
# Model accuracy TEST on existing data
#Prepare test data
#test_start = dt.datetime(the_year-1,1,1)
test_end = dt.datetime.now()
test_span = dt.timedelta(days=90)
test_start = test_end - test_span

test_data = web.DataReader(company, "yahoo", test_start, test_end)
actual_prices = test_data['High'].values


# Make predictions on Test data
test__input_prices_scaled = []
test__actual_prices = []
test__buy_prices = []
test__sell_prices = []
test__profit = []
test__total_profit = []
test_data_set_len = prediction_days

total_profit = 0
max_data_index = len(actual_prices) - 2
the_model.inventory = []
state = the_model.getState(actual_prices, test_data_set_len)
for i in range(test_data_set_len+1, max_data_index):
    next_state = the_model.getState(actual_prices, i+1)
    test__actual_prices.append(actual_prices[i])
    #predict
    action = the_model.act(state)
    reward = 0
    test__buy_prices.append(None)
    test__sell_prices.append(None)
    test__profit.append(0)
    test__total_profit.append(0)
    if action == 1: # buy
        the_model.inventory.append(actual_prices[i])
        test__buy_prices[len(test__actual_prices)-1] = actual_prices[i]
        print("Buy: " + formatPrice(actual_prices[i]))
    elif action == 2 and len(the_model.inventory) > 0: # sell
        bought_price = the_model.inventory.pop(0)
        reward = max(actual_prices[i] - bought_price, 0)
        profit = actual_prices[i] - bought_price
        total_profit += profit
        test__sell_prices[len(test__actual_prices)-1] = actual_prices[i]
        test__profit[len(test__actual_prices)-1] = profit
        test__total_profit[len(test__actual_prices)-1] = total_profit
        print("Sell: " + formatPrice(actual_prices[i]) + " | Profit: " + formatPrice(actual_prices[i] - bought_price))
    done = True if max_data_index - i == 1 else False
    the_model.memory.append((state, action, reward, next_state, done))
    state = next_state

print("--------------------------------")
print(company + " Total Profit: " + formatPrice(total_profit))
print("--------------------------------")
print ("Total profit is: " + formatPrice(total_profit))



# Plot the test predictions
index_arr = []
for i in range(0,len(test__actual_prices)):
    index_arr.append(i)
index_arr = np.array(index_arr)

#plt.vlines(x=index_arr, ymin=0, ymax=test__actual_prices, color='firebrick', alpha=0.7, linewidth=2)
plt.grid(which="both", axis="x")
plt.scatter(x=index_arr, y=test__buy_prices, s=15, color='firebrick', alpha=0.7, label=f"Buy {company} price")
plt.plot(test__actual_prices, color="black", linestyle='solid', solid_joinstyle='round', linewidth=1, label=f"Actual {company} price")
plt.scatter(x=index_arr, y=test__sell_prices, s=10, color='green', alpha=0.7,  label=f"Sell {company} price")
#plt.plot(predicted_prices, color='green', linestyle='solid', label=f"Predicted {company} price")
plt.vlines(x=index_arr, ymin=np.amin(test__actual_prices)-2, ymax=np.amax(test__actual_prices)+2, color='gray', alpha=0.7, linewidth=1, linestyles='dashed')
plt.bar(x=index_arr, height=test__profit, color='blue', width=.5)
plt.bar(x=index_arr, height=test__total_profit, color='green', width=.5)

plt.title(f"{company} Share Price")
plt.xlabel('Time')
plt.ylabel(f"{company} Share Price")
plt.legend()
plt.show()

print("end")





