import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
import streamlit as st

start = "2020-01-01"
end = "2021-12-31"
scaler = MinMaxScaler(feature_range=(0,1))

st.title("Stock Trend Prediction")
user_input = st.text_input("Enter Stock Ticker", "LINK-USD")

df = data.DataReader(user_input, "yahoo", start, end)
df = df.reset_index()

#Describing Data
st.subheader("Data from 2020-2021")
st.write(df.describe())
#Visualization
st.subheader("Closing Price vs Time chart")
figure = plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(figure)
#100MA Plot
st.subheader("Closing Price vs Time chart with 100MA")
ma100 = df.Close.rolling(100).mean()
figure = plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(figure)
#200MA Plot
st.subheader("Closing Price vs Time chart with 100 & 200MA")
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
figure = plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(ma200)
plt.plot(df.Close)
st.pyplot(figure)

#Seperate DATA to training and testing

data_training = pd.DataFrame(df["Close"][0:int(len(df)*0.70)])
data_training_array = scaler.fit_transform(data_training)

data_testing = pd.DataFrame(df["Close"][int(len(df)*0.70):int(len(df))])
#data_testing_array = scaler.fit_transform(data_testing)


#Setup train data
#x_train = []
#y_train = []

#for i in range (100, data_training_array.shape[0]):
#    x_train.append(data_training_array[i-100: i])
#    y_train.append(data_training_array[i, 0])

#x_train, y_train = np.array(x_train), np.array(y_train)

#Load Model for streamlit app only
loadModel = load_model("keras_model.h5")

#ML Model - only include if you want to generate the model
#model = Sequential()
#1st layer
#model.add(LSTM(units = 50, activation = "relu", return_sequences=True, input_shape = (x_train.shape[1], 1)))
#model.add(Dropout(0.2))
#2nd layer
#model.add(LSTM(units = 60, activation = "relu", return_sequences=True))
#model.add(Dropout(0.3))
#3nd layer
#model.add(LSTM(units = 80, activation = "relu", return_sequences=True))
#model.add(Dropout(0.4))
#4th layer
#model.add(LSTM(units = 120, activation = "relu"))
#model.add(Dropout(0.5))
#Dense layer
#model.add(Dense(units = 1))
#Compile model
#model.compile(optimizer="adam", loss = "mean_squared_error")
#Fit model
#model.fit(x_train, y_train, epochs = 50)
#Save model
#model.save("keras_model.h5")

past_100_days = data_training.tail(100)
final_df = past_100_days.append(data_testing, ignore_index=True)
input_data = scaler.fit_transform(final_df)

#Setup test data
x_test = []
y_test = []

for i in range (100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

#Make predictions
#y_predicted = model.predict(x_test)
y_predicted = loadModel.predict(x_test)
dataScaler = scaler.scale_
scale_factor = 1/dataScaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

#Final Graph
st.subheader("Predictions vs Original")
figurePredicted = plt.figure(figsize=(12,6))
plt.plot(y_test, "b", label = "Original Price")
plt.plot(y_predicted, "r", label = "Predicted Price")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
st.pyplot(figurePredicted)