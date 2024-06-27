import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

# Define function to load data
@st.cache
def load_data():
    df = pd.read_csv("BBCA.JK.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    return df

# Define function to split data
def split_data(df, train_size):
    size = int(len(df) * train_size)
    train, test = df.iloc[0:size], df.iloc[size:len(df)]
    return train, test

# Define function to prepare target variables
def split_target(df, look_back=1):
    X, y = [], []
    for i in range(len(df) - look_back):
        a = df[i:(i + look_back), 0]
        X.append(a)
        y.append(df[i + look_back, 0])
    return np.array(X), np.array(y)

# Streamlit app
st.title('Stock Price Prediction with LSTM')

# Load data
df = load_data()

# Display dataset
st.subheader('Dataset')
st.write(df.head())

# Plot High and Low prices
st.subheader('High and Low Prices')
sns.set_style('darkgrid')
fig, ax = plt.subplots(figsize=(15, 7))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=12))
x_dates = df.index.values
ax.plot(x_dates, df['High'], label='High')
ax.plot(x_dates, df['Low'], label='Low')
ax.set_xlabel('Date')
ax.set_ylabel('Price (Rp)')
ax.set_title("Stock Price\nUnilever Indonesia", fontsize=20)
ax.legend()
plt.gcf().autofmt_xdate()
st.pyplot(fig)

# Plot Open and Close prices
st.subheader('Open and Close Prices')
fig, ax = plt.subplots(figsize=(15, 7))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=12))
ax.plot(x_dates, df['Open'], label='Open')
ax.plot(x_dates, df['Close'], label='Close')
ax.set_xlabel('Date')
ax.set_ylabel('Price (Rp)')
ax.set_title("Stock Price\nUnilever Indonesia", fontsize=20)
ax.legend()
plt.gcf().autofmt_xdate()
st.pyplot(fig)

# User input for training size
train_size = st.slider('Select training data size (in %)', 50, 90, 80) / 100

# Feature Scaling
ms = MinMaxScaler()
df['Close_ms'] = ms.fit_transform(df[['Close']])

# Split data
train, test = split_data(df['Close_ms'], train_size)

# Plot training and testing data
st.subheader('Training and Testing Data')
fig, ax = plt.subplots(figsize=(15, 7))
ax.plot(train, label='Training')
ax.plot(test, label='Testing')
ax.set_title('Stock Price\nUnilever Indonesia\nTraining & Testing Split', fontsize=20)
ax.legend()
st.pyplot(fig)

# Prepare target variables
X_train, y_train = split_target(train.values.reshape(len(train), 1))
X_test, y_test = split_target(test.values.reshape(len(test), 1))

X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# Build model
model = Sequential([
    LSTM(128, input_shape=(1, 1), return_sequences=True),
    Dropout(0.2),
    LSTM(64),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, metrics=["mae"], loss=tf.keras.losses.Huber())

# Training
if st.button('Train Model'):
    with st.spinner('Training model...'):
        history = model.fit(X_train, y_train, epochs=200, batch_size=32, validation_data=(X_test, y_test), shuffle=False)
    
    # Plot Loss and MAE
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.plot(history.history['loss'])
    ax1.plot(history.history['val_loss'])
    ax1.legend(['Loss', 'Val Loss'])
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Loss', fontsize=20)
    ax2.plot(history.history['mae'])
    ax2.plot(history.history['val_mae'])
    ax2.legend(['MAE', 'Val MAE'])
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Mean Absolute Error', fontsize=12)
    ax2.set_title('Mean Absolute Error', fontsize=20)
    st.pyplot(fig)

    # Predict
    pred = model.predict(X_test)
    y_pred = np.array(pred).reshape(-1)

    # Plot actual vs predicted prices
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.plot(test.index[:-1], y_test, color='blue', label='Actual')
    ax.plot(test.index[:-1], y_pred, color='red', label='Predicted')
    ax.text(test.index[100], 0.45, f"MAE = {mean_absolute_error(y_test, y_pred)}", style='italic', bbox={'facecolor': 'orange', 'alpha': 0.5, 'pad': 10})
    ax.set_xlabel('Date')
    ax.set_ylabel('Stock Price')
    ax.set_title('Stock Price Prediction\nUnilever Indonesia\nLSTM', fontsize=20)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gcf().autofmt_xdate()
    ax.legend()
    st.pyplot(fig)

    # Evaluation
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    st.write('MAE: ', mae)
    st.write('RSME: ', rmse)
    st.write('MAPE: ', mape)

    # Plot all data with predicted result
    y_pred_original = ms.inverse_transform(np.array(y_pred).reshape(-1, 1)).flatten()
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.plot(df.index, df['Close'], color='blue', label='Actual')
    ax.plot(df.index[len(train.index):-1], y_pred_original, color='red', label='Predicted')
    ax.set_xlabel('Time')
    ax.set_ylabel('Stock Price')
    ax.set_title('Stock Price Prediction\nUnilever Indonesia\nLSTM', fontsize=20)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=12))
    plt.xticks(rotation=30)
    ax.legend()
    st.pyplot(fig)