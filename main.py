import numpy as np
import pandas as pd
import datetime  
from sklearn.preprocessing import MinMaxScaler  # Scales data between 0 and 1
from sklearn.model_selection import train_test_split   
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt  
import yfinance as yf
import matplotlib.pyplot as plt


symbol = "AAPL" 
start_date = "2023-01-01"
end_date = "2024-01-01"
df = yf.download(symbol, start=start_date, end=end_date)
plt.plot(df['Close'])
plt.title(f"{symbol} Closing Price")
plt.show()


df.dropna(inplace=True)


# Only fit on the 'close' column
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df[['Close']])  



def create_sequences(data, seq_length=60):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i])  # past 60 days as input sequence 
        y.append(data[i])  # Next day's price output
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)


model = Sequential()
model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1))  # Predict one price
model.compile(optimizer='adam', loss='mean_squared_error')

history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Predict
predicted = model.predict(X_test)
predicted_rescaled = scaler.inverse_transform(predicted.reshape(-1, 1)).flatten()
real_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()


plt.figure(figsize=(10, 5))
plt.plot(real_rescaled, label='Actual Price')
plt.plot(predicted_rescaled, label='Predicted Price')
plt.title(f"{symbol} Stock Price Prediction")
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('LSTM Model Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.show()
