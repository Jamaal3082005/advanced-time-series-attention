import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from models import attention_lstm, baseline_lstm

# Load dataset
url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip"
data = pd.read_csv(url)

data['Date Time'] = pd.to_datetime(data['Date Time'])
data.set_index('Date Time', inplace=True)
data = data.resample('1H').mean().fillna(method='ffill')

target_col = 'T (degC)'
target_idx = data.columns.get_loc(target_col)

# Scale data
scaler = StandardScaler()
train_size = int(len(data) * 0.7)
train = scaler.fit_transform(data.iloc[:train_size])
val = scaler.transform(data.iloc[train_size:])

def create_sequences(data, window=24):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i+window])
        y.append(data[i+window, target_idx])
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(train)
X_val, y_val = create_sequences(val)

# Models
att_model = attention_lstm(X_train.shape[1:])
att_model.compile(optimizer='adam', loss='mse')

base_model = baseline_lstm(X_train.shape[1:])
base_model.compile(optimizer='adam', loss='mse')

# Train
att_model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=1)
base_model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=1)

# Evaluate
att_pred = att_model.predict(X_val)
base_pred = base_model.predict(X_val)

rmse_att = np.sqrt(mean_squared_error(y_val, att_pred))
rmse_base = np.sqrt(mean_squared_error(y_val, base_pred))

print("Attention LSTM RMSE:", rmse_att)
print("Baseline LSTM RMSE:", rmse_base)
