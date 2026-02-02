from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.models import Model
from attention import BahdanauAttention

def attention_lstm(input_shape):
    inputs = Input(shape=input_shape)
    lstm_out, h, _ = LSTM(64, return_sequences=True, return_state=True)(inputs)
    context, _ = BahdanauAttention(32)(lstm_out, h)
    output = Dense(1)(context)
    return Model(inputs, output)

def baseline_lstm(input_shape):
    inputs = Input(shape=input_shape)
    x = LSTM(64)(inputs)
    output = Dense(1)(x)
    return Model(inputs, output)
