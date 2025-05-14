# ===== env_mgu.py =====
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras import layers, models

# ─── Seeds for Reproducibility ───────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ─── Custom MGU Layer ─────────────────────────────────────────────────────────
class MGULayer(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        dim = input_shape[-1]
        self.W_f = self.add_weight(name="W_f", shape=(dim, self.units), initializer="glorot_uniform", trainable=True)
        self.U_f = self.add_weight(name="U_f", shape=(self.units, self.units), initializer="orthogonal", trainable=True)
        self.b_f = self.add_weight(name="b_f", shape=(self.units,), initializer="zeros", trainable=True)
        self.W_c = self.add_weight(name="W_c", shape=(dim, self.units), initializer="glorot_uniform", trainable=True)
        self.U_c = self.add_weight(name="U_c", shape=(self.units, self.units), initializer="orthogonal", trainable=True)
        self.b_c = self.add_weight(name="b_c", shape=(self.units,), initializer="zeros", trainable=True)

    def call(self, inputs):
        bsz = tf.shape(inputs)[0]
        tsz = tf.shape(inputs)[1]
        h = tf.zeros((bsz, self.units))
        ta = tf.TensorArray(tf.float32, size=tsz)

        def step(t, h, ta):
            x_t = inputs[:, t, :]
            f = tf.sigmoid(x_t @ self.W_f + h @ self.U_f + self.b_f)
            c = tf.tanh(x_t @ self.W_c + (f * h) @ self.U_c + self.b_c)
            h_new = (1 - f) * h + f * c
            return t + 1, h_new, ta.write(t, h_new)

        _, h_final, ta = tf.while_loop(lambda t, *_: t < tsz, step, [tf.constant(0), h, ta])
        seq = ta.stack()  # (time, batch, units)
        return tf.transpose(seq, [1, 0, 2])  # (batch, time, units)

# ─── MGU Environment ──────────────────────────────────────────────────────────
class MGUEnvironment:
    def __init__(self, df, base_window=30):
        self.df = df.copy().sort_values("Date")
        self.base_window = base_window
        self.actions = [
            (32, 0.001, 30),
            (64, 0.001, 30),
            (32, 0.0005, 60),
            (64, 0.0005, 60),
            (32, 0.0001, 90),
        ]

    def create_sequences(self, data, w):
        X, y = [], []
        for i in range(len(data) - w):
            X.append(data[i:i + w])
            y.append(data[i + w])
        return np.array(X), np.array(y)

    def build_model(self, input_shape, units, lr):
        inp = layers.Input(shape=input_shape)
        x = MGULayer(units)(inp)
        x = layers.Lambda(lambda z: z[:, -1, :])(x)
        out = layers.Dense(1)(x)
        m = models.Model(inp, out)
        m.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="mse")
        return m

    def step(self, idx, return_model=False):
        units, lr, window = self.actions[idx]
        scaler = MinMaxScaler()
        self.df["Norm"] = scaler.fit_transform(self.df[["Price"]])
        series = self.df["Norm"].values

        X, y = self.create_sequences(series, window)
        split = int(0.8 * len(X))
        Xtr, yt = X[:split], y[:split]
        Xte, ye = X[split:], y[split:]
        Xtr = Xtr[..., None]
        Xte = Xte[..., None]

        model = self.build_model(Xtr.shape[1:], units, lr)
        model.fit(Xtr, yt, epochs=10, batch_size=32, verbose=0)

        yp = model.predict(Xte).flatten()
        ypr = scaler.inverse_transform(yp[:, None]).flatten()
        ytr = scaler.inverse_transform(ye[:, None]).flatten()

        mae = mean_absolute_error(ytr, ypr)
        rmse = np.sqrt(mean_squared_error(ytr, ypr))
        mape = np.mean(np.abs((ytr - ypr) / ytr)) * 100
        reward = -(mape + mae + rmse)

        return (reward, mape, mae, rmse, model, ytr, ypr) if return_model else (reward, mape, mae, rmse)

    def reset(self):
        return 0
