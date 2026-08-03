"""
Baseline LSTM Autoencoder (Section 3.4.1) -- TensorFlow/Keras port of
../../../src/models/lstm_ae.py (PyTorch).

Encoder: LSTM -> last hidden state -> latent bottleneck
Decoder: RepeatVector -> LSTM -> TimeDistributed Dense -> reconstruction
"""

from __future__ import annotations

import keras
from keras import layers


def build_lstm_autoencoder(
    seq_len: int,
    feature_dim: int = 2,
    hidden_dim: int = 64,
    latent_dim: int = 32,
    dropout: float = 0.1,
) -> keras.Model:
    inputs = keras.Input(shape=(seq_len, feature_dim), name="input")

    # Encoder: LSTM returns only the final hidden state (return_sequences=False)
    x = layers.LSTM(hidden_dim, dropout=dropout, name="encoder_lstm")(inputs)
    latent = layers.Dense(latent_dim, name="to_latent")(x)

    # Decoder: project back up, repeat across time, decode, reconstruct
    x = layers.Dense(hidden_dim, name="from_latent")(latent)
    x = layers.RepeatVector(seq_len, name="repeat_vector")(x)
    x = layers.LSTM(hidden_dim, return_sequences=True, dropout=dropout, name="decoder_lstm")(x)
    outputs = layers.TimeDistributed(layers.Dense(feature_dim), name="output_projection")(x)

    return keras.Model(inputs, outputs, name="LSTM_Autoencoder")
