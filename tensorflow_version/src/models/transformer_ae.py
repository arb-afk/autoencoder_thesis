"""
Challenger Transformer Autoencoder (Section 3.4.2) -- TensorFlow/Keras port
of ../../../src/models/transformer_ae.py (PyTorch).

Positional encoding -> Multi-Head Self-Attention blocks -> FFNN -> reconstruction.
"""

from __future__ import annotations

import numpy as np
import keras
import tensorflow as tf
from keras import layers


class PositionalEncoding(layers.Layer):
    """Sinusoidal positional encoding (Vaswani et al., 2017)."""

    def __init__(self, d_model: int, max_len: int = 500, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.max_len = max_len

        position = np.arange(max_len)[:, None].astype(np.float32)
        div_term = np.exp(np.arange(0, d_model, 2).astype(np.float32) * (-np.log(10000.0) / d_model))
        pe = np.zeros((max_len, d_model), dtype=np.float32)
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        self.pe = tf.constant(pe[np.newaxis, ...], dtype=tf.float32)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + self.pe[:, :seq_len, :]

    def get_config(self):
        config = super().get_config()
        config.update({"d_model": self.d_model, "max_len": self.max_len})
        return config


def transformer_encoder_block(
    x,
    d_model: int,
    nhead: int,
    dim_feedforward: int,
    dropout: float,
    name_prefix: str,
):
    """One post-norm Transformer encoder block (self-attn + FFN, residual + LayerNorm)."""
    attn_out = layers.MultiHeadAttention(
        num_heads=nhead,
        key_dim=d_model // nhead,
        dropout=dropout,
        name=f"{name_prefix}_mhsa",
    )(x, x)
    attn_out = layers.Dropout(dropout, name=f"{name_prefix}_attn_dropout")(attn_out)
    x = layers.LayerNormalization(epsilon=1e-5, name=f"{name_prefix}_norm1")(x + attn_out)

    ffn = layers.Dense(dim_feedforward, activation="gelu", name=f"{name_prefix}_ffn1")(x)
    ffn = layers.Dropout(dropout, name=f"{name_prefix}_ffn_dropout")(ffn)
    ffn = layers.Dense(d_model, name=f"{name_prefix}_ffn2")(ffn)
    ffn = layers.Dropout(dropout, name=f"{name_prefix}_ffn_dropout2")(ffn)
    x = layers.LayerNormalization(epsilon=1e-5, name=f"{name_prefix}_norm2")(x + ffn)
    return x


def build_transformer_autoencoder(
    seq_len: int,
    feature_dim: int = 2,
    d_model: int = 64,
    nhead: int = 4,
    num_layers: int = 2,
    dim_feedforward: int = 128,
    dropout: float = 0.1,
) -> keras.Model:
    inputs = keras.Input(shape=(seq_len, feature_dim), name="input")

    x = layers.Dense(d_model, name="input_proj")(inputs)
    x = PositionalEncoding(d_model, max_len=seq_len + 10, name="pos_encoding")(x)
    x = layers.Dropout(dropout, name="pos_dropout")(x)

    for i in range(num_layers):
        x = transformer_encoder_block(x, d_model, nhead, dim_feedforward, dropout, f"enc{i}")

    x = layers.Dense(dim_feedforward, activation="gelu", name="output_ffn1")(x)
    x = layers.Dropout(dropout, name="output_dropout")(x)
    outputs = layers.Dense(feature_dim, name="output_proj")(x)

    return keras.Model(inputs, outputs, name="Transformer_Autoencoder")
