"""
Proposed Parallel LST-Former Autoencoder (Section 3.4.3) -- TensorFlow/Keras
port of ../../../src/models/lstformer_ae.py (PyTorch).

Default fusion strategy: "hybrid_nobottleneck" -- Track B (Transformer)
keeps its full per-timestep sequence (never pooled to a single vector, same
as TransformerAutoencoder), and Track A (LSTM)'s single latent vector is
injected as a per-timestep conditioning bias instead of being fused via an
equally-weighted single-vector bottleneck. See ../../../outputs/lstformer_ablation.md
and README Section 7 for why this was needed: the original "concat" fusion
(kept here for reference/reproducing that ablation) forced both branches
through the same hard bottleneck, which made LST-Former track the weaker
LSTM baseline instead of the stronger, un-bottlenecked Transformer.
"""

from __future__ import annotations

import keras
from keras import layers

from src.models.transformer_ae import PositionalEncoding, transformer_encoder_block


def build_lstformer_autoencoder(
    seq_len: int,
    feature_dim: int = 2,
    hidden_dim: int = 64,
    latent_dim: int = 32,
    d_model: int = 64,
    nhead: int = 4,
    num_transformer_layers: int = 2,
    dim_feedforward: int = 128,
    dropout: float = 0.1,
    fusion_type: str = "hybrid_nobottleneck",
) -> keras.Model:
    if fusion_type == "hybrid_nobottleneck":
        return _build_hybrid_nobottleneck(
            seq_len, feature_dim, hidden_dim, latent_dim, d_model, nhead,
            num_transformer_layers, dim_feedforward, dropout,
        )
    return _build_bottlenecked(
        seq_len, feature_dim, hidden_dim, latent_dim, d_model, nhead,
        num_transformer_layers, dim_feedforward, dropout, fusion_type,
    )


def _build_hybrid_nobottleneck(
    seq_len, feature_dim, hidden_dim, latent_dim, d_model, nhead,
    num_transformer_layers, dim_feedforward, dropout,
) -> keras.Model:
    inputs = keras.Input(shape=(seq_len, feature_dim), name="input")

    # Track A -- single conditioning vector (local rhythm summary)
    lstm_out = layers.LSTM(hidden_dim, dropout=dropout, name="lstm_branch")(inputs)
    z_lstm = layers.Dense(latent_dim, name="lstm_latent")(lstm_out)
    lstm_bias = layers.Dense(d_model, name="lstm_to_dmodel")(z_lstm)
    lstm_bias = layers.Reshape((1, d_model), name="lstm_bias_reshape")(lstm_bias)

    # Track B -- full per-timestep sequence, never pooled to one vector
    t = layers.Dense(d_model, name="input_proj")(inputs)
    t = PositionalEncoding(d_model, max_len=seq_len + 10, name="pos_encoding")(t)
    t = layers.Dropout(dropout, name="pos_dropout")(t)
    for i in range(num_transformer_layers):
        t = transformer_encoder_block(t, d_model, nhead, dim_feedforward, dropout, f"enc{i}")

    # Inject Track A's summary at every position, then decode in parallel
    fused_seq = layers.Add(name="inject_lstm_bias")([t, lstm_bias])
    fused_seq = layers.LayerNormalization(epsilon=1e-5, name="hybrid_norm")(fused_seq)
    for i in range(num_transformer_layers):
        fused_seq = transformer_encoder_block(
            fused_seq, d_model, nhead, dim_feedforward, dropout, f"dec{i}"
        )

    x = layers.Dense(dim_feedforward, activation="gelu", name="output_ffn1")(fused_seq)
    x = layers.Dropout(dropout, name="output_dropout")(x)
    outputs = layers.Dense(feature_dim, name="output_proj")(x)

    return keras.Model(inputs, outputs, name="LSTFormer_Autoencoder")


def _build_bottlenecked(
    seq_len, feature_dim, hidden_dim, latent_dim, d_model, nhead,
    num_transformer_layers, dim_feedforward, dropout, fusion_type,
) -> keras.Model:
    """Original single-vector-bottleneck fusion (concat only) -- kept for
    reproducing the ablation study that motivated the hybrid design above."""
    if fusion_type != "concat":
        raise ValueError(
            f"TensorFlow port only implements 'concat' and 'hybrid_nobottleneck' "
            f"(got {fusion_type!r}); see the PyTorch version for the full ablation set."
        )

    inputs = keras.Input(shape=(seq_len, feature_dim), name="input")

    lstm_out = layers.LSTM(hidden_dim, dropout=dropout, name="lstm_branch")(inputs)
    z_lstm = layers.Dense(latent_dim, name="lstm_latent")(lstm_out)

    t = layers.Dense(d_model, name="input_proj")(inputs)
    t = PositionalEncoding(d_model, max_len=seq_len + 10, name="pos_encoding")(t)
    t = layers.Dropout(dropout, name="pos_dropout")(t)
    for i in range(num_transformer_layers):
        t = transformer_encoder_block(t, d_model, nhead, dim_feedforward, dropout, f"enc{i}")
    t = layers.GlobalAveragePooling1D(name="global_avg_pool")(t)
    z_trans = layers.Dense(latent_dim, name="transformer_latent")(t)

    fused = layers.Concatenate(name="fusion_concat")([z_lstm, z_trans])
    fused = layers.Dense(latent_dim * 2, activation="relu", name="fusion_dense")(fused)

    x = layers.Dense(hidden_dim, name="from_fusion")(fused)
    x = layers.RepeatVector(seq_len, name="repeat_vector")(x)
    x = layers.LSTM(hidden_dim, return_sequences=True, dropout=dropout, name="decoder_lstm")(x)
    outputs = layers.TimeDistributed(layers.Dense(feature_dim), name="output_projection")(x)

    return keras.Model(inputs, outputs, name="LSTFormer_Autoencoder")
