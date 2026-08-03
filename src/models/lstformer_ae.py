"""
Proposed Parallel LST-Former Autoencoder (Section 3.4.3).

Track A (LSTM): localized micro-rhythms -> 1D latent vector
Track B (Transformer): global macro-rhythms -> GAP -> 1D latent vector
Fusion: several selectable strategies (see `fusion_type`) -> shared LSTM decoder -> reconstruction

`fusion_type` and the capacity knobs below were added during an ablation
study (see ../../outputs/lstformer_ablation.md) investigating why the
original concat+Dense fusion underperformed a standalone Transformer.
That study found the fusion strategy and decoder type didn't matter --
the real cause was that both branches were forced through the same hard
single-vector bottleneck, unlike the (un-bottlenecked) Transformer
baseline. `fusion_type="hybrid_nobottleneck"` tests that hypothesis
directly: Track B keeps its full per-timestep sequence, and Track A's
latent is injected as a per-timestep conditioning bias instead.

Follow-up: `hybrid_nobottleneck` made LST-Former statistically tied with
(not better than) the standalone Transformer. Track A's contribution there
is still just ONE global summary vector (final LSTM hidden state)
broadcast identically to every position -- likely redundant with what
self-attention already sees, since it's not really "local" information
anymore. `fusion_type="local_injection"` tests a genuinely local signal
instead: Track A returns its hidden state AT EVERY timestep (not just the
final one), and each position's local state is injected into the
*matching* position of Track B's sequence -- so every position gets both
"what does global attention see" and "what's the local micro-rhythm right
here", instead of one global summary duplicated across positions twice.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.transformer_ae import PositionalEncoding


class LSTFormerAutoencoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        d_model: int = 64,
        nhead: int = 4,
        num_transformer_layers: int = 2,
        dim_feedforward: int = 128,
        seq_len: int = 50,
        dropout: float = 0.1,
        fusion_type: str = "concat",
        decoder_type: str = "lstm",
    ):
        super().__init__()
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.fusion_type = fusion_type
        self.decoder_type = decoder_type

        # Track A — Sequential (LSTM) branch
        self.lstm_encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.lstm_latent = nn.Linear(hidden_dim, latent_dim)

        # Track B — Attention (Transformer) branch
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len + 10, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_transformer_layers
        )
        self.transformer_latent = nn.Linear(d_model, latent_dim)

        # Feature fusion (strategy depends on fusion_type) + shared decoder
        if fusion_type == "concat":
            # Original: concat -> Linear -> ReLU
            fusion_dim = latent_dim * 2
            self.fusion = nn.Linear(fusion_dim, fusion_dim)
            self.from_fusion = nn.Linear(fusion_dim, hidden_dim)
        elif fusion_type == "norm_concat":
            # Concat, but LayerNorm each branch first so neither dominates by scale
            self.lstm_norm = nn.LayerNorm(latent_dim)
            self.trans_norm = nn.LayerNorm(latent_dim)
            fusion_dim = latent_dim * 2
            self.fusion = nn.Linear(fusion_dim, fusion_dim)
            self.from_fusion = nn.Linear(fusion_dim, hidden_dim)
        elif fusion_type == "gated":
            # Learned per-sample gate decides the LSTM/Transformer mix
            self.gate = nn.Sequential(nn.Linear(latent_dim * 2, latent_dim), nn.Sigmoid())
            self.from_fusion = nn.Linear(latent_dim, hidden_dim)
        elif fusion_type == "weighted_sum":
            # Single learned scalar mixing weight (not per-sample)
            self.mix_logit = nn.Parameter(torch.zeros(1))
            self.from_fusion = nn.Linear(latent_dim, hidden_dim)
        elif fusion_type == "hybrid_nobottleneck":
            # Unlike the other strategies, Track B is NOT pooled to a single
            # vector here -- its full per-timestep sequence is kept (same as
            # TransformerAutoencoder, no compression). Track A's single latent
            # vector is broadcast to every timestep and added as a local-rhythm
            # conditioning signal, rather than forcing both branches through
            # the same hard bottleneck before fusion.
            self.lstm_to_dmodel = nn.Linear(latent_dim, d_model)
            self.hybrid_norm = nn.LayerNorm(d_model)
            decoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            self.hybrid_decoder = nn.TransformerEncoder(decoder_layer, num_layers=num_transformer_layers)
            self.output_layer = nn.Sequential(
                nn.Linear(d_model, dim_feedforward),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim_feedforward, input_dim),
            )
        elif fusion_type == "local_injection":
            # Track A contributes its per-timestep hidden state (not just the
            # final one) -- a genuinely local, position-varying signal that's
            # injected into the matching position of Track B's sequence.
            self.lstm_seq_to_dmodel = nn.Linear(hidden_dim, d_model)
            self.local_norm = nn.LayerNorm(d_model)
            decoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            self.local_decoder = nn.TransformerEncoder(decoder_layer, num_layers=num_transformer_layers)
            self.output_layer = nn.Sequential(
                nn.Linear(d_model, dim_feedforward),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim_feedforward, input_dim),
            )
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")

        if fusion_type in ("hybrid_nobottleneck", "local_injection"):
            pass  # decoder already built above; decoder_type is ignored for this variant
        elif decoder_type == "lstm":
            self.decoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
            self.output_layer = nn.Linear(hidden_dim, input_dim)
        elif decoder_type == "transformer":
            # Parallel (non-recurrent) decoding: broadcast the latent to every
            # timestep, add positional encoding so identical broadcast values
            # become position-distinguishable, then let self-attention (not
            # recurrence) reconstruct the sequence -- mirrors TransformerAutoencoder's
            # decoding style instead of LSTM-AE's.
            self.decoder_pos_encoder = PositionalEncoding(hidden_dim, max_len=seq_len + 10, dropout=dropout)
            decoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=num_transformer_layers)
            self.output_layer = nn.Sequential(
                nn.Linear(hidden_dim, dim_feedforward),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim_feedforward, input_dim),
            )
        else:
            raise ValueError(f"Unknown decoder_type: {decoder_type}")

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # Track A
        _, (h_n, _) = self.lstm_encoder(x)
        z_lstm = self.lstm_latent(h_n[-1])

        # Track B
        t = self.input_proj(x)
        t = self.pos_encoder(t)
        t = self.transformer_encoder(t)
        t = t.mean(dim=1)  # Global Average Pooling
        z_trans = self.transformer_latent(t)

        if self.fusion_type == "concat":
            fused = torch.cat([z_lstm, z_trans], dim=-1)
            return torch.relu(self.fusion(fused))
        if self.fusion_type == "norm_concat":
            fused = torch.cat([self.lstm_norm(z_lstm), self.trans_norm(z_trans)], dim=-1)
            return torch.relu(self.fusion(fused))
        if self.fusion_type == "gated":
            g = self.gate(torch.cat([z_lstm, z_trans], dim=-1))
            return g * z_lstm + (1 - g) * z_trans
        if self.fusion_type == "weighted_sum":
            alpha = torch.sigmoid(self.mix_logit)
            return alpha * z_lstm + (1 - alpha) * z_trans
        raise ValueError(f"Unknown fusion_type: {self.fusion_type}")

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        repeated = self.from_fusion(z).unsqueeze(1).repeat(1, self.seq_len, 1)
        if self.decoder_type == "lstm":
            dec_out, _ = self.decoder(repeated)
        else:
            dec_out = self.decoder(self.decoder_pos_encoder(repeated))
        return self.output_layer(dec_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fusion_type == "hybrid_nobottleneck":
            return self._forward_hybrid_nobottleneck(x)
        if self.fusion_type == "local_injection":
            return self._forward_local_injection(x)
        z = self.encode(x)
        return self.decode(z)

    def _forward_hybrid_nobottleneck(self, x: torch.Tensor) -> torch.Tensor:
        # Track A -- single conditioning vector (local rhythm summary)
        _, (h_n, _) = self.lstm_encoder(x)
        z_lstm = self.lstm_latent(h_n[-1])
        lstm_bias = self.lstm_to_dmodel(z_lstm).unsqueeze(1)  # (batch, 1, d_model)

        # Track B -- full per-timestep sequence, never pooled to one vector
        t = self.input_proj(x)
        t = self.pos_encoder(t)
        t = self.transformer_encoder(t)

        # Inject Track A's summary at every position, then decode in parallel
        fused_seq = self.hybrid_norm(t + lstm_bias)
        dec_out = self.hybrid_decoder(fused_seq)
        return self.output_layer(dec_out)

    def _forward_local_injection(self, x: torch.Tensor) -> torch.Tensor:
        # Track A -- per-timestep hidden state (genuinely local, varies by position)
        lstm_seq, _ = self.lstm_encoder(x)  # (batch, seq_len, hidden_dim)
        local_signal = self.lstm_seq_to_dmodel(lstm_seq)  # (batch, seq_len, d_model)

        # Track B -- full per-timestep sequence (global context via self-attention)
        t = self.input_proj(x)
        t = self.pos_encoder(t)
        t = self.transformer_encoder(t)

        # Inject Track A's LOCAL signal at each matching position (not broadcast)
        fused_seq = self.local_norm(t + local_signal)
        dec_out = self.local_decoder(fused_seq)
        return self.output_layer(dec_out)
