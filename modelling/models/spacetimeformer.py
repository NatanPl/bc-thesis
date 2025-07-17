# spacetimeformer.py
import torch
import torch.nn as nn

from modelling.models.components.attentions import (
    AttentionLayer, FullAttention, ProbSparseAttention
)
from modelling.models.components.encoder import Encoder
from modelling.models.components.encoder_layer import EncoderLayer
from modelling.models.components.decoder import Decoder
from modelling.models.components.decoder_layer import DecoderLayer
from modelling.models.components.masking import TriangularCausalMask
from modelling.models.components.distiller import Distiller
from modelling.models.components.embedding import *


def _causal_mask(batch_size, seq_len, device):
    # [B,1,L,L]
    return TriangularCausalMask(batch_size, seq_len, device=device).mask


class Spacetimeformer(nn.Module):
    def __init__(
        self,
        feature_dim: int,      # V – number of variables
        prediction_len: int,   # P – forecast horizon
        label_len: int,        # L_d – context fed to decoder
        model_dim: int = 512,
        n_heads: int = 8,
        d_ff: int = 512,
        encoder_layers: int = 3,
        decoder_layers: int = 2,
        distil_layers: int = 0,
        dropout_rate: float = 0.1,
        attention: str = "full",            # "full" or "prob_sparse"
        attention_factor: int = 5,          # only for ProbSparse
        activation: str = "gelu",
        output_attention: bool = False,
        freq: str = "a",                    # 'a' = annual
        mix: bool = True,                   # keep Informer’s mix option
    ):
        super().__init__()
        self.pred_len = prediction_len
        self.output_attention = output_attention

        # Helper function to create attention with compatible parameters
        def create_attention(mask_flag, is_self_attention=True):
            if attention == "full":
                return FullAttention(
                    mask_flag=mask_flag,
                    attention_dropout=dropout_rate
                )
            else:
                return ProbSparseAttention(
                    mask_flag=mask_flag,
                    factor=attention_factor,
                    attention_dropout=dropout_rate,
                    output_attention=output_attention if is_self_attention else False
                )

        # ---------- embeddings ----------
        self.enc_embedding = ChannelDataEmbedding(
            feature_dim, model_dim, dropout_rate, freq)
        self.dec_embedding = ChannelDataEmbedding(
            feature_dim, model_dim, dropout_rate, freq)

        # ---------- encoder ----------
        self.encoder = Encoder(
            attention_layers=[
                EncoderLayer(
                    attention=AttentionLayer(
                        attention=create_attention(mask_flag=True),  # causal!
                        d_model=model_dim,
                        n_heads=n_heads,
                        mix=False
                    ),
                    d_model=model_dim,
                    d_ff=d_ff,
                    dropout=dropout_rate,
                    activation=activation
                ) for _ in range(encoder_layers)
            ],
            distiller_layers=[
                Distiller(model_dim) for _ in range(distil_layers)
            ] if distil_layers > 0 else None,
            normalization=nn.LayerNorm(model_dim)
        )

        # ---------- decoder ----------
        self.decoder = Decoder(
            decoder_layers=[
                DecoderLayer(
                    self_attention=AttentionLayer(
                        # causal in decoder, too
                        attention=create_attention(mask_flag=True),
                        d_model=model_dim,
                        n_heads=n_heads,
                        mix=mix
                    ),
                    cross_attention=AttentionLayer(
                        attention=FullAttention(
                            mask_flag=False,
                            attention_dropout=dropout_rate
                        ),
                        d_model=model_dim,
                        n_heads=n_heads,
                        mix=False
                    ),
                    d_model=model_dim,
                    d_ff=d_ff,
                    dropout=dropout_rate,
                    activation=activation
                ) for _ in range(decoder_layers)
            ],
            normalization=nn.LayerNorm(model_dim)
        )

        # ---------- head ----------
        # predict all V variables
        self.projection = nn.Linear(model_dim, feature_dim, bias=True)

    def forward(
        self,
        enc_x,         # [B, L_enc, V]
        enc_x_mark,    # [B, L_enc, C_t]
        # [B, label_len + pred_len, V]  (teacher-forcing zeros for future part)
        dec_x,
        dec_x_mark,    # [B, label_len + pred_len, C_t]
        enc_mask=None, dec_self_mask=None, dec_cross_mask=None
    ):
        B, L_enc, _ = enc_x.shape
        _, L_dec, _ = dec_x.shape

        # default causal masks on first forward pass
        if enc_mask is None:
            enc_mask = _causal_mask(B, L_enc, enc_x.device)
        if dec_self_mask is None:
            dec_self_mask = _causal_mask(B, L_dec, dec_x.device)

        # embeddings
        enc_out = self.enc_embedding(enc_x, enc_x_mark)
        dec_in = self.dec_embedding(dec_x, dec_x_mark)

        # pass through encoder–decoder
        enc_out, attns = self.encoder(enc_out, attention_mask=enc_mask)
        dec_out = self.decoder(dec_in, enc_out,
                               x_mask=dec_self_mask,
                               cross_mask=dec_cross_mask)
        pred = self.projection(
            dec_out)[:, -self.pred_len:, :]   # keep last P steps

        return (pred, attns) if self.output_attention else pred


class SpaceTimeFormer(nn.Module):
    """
    Wrapper to make Spacetimeformer compatible with the experiment framework.

    Expected interface:
    - __init__(n_features: int, horizon: int, **kwargs)
    - forward(inputs: dict) where inputs has keys matching the collator output
    """

    def __init__(self, n_features: int, horizon: int, **kwargs):
        super().__init__()

        # Extract parameters with defaults
        # Default label_len to horizon
        label_len = kwargs.get('label_len', horizon)
        # Smaller default for stability
        model_dim = kwargs.get('model_dim', 256)
        n_heads = kwargs.get('n_heads', 8)
        d_ff = kwargs.get('d_ff', 512)
        encoder_layers = kwargs.get('encoder_layers', 2)
        decoder_layers = kwargs.get('decoder_layers', 1)
        distil_layers = kwargs.get('distil_layers', 0)
        dropout_rate = kwargs.get('dropout_rate', 0.1)
        attention = kwargs.get('attention', 'full')
        attention_factor = kwargs.get('attention_factor', 5)
        activation = kwargs.get('activation', 'gelu')
        output_attention = kwargs.get('output_attention', False)
        freq = kwargs.get('freq', 'a')
        mix = kwargs.get('mix', True)

        self.spacetimeformer = Spacetimeformer(
            feature_dim=n_features,
            prediction_len=horizon,
            label_len=label_len,
            model_dim=model_dim,
            n_heads=n_heads,
            d_ff=d_ff,
            encoder_layers=encoder_layers,
            decoder_layers=decoder_layers,
            distil_layers=distil_layers,
            dropout_rate=dropout_rate,
            attention=attention,
            attention_factor=attention_factor,
            activation=activation,
            output_attention=output_attention,
            freq=freq,
            mix=mix
        )

    def forward(self, x_encoder=None, x_encoder_mark=None, x_decoder=None, x_decoder_mark=None, **kwargs):
        """
        Forward pass expecting keyword arguments from the ForecastModule.

        Args:
        - x_encoder: [B, L_enc, V]
        - x_encoder_mark: [B, L_enc, C_t] 
        - x_decoder: [B, L_dec, V]
        - x_decoder_mark: [B, L_dec, C_t]
        """
        # Handle both dict input (from test) and keyword args (from framework)
        if x_encoder is None and len(kwargs) == 1 and isinstance(list(kwargs.values())[0], dict):
            # Called with a single dict argument (test case)
            inputs = list(kwargs.values())[0]
            return self.spacetimeformer(
                enc_x=inputs["x_encoder"],
                enc_x_mark=inputs["x_encoder_mark"],
                dec_x=inputs["x_decoder"],
                dec_x_mark=inputs["x_decoder_mark"]
            )
        else:
            # Called with keyword arguments (framework case)
            return self.spacetimeformer(
                enc_x=x_encoder,
                enc_x_mark=x_encoder_mark,
                dec_x=x_decoder,
                dec_x_mark=x_decoder_mark
            )
