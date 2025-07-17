import torch.nn as nn

from modelling.models.components.attentions import AttentionLayer, ProbSparseAttention, FullAttention
from modelling.models.components.distiller import Distiller
from modelling.models.components.encoder import Encoder
from modelling.models.components.encoder_layer import EncoderLayer
from modelling.models.components.decoder import Decoder
from modelling.models.components.decoder_layer import DecoderLayer
from modelling.models.components.embedding import DataEmbedding


class Informer(nn.Module):
    def __init__(self,
                 feature_dim_out: int, feature_dim_enc: int, feature_dim_dec: int,
                 prediction_len: int,
                 model_dim: int = 512, n_heads: int = 8,
                 d_ff: int = 512,
                 attention_factor: int = 5,
                 encoder_layers: int = 3,
                 distil_layers: int = 0,
                 decoder_layers: int = 2,
                 dropout_rate: float = 0.0,
                 attention: str = 'prob_sparse',
                 output_attention: bool = False,
                 activation: str = 'relu',
                 embedding_type: str = 'fixed',
                 frequency: str = 'a',
                 mix: bool = True,
                 ):
        super().__init__()
        self.output_attention = output_attention
        self.feature_dim_out = feature_dim_out
        self.prediction_len = prediction_len

        Attention = ProbSparseAttention if attention == 'prob_sparse' else FullAttention

        # Create attention with appropriate parameters
        def create_attention(mask_flag, **kwargs):
            if attention == 'prob_sparse':
                return Attention(
                    mask_flag=mask_flag,
                    factor=attention_factor,
                    attention_dropout=dropout_rate,
                    output_attention=output_attention
                )
            else:  # FullAttention
                return Attention(
                    mask_flag=mask_flag,
                    attention_dropout=dropout_rate
                )

        self.encoder_embedding = DataEmbedding(
            feature_dim_enc, model_dim, embedding_type, frequency, dropout_rate)
        self.decoder_embedding = DataEmbedding(
            feature_dim_dec, model_dim, embedding_type, frequency, dropout_rate)

        self.encoder = Encoder(
            attention_layers=[
                EncoderLayer(
                    attention=AttentionLayer(
                        attention=create_attention(mask_flag=False),
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
                Distiller(
                    dims=model_dim,
                ) for _ in range(distil_layers)
            ] if distil_layers > 0 else None,
            normalization=nn.LayerNorm(model_dim),
        )
        self.decoder = Decoder(
            decoder_layers=[
                DecoderLayer(
                    self_attention=AttentionLayer(
                        attention=create_attention(mask_flag=True),
                        d_model=model_dim,
                        n_heads=n_heads,
                        mix=mix
                    ),
                    cross_attention=AttentionLayer(
                        attention=FullAttention(
                            mask_flag=False,
                            attention_dropout=dropout_rate,
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
            normalization=nn.LayerNorm(model_dim),
        )
        self.projection = nn.Linear(model_dim, feature_dim_out, bias=True)

    def forward(self, x_encoder, x_encoder_mark, x_decoder, x_decoder_mark,
                encoder_self_mask=None, decoder_self_mask=None, decoder_encoder_mask=None):
        encoder_embed = self.encoder_embedding(x_encoder, x_encoder_mark)
        encoder_output, attention_weights = self.encoder(
            encoder_embed, attention_mask=encoder_self_mask)

        decoder_embed = self.decoder_embedding(x_decoder, x_decoder_mark)
        decoder_output = self.decoder(
            decoder_embed, encoder_output, x_mask=decoder_self_mask, cross_mask=decoder_encoder_mask)
        decoder_output = self.projection(decoder_output)

        if self.output_attention:
            return decoder_output[:, -self.prediction_len:, :], attention_weights
        else:
            return decoder_output[:, -self.prediction_len:, :]
