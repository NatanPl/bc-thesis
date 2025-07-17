import torch
import torch.nn as nn
import math


class DataEmbedding(nn.Module):
    def __init__(self, dimension_in, dimension_model, embedding_type='fixed', frequency='a', dropout=0.1):
        super().__init__()

        self.value_embedding = TokenEmbedding(dimension_in, dimension_model)
        self.position_embedding = PositionEmbedding(dimension_model)
        # if embedding_type == 'timeF':
        self.temporal_embedding = TimeFeatureEmbedding(
            d_model=dimension_model, freq=frequency)
        # else:
        # self.temporal_embedding = TemporalEmbedding(
        #     d_model=dimension_model, embed_type=embedding_type, freq=frequency)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x)
        x += self.position_embedding(x)
        x += self.temporal_embedding(x_mark)
        return self.dropout(x)


class TokenEmbedding(nn.Module):
    def __init__(self, dimension_in, dimension_model):
        super().__init__()
        self.convolution = nn.Conv1d(
            in_channels=dimension_in, out_channels=dimension_model,
            kernel_size=3,
            padding=1,
            padding_mode='circular'
        )

        nn.init.kaiming_normal_(self.convolution.weight)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.convolution(x).transpose(1, 2)
        return x


class PositionEmbedding(nn.Module):
    def __init__(self, dimension_model, max_length=5000):
        super().__init__()

        position_embedding = torch.zeros(
            max_length, dimension_model).float()

        position_embedding.requires_grad = False

        position = torch.arange(0, max_length).float().unsqueeze(1)
        div_term = torch.arange(0, dimension_model, 2).float()
        div_term *= -(math.log(10000.0) / dimension_model)
        div_term = div_term.exp()

        position_embedding[:, 0::2] = torch.sin(position * div_term)
        position_embedding[:, 1::2] = torch.cos(position * div_term)

        position_embedding = position_embedding.unsqueeze(0)

        self.register_buffer('position_embedding', position_embedding)

    def forward(self, x):
        return self.position_embedding[:, :x.size(1)]  # type: ignore


class FixedEmbedding(nn.Module):
    def __init__(self, dimension_in, dimension_model):
        super(FixedEmbedding, self).__init__()

        weights = torch.zeros(dimension_in, dimension_model).float()
        weights.requires_grad = False

        position = torch.arange(0, dimension_in).float().unsqueeze(1)
        div_term = torch.arange(0, dimension_model, 2).float()
        div_term *= -(math.log(10000.0) / dimension_model)
        div_term = div_term.exp()

        weights[:, 0::2] = torch.sin(position * div_term)
        weights[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(dimension_in, dimension_model)
        self.emb.weight = nn.Parameter(weights, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


# class TemporalEmbedding(nn.Module):
#     def __init__(self, d_model, embed_type='fixed', freq='a'):
#         super(TemporalEmbedding, self).__init__()

#         minute_size = 4
#         hour_size = 24
#         weekday_size = 7
#         day_size = 32
#         month_size = 13

#         Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
#         if freq == 't':
#             self.minute_embed = Embed(minute_size, d_model)
#         self.hour_embed = Embed(hour_size, d_model)
#         self.weekday_embed = Embed(weekday_size, d_model)
#         self.day_embed = Embed(day_size, d_model)
#         self.month_embed = Embed(month_size, d_model)

#     def forward(self, x):
#         x = x.long()

#         minute_x = self.minute_embed(x[:, :, 4]) if hasattr(
#             self, 'minute_embed') else 0.
#         hour_x = self.hour_embed(x[:, :, 3])
#         weekday_x = self.weekday_embed(x[:, :, 2])
#         day_x = self.day_embed(x[:, :, 1])
#         month_x = self.month_embed(x[:, :, 0])

#         return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, freq='a'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model)

    def forward(self, x):
        return self.embed(x)


class ChannelDataEmbedding(nn.Module):
    """
    Token = one time step; variables are channels.
    Spatial (variable) information is carried by the learnable
    weight matrix of the value projection.
    """

    def __init__(self, feature_dim, d_model, dropout=0.1, freq='a'):
        super().__init__()
        # rows act as spatial embeddings
        self.value_proj = nn.Linear(feature_dim, d_model, bias=False)
        self.position_emb = PositionEmbedding(d_model)
        self.time_emb = TimeFeatureEmbedding(d_model, freq=freq)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_val, x_mark):
        """
        x_val  : [B, L, V]   - raw series
        x_mark : [B, L, C_t] - time-features already scaled ∈[0,1]
        """
        x = self.value_proj(
            x_val)                     # + learnable spatial embedding
        x += self.position_emb(x)
        x += self.time_emb(x_mark)
        return self.dropout(x)
