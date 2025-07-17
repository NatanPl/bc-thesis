import torch.nn as nn
import torch.nn.functional as F


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation='relu'):
        """
        Initializes the EncoderLayer with an attention layer, feed-forward dimensions, and dropout.

        Args:
            attention_layer (nn.Module): The attention layer to be used in the encoder.
            d_model (int): Dimension of the model.
            d_ff (int): Dimension of the feed-forward network.
            dropout (float): Dropout rate for regularization.
        """
        super().__init__()
        d_ff = d_ff or 4*d_model
        self.attention = attention

        self.conv1 = nn.Conv1d(in_channels=d_model,
                               out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff,
                               out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == 'relu' else F.gelu

    def forward(self, x, attention_mask=None):
        """
        Forward pass of the encoder layer.

        Args:
            x (torch.Tensor): Input tensor of shape [B, L, D].
            attention_mask (torch.Tensor, optional): Attention mask of shape [B, 1, L, S]. Defaults to None.

        Returns:
            torch.Tensor: Output tensor of shape [B, L, D]
            torch.Tensor: Attention weights of shape [B, H, L, S] or None.
        """
        new_x, attention = self.attention(x, x, x, attention_mask)

        x_out = x + self.dropout(new_x)

        y = x_out = self.norm1(x_out)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x_out + y), attention


# class EncoderLayerSpacetimeformer(nn.Module):
#     def __init__(self,
#         global_attention, local_attention, d_model,
#         d_yc, time_windows
#         d_ff, dropout=0.1, activation='relu'):
#         """
#         Initializes the EncoderLayer with global and local attention layers, feed-forward dimensions, and dropout.

#         Args:
#             global_attention (nn.Module): Global attention layer.
#             local_attention (nn.Module): Local attention layer.
#             d_model (int): Dimension of the model.
#             d_ff (int): Dimension of the feed-forward network.
#             dropout (float): Dropout rate for regularization.
#         """
#         super().__init__()
#         d_ff = d_ff or 4*d_model
#         self.global_attention = global_attention
#         self.local_attention = local_attention

#         self.conv1 = nn.Conv1d(in_channels=d_model,
#                                out_channels=d_ff, kernel_size=1)
#         self.conv2 = nn.Conv1d(in_channels=d_ff,
#                                out_channels=d_model, kernel_size=1)
#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
#         self.dropout = nn.Dropout(dropout)
#         self.activation = F.relu if activation == 'relu' else F.gelu
