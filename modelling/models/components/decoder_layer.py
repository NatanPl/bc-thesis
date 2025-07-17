import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        """
        Initializes a single decoder layer with self-attention and cross-attention mechanisms.

        Args:
            self_attention (nn.Module): Self-attention mechanism.
            cross_attention (nn.Module): Cross-attention mechanism.
            d_model (int): Dimension of the model.
            d_ff (int, optional): Dimension of the feed-forward network. Defaults to None.
            dropout (float, optional): Dropout rate. Defaults to 0.1.
            activation (str, optional): Activation function to use. Defaults to "relu".
        """
        super().__init__()
        d_ff = d_ff or 4 * d_model

        self.self_attention = self_attention
        self.cross_attention = cross_attention

        self.conv1 = nn.Conv1d(
            in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(
            in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x: torch.Tensor, cross, x_mask=None, cross_mask=None):
        """
        Forward pass through the decoder layer.

        Args:
            x (torch.Tensor): Input tensor for self-attention.
            cross (torch.Tensor): Input tensor for cross-attention.
            x_mask (torch.Tensor, optional): Mask for self-attention input. Defaults to None.
            cross_mask (torch.Tensor, optional): Mask for cross-attention input. Defaults to None.

        Returns:
            torch.Tensor: Output tensor after passing through the decoder layer.
        """
        # Self-attention
        x2, _ = self.self_attention(x, x, x, x_mask)
        x = x + self.dropout(x2)
        x = self.norm1(x)

        # Cross-attention
        x2, _ = self.cross_attention(x, cross, cross, cross_mask)
        x = x + self.dropout(x2)
        x = self.norm2(x)

        # Feed-forward network
        residual = x.clone()
        x = self.conv1(x.transpose(-1, 1))
        x = self.activation(x)
        x = self.dropout(x)
        x = self.conv2(x).transpose(-1, 1)
        x = self.dropout(x)
        x = self.norm3(x + residual)

        return x
