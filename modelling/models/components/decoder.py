import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, decoder_layers, normalization=None):
        """
        Initializes the Decoder with decoder layers and optional normalization.

        Args:
            decoder_layers (list): List of decoder layer instances.
            normalization (nn.Module, optional): Normalization layer. Defaults to None.
        """
        super().__init__()
        self.decoder_layers = nn.ModuleList(decoder_layers)
        self.normalization = normalization

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        """
        Forward pass through the decoder.

        Args:
            x (torch.Tensor): Input tensor for the decoder.
            cross (torch.Tensor): Cross-attention input tensor.
            x_mask (torch.Tensor, optional): Mask for the decoder input. Defaults to None.
            cross_mask (torch.Tensor, optional): Mask for the cross-attention input. Defaults to None.

        Returns:
            torch.Tensor: Output tensor after passing through the decoder layers.
        """
        for layer in self.decoder_layers:
            x = layer(x, cross, x_mask, cross_mask)

        if self.normalization:
            x = self.normalization(x)

        return x
