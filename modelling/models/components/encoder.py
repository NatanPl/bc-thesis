import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, attention_layers, distiller_layers=None, normalization=None, embedding_dropout=0.0):
        """
        Initializes the Encoder with attention layers, optional distiller layers, and normalization.

        Args:
            attention_layers (list): List of attention layer instances.
            distiller_layers (list, optional): List of distiller layer instances. Defaults to None.
            normalization (nn.Module, optional): Normalization layer. Defaults to None.
            embedding_dropout (float, optional): Dropout rate for embeddings. Defaults to 0.0.
        """
        super().__init__()
        self.attention_layers = nn.ModuleList(attention_layers)
        self.distiller_layers = nn.ModuleList(
            distiller_layers) if distiller_layers else None
        self.normalization = normalization
        self.embedding_dropout = nn.Dropout(embedding_dropout)

    def forward(self, x, attention_mask=None):
        x = self.embedding_dropout(x)
        args = {"attention_mask": attention_mask}
        return self._forward(x, args)

    def _forward(self, x, args):
        attentions = []
        for i in range(len(self.attention_layers)):
            attention_layer = self.attention_layers[i]
            x, attention = attention_layer(x, **args)
            attentions.append(attention)

            if self.distiller_layers and len(self.distiller_layers) > i:
                if self.distiller_layers[i] is not None:
                    x = self.distiller_layers[i](x)

        if self.normalization:
            x = self.normalization(x)

        return x, attentions


class EncoderSpacetimeformer(Encoder):
    def forward(self, x, attention_mask=None):
        # x is expected to be a tuple: (value_time_embedding, space_embedding)
        value_time_embedding, space_embedding = x
        x = self.embedding_dropout(value_time_embedding)
        x += self.embedding_dropout(space_embedding)
        args = {"attention_mask": attention_mask}
        return self._forward(x, args)
