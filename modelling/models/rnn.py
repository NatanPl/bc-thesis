import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNmodel(nn.Module):
    def __init__(
        self,
        n_features,
        cell_type: str = "lstm",
        num_layers: int = 1,
        hidden_dim: int = 64,
        dropout_rate: float = 0.0,
        horizon: int = 1
    ):
        super().__init__()
        self.n_features = n_features
        self.horizon = horizon

        cell_type = cell_type.lower().strip()
        if cell_type == "lstm":
            rnn_cls = nn.LSTM
            self.cell_type = "LSTM"
        elif cell_type == "gru":
            rnn_cls = nn.GRU
            self.cell_type = "GRU"
        else:
            raise ValueError(f"Unsupported cell_type {cell_type}")

        self.rnn = rnn_cls(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0.0,
        )

        self.proj = nn.Linear(hidden_dim, n_features * horizon)
        
        # Better weight initialization for large-scale data
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with smaller values to prevent immediate overflow."""
        # Initialize LSTM/GRU weights with smaller scale
        for name, param in self.rnn.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param, gain=0.1)  # Much smaller gain
            elif 'bias' in name:
                nn.init.constant_(param, 0)
        
        # Initialize projection layer with very small weights
        nn.init.xavier_uniform_(self.proj.weight, gain=0.01)
        nn.init.constant_(self.proj.bias, 0)

    def forward(self, x):
        """
        If x has shape (L, F) add a batch dimension;
        output always has shape (B, horizon, F).
        """
        single_series = x.ndim == 2
        if single_series:
            x = x.unsqueeze(0)                       # (1, L, F)

        if self.cell_type == "LSTM":
            _, (h_n, _) = self.rnn(x)
        else:
            _, h_n = self.rnn(x)
        last_h = h_n[-1]

        # Project to original feature space
        prediction = self.proj(last_h)                    # (B, horizon * F)
        prediction = prediction.view(x.size(0), self.horizon,    # (B, H, F)
                                     self.n_features)

        # Safety check for NaN/inf values
        if torch.isnan(prediction).any() or torch.isinf(prediction).any():
            print(f"WARNING: Model output contains NaN/inf!")
            print(f"Input stats: min={x.min().item():.2f}, max={x.max().item():.2f}, mean={x.mean().item():.2f}")
            print(f"Hidden state stats: min={last_h.min().item():.2f}, max={last_h.max().item():.2f}")
            print(f"Prediction stats: min={prediction.min().item()}, max={prediction.max().item()}")
            # Return zeros as fallback to prevent crash
            prediction = torch.zeros_like(prediction)

        return prediction.squeeze(0) if single_series else prediction
