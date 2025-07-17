import torch.nn as nn


class Distiller(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.down_conv = nn.Conv1d(
            in_channels=dims, out_channels=dims, kernel_size=3, padding=1, padding_mode='circular')
        self.norm = nn.BatchNorm1d(dims)
        self.activation = nn.ELU()
        self.max_pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.down_conv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.max_pool(x)
        x = x.transpose(1, 2)
        return x
