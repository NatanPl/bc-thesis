# Modelling

This directory contains the neural network implementations and training infrastructure for time series forecasting.

## Structure

```
modelling/
├── core/                   # Core ML infrastructure
│   ├── datamodule.py       # PyTorch Lightning data handling
│   └── forecast.py         # Base forecasting module
├── models/                 # Model implementations
│   ├── rnn.py              # LSTM/GRU models
│   ├── informer.py         # Informer transformer
│   ├── spacetimeformer.py  # SpaceTimeFormer
│   ├── baseline.py         # Simple baseline models
│   ├── wrappers.py         # Model wrapper utilities
│   └── components/         # Shared model components
├── configs/                # Model configuration files
└── experiment_wrapper.py   # Main experiment orchestration
```
