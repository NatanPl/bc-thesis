"""
Model wrappers to standardize interfaces for the ExperimentRunner.

These wrappers convert models with different forward signatures to match
the standard interface expected by the forecasting framework.
"""

import torch
import torch.nn as nn
from .informer import Informer
from .spacetimeformer import Spacetimeformer


class InformerWrapper(nn.Module):
    """Wrapper for Informer to match ExperimentRunner interface."""

    def __init__(self, n_features, horizon, **kwargs):
        super().__init__()
        # Map ExperimentRunner parameters to Informer parameters
        informer_kwargs = {
            "feature_dim_out": n_features,
            "feature_dim_enc": n_features,
            "feature_dim_dec": n_features,
            "prediction_len": horizon,
            **kwargs  # Pass through other parameters
        }
        self.model = Informer(**informer_kwargs)
        self.horizon = horizon

    def forward(self, x_encoder, x_encoder_mark, x_decoder, x_decoder_mark,
                encoder_self_mask=None, decoder_self_mask=None, decoder_encoder_mask=None):
        """
        Forward pass that matches the original Informer interface.
        """
        return self.model(x_encoder, x_encoder_mark, x_decoder, x_decoder_mark,
                         encoder_self_mask, decoder_self_mask, decoder_encoder_mask)


class SpacetimeformerWrapper(nn.Module):
    """Wrapper for Spacetimeformer to match ExperimentRunner interface."""

    def __init__(self, n_features, horizon, label_len=24, **kwargs):
        super().__init__()
        # Map ExperimentRunner parameters to Spacetimeformer parameters
        spacetimeformer_kwargs = {
            "feature_dim": n_features,
            "prediction_len": horizon,
            "label_len": label_len,  # Decoder context length
            **kwargs  # Pass through other parameters
        }
        self.model = Spacetimeformer(**spacetimeformer_kwargs)
        self.horizon = horizon
        self.label_len = label_len

    def forward(self, x_encoder=None, x_encoder_mark=None, x_decoder=None, x_decoder_mark=None, **kwargs):
        """
        Forward pass that maps framework parameters to Spacetimeformer interface.
        
        Framework uses: x_encoder, x_encoder_mark, x_decoder, x_decoder_mark
        Spacetimeformer expects: enc_x, enc_x_mark, dec_x, dec_x_mark
        """
        # Debug: Check what we received
        if x_encoder is None:
            print(f"DEBUG: SpacetimeformerWrapper received None for x_encoder")
            print(f"DEBUG: kwargs keys: {list(kwargs.keys()) if kwargs else 'No kwargs'}")
            if kwargs:
                for k, v in kwargs.items():
                    if isinstance(v, dict):
                        print(f"  kwargs[{k}] is dict with keys: {list(v.keys())}")
                    else:
                        print(f"  kwargs[{k}] type: {type(v)}")
        
        # Handle both dict input (from test) and keyword args (from framework)
        if x_encoder is None and len(kwargs) == 1 and isinstance(list(kwargs.values())[0], dict):
            # Called with a single dict argument (test case)
            inputs = list(kwargs.values())[0]
            return self.model(
                enc_x=inputs["x_encoder"],
                enc_x_mark=inputs["x_encoder_mark"],
                dec_x=inputs["x_decoder"], 
                dec_x_mark=inputs["x_decoder_mark"]
            )
        else:
            # Called with keyword arguments (framework case)
            # Map framework parameter names to Spacetimeformer parameter names
            return self.model(
                enc_x=x_encoder,
                enc_x_mark=x_encoder_mark,
                dec_x=x_decoder, 
                dec_x_mark=x_decoder_mark
            )
