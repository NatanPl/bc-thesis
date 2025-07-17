import lightning as L
import torch
import torch.nn.functional as F
import torchmetrics
import torch.optim as optim

from collections.abc import Mapping
from typing import Any, Sequence


class ForecastModule(L.LightningModule):
    def __init__(self, model, loss_fn=F.mse_loss, learning_rate=0.001, horizon=1, metric_cls=torchmetrics.MeanAbsoluteError):
        super(ForecastModule, self).__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.learning_rate = learning_rate
        self.horizon = horizon
        self.train_mae = metric_cls()
        self.val_mae = metric_cls()
        self.test_mae = metric_cls()

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def configure_optimizers(self):  # type: ignore
        opt = optim.AdamW(self.parameters(), lr=self.learning_rate)
        sch = optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=5)
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sch,
                             "interval": "epoch",
                             "monitor": "val_loss"}
        }

    def _call_model(self, inputs: Any):
        if isinstance(inputs, Mapping):               # dict -> **kwargs
            return self(**inputs)
        if isinstance(inputs, Sequence):              # tuple/list -> *args
            return self(*inputs)
        return self(inputs)                           # single tensor

    def _step(self, batch, mode: str):
        inputs, target, *rest = batch

        # Check for NaN/inf in inputs (handle both tensor and dict inputs)
        if isinstance(inputs, dict):
            # For dict inputs (like Informer), check each tensor in the dict
            for key, tensor in inputs.items():
                if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                    print(f"WARNING: {mode} inputs[{key}] contain NaN/inf!")
                    print(
                        f"Input[{key}] stats: min={tensor.min()}, max={tensor.max()}, mean={tensor.mean()}")
        else:
            # For tensor inputs (like RNN)
            if torch.isnan(inputs).any() or torch.isinf(inputs).any():
                print(f"WARNING: {mode} inputs contain NaN/inf!")
                print(
                    f"Input stats: min={inputs.min()}, max={inputs.max()}, mean={inputs.mean()}")

        if torch.isnan(target).any() or torch.isinf(target).any():
            print(f"WARNING: {mode} targets contain NaN/inf!")
            print(
                f"Target stats: min={target.min()}, max={target.max()}, mean={target.mean()}")

        preds = self._call_model(inputs)

        # Check for NaN/inf in predictions
        if torch.isnan(preds).any() or torch.isinf(preds).any():
            print(f"WARNING: {mode} predictions contain NaN/inf!")
            print(
                f"Prediction stats: min={preds.min()}, max={preds.max()}, mean={preds.mean()}")
            if isinstance(inputs, dict):
                print(f"Input is a dict with keys: {list(inputs.keys())}")
            else:
                print(f"Input range: {inputs.min():.2f} to {inputs.max():.2f}")

        # Ensure predictions are contiguous for metrics
        preds = preds.contiguous()
        target = target.contiguous()

        loss = self.loss_fn(preds, target)

        # Check for NaN/inf in loss
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"WARNING: {mode} loss is NaN/inf!")
            print(f"Loss value: {loss}")
            print(f"Pred stats: min={preds.min()}, max={preds.max()}")
            print(f"Target stats: min={target.min()}, max={target.max()}")

        # Update and compute the metric
        mae_metric = getattr(self, f"{mode}_mae")
        mae_metric.update(preds, target)
        mae_value = mae_metric.compute()

        self.log_dict({f"{mode}_loss": loss,
                       f"{mode}_mae": mae_value},
                      prog_bar=True)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch, "train")

        # Early stopping if loss explodes (more reasonable threshold)
        if torch.isnan(loss) or torch.isinf(loss) or loss > 1e6:
            print(f"WARNING: Training loss exploded: {loss}")
            print("Stopping training to prevent further divergence")
            self.trainer.should_stop = True

        return loss

    def validation_step(self, batch, _): return self._step(batch, "val")
    def test_step(self, batch, _): return self._step(batch, "test")

    def on_before_optimizer_step(self, optimizer):
        """Monitor gradients and detect numerical issues."""
        # Check for NaN/inf gradients
        total_norm = 0.0
        param_count = 0

        for name, param in self.named_parameters():
            if param.grad is not None:
                # Check for NaN/inf
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    print(f"WARNING: NaN/inf gradients detected in {name}!")
                    print(
                        f"Grad stats: min={param.grad.min()}, max={param.grad.max()}")
                    # Zero out problematic gradients
                    param.grad.data.zero_()

                # Calculate gradient norm
                param_norm = param.grad.data.norm()
                total_norm += param_norm.item() ** 2
                param_count += 1

                # Log large gradients
                if param_norm > 10.0:
                    print(
                        f"WARNING: Large gradient norm in {name}: {param_norm:.4f}")

        # Log total gradient norm (manual calculation)
        if param_count > 0:
            total_norm = total_norm ** 0.5
            self.log('grad_norm', total_norm, prog_bar=False, logger=True)

    # def forecast(self, n_steps: int | None = None, *args, **kwargs):
    #     """
    #     Autoregressively roll the model forward.
    #     If `n_steps` is None, use `self.horizon` (single forward pass).

    #     Returns
    #     -------
    #     (n_steps, n_features)  for an unbatched input, or
    #     (B, n_steps, n_features) if input already had batch dimension.
    #     """
    #     self.eval()
    #     n_steps = n_steps or self.horizon
    #     with torch.no_grad():
    #         return self.model.forecast(n_steps=n_steps, *args, **kwargs)
