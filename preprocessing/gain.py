from typing import Tuple, Optional
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
torch.set_float32_matmul_precision('high')


class GAIN(pl.LightningModule):
    """
    Generative Adversarial Imputation Network that operates on indicator vectors.

    One training sample = one year slice of indicators for a country.  Batching is handled by the
    `DataLoader`, so every tensor that reaches this module is either
    * `shape == (batch, n_indicators)`, or
    * `shape == (n_indicators,)` for a single sample.
    """

    def __init__(
        self,
        input_dim: int,              # n_indicators
        *,
        hint_rate: float = 0.9,
        alpha: float = 10.0,
        learning_rate: float = 1e-3,
        hidden_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.automatic_optimization = False  # Set to use manual optimization
        self.save_hyperparameters()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim or 2 * self.input_dim

        # ── Generator ──────────────────────────────────────────────
        self.generator = nn.Sequential(
            nn.Linear(self.input_dim * 2, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim // 2, self.input_dim),
            nn.Sigmoid(),
        )

        # ── Discriminator ──────────────────────────────────────────
        self.discriminator = nn.Sequential(
            nn.Linear(self.input_dim * 2, self.hidden_dim),
            nn.LeakyReLU(),
            # nn.Dropout(0.3),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.LeakyReLU(),
            # nn.Dropout(0.3),
            nn.Linear(self.hidden_dim // 2, self.input_dim),
        )

        self.bce = nn.BCEWithLogitsLoss(reduction="mean")
        self.lr = learning_rate
        self.hint_rate = hint_rate

        self.alpha = alpha
        self.register_buffer("adv_scale", torch.tensor(0.0001))

    def _assert_finite_params(self):
        for n, p in self.named_parameters():
            if p.isnan().any() or p.isinf().any():
                raise RuntimeError(f"parameter {n} became NaN/Inf")

    @staticmethod
    def _mask(x: torch.Tensor) -> torch.Tensor:
        """1 for observed, 0 for NaN."""
        return (~torch.isnan(x)).float()

    def _hint(self, mask: torch.Tensor) -> torch.Tensor:
        B = (torch.rand_like(mask) < self.hint_rate).float()
        return mask * B + 0.5 * (1 - B)

    def _corrupt(self, flat, observ_mask, p=0.2):
        """Randomly drop a fraction *p* of the observed entries."""
        drop = (torch.rand_like(flat) < p) & observ_mask.bool()
        new_mask = observ_mask.clone()
        # now 0 = missing (either real or dropped)
        new_mask[drop] = 0.
        return new_mask, drop              # drop marks the cells whose targets we know

    @staticmethod
    def _random_fill(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return torch.where(mask.bool(), x, torch.rand_like(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B?, I)
        """Single pass imputation."""
        added_batch = False
        if x.ndim == 1:  # single sample
            x = x.unsqueeze(0)
            added_batch = True

        mask = self._mask(x)
        x_in = self._random_fill(x, mask)
        gen_out = self.generator(torch.cat([x_in, mask], dim=1))
        completed = x_in * mask + gen_out * (1 - mask)

        return completed.squeeze(0) if added_batch else completed

    def configure_optimizers(self):
        opt_d = torch.optim.Adam(
            self.discriminator.parameters(), lr=self.lr * 0.5)
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.lr)
        return [opt_d, opt_g], []

    def _gen_step(self, flat, orig_observ):
        # 0 ) make a *training* mask (real NaNs + extra random drops)
        mask, drop_mask = self._corrupt(flat, orig_observ, p=0.2)

        # 1 ) build the visible input for G
        x_in = torch.where(mask.bool(), flat, torch.rand_like(flat))
        gen_out = self.generator(torch.cat([x_in, mask], dim=1))
        gen_out = torch.nan_to_num(
            gen_out, nan=0.5, posinf=1.0, neginf=0.0)  # sanity check

        # 2 ) completed sample fed to D
        completed = x_in * mask + gen_out * (1 - mask)

        # 3 ) discriminator pass
        hint = self._hint(mask)
        logits = self.discriminator(torch.cat([completed, hint], dim=1))
        logits = torch.clamp(logits, -80.0, 80.0)

        # 4 ) losses
        recon = (gen_out - flat)[drop_mask].pow(2).mean()
        impute_mask = 1.0 - mask
        adv = (F.binary_cross_entropy_with_logits(logits, torch.ones_like(
            mask), reduction="none") * impute_mask).sum() / impute_mask.sum()
        loss = recon + self.alpha * self.adv_scale * adv

        return loss, recon, adv, completed, hint, mask

    def training_step(self, batch, batch_idx):
        (x,) = batch                           # (B, I)
        B = x.size(0)
        flat = x.view(B, -1)
        orig_observ = self._mask(flat)

        # Get optimizers manually
        opt_d, opt_g = self.optimizers()

        # ── Discriminator ──────────────────────────────────────────
        opt_d.zero_grad()
        with torch.no_grad():
            _, _, _, completed, hint, tr_mask = self._gen_step(
                flat, orig_observ)

        # Important: these tensors need to have requires_grad=True for discriminator training
        completed = completed.detach().clone().requires_grad_(True)
        hint = hint.detach().clone().requires_grad_(True)

        logits_d = self.discriminator(torch.cat([completed, hint], dim=1))
        logits_d = torch.clamp(logits_d, -80.0, 80.0)
        impute_d = 1.0 - tr_mask
        loss_d = (F.binary_cross_entropy_with_logits(logits_d, tr_mask,
                  reduction="none") * impute_d).sum() / impute_d.sum()

        # Manual backward pass
        self.manual_backward(loss_d)

        # Manual gradient clipping for discriminator
        self.clip_gradients(
            optimizer=opt_d,
            gradient_clip_val=1.0,
            gradient_clip_algorithm="norm",
        )

        opt_d.step()
        self.log("disc_loss", loss_d, prog_bar=True)

        # ── Generator ─────────────────────────────────────────────
        opt_g.zero_grad()
        loss_g, recon, adv, _, _, _ = self._gen_step(flat, orig_observ)

        # Manual backward pass
        self.manual_backward(loss_g)

        # Manual gradient clipping for generator
        self.clip_gradients(
            optimizer=opt_g,
            gradient_clip_val=1.0,
            gradient_clip_algorithm="norm",
        )

        opt_g.step()
        self.log_dict(
            {"gen_loss": loss_g, "gen_loss_recon": recon, "gen_loss_adv": adv},
            prog_bar=True
        )

        self._assert_finite_params()

        if torch.isnan(loss_d) or torch.isnan(loss_g):
            raise RuntimeError("loss blew up")
        for n, p in self.named_parameters():
            if torch.isnan(p).any():
                print(f"{n} has NaNs")

        return {"loss": loss_g}

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        (x,) = batch
        B = x.size(0)
        flat = x.view(B, -1)
        observ = self._mask(flat)
        loss_g, _, _, _, _, _ = self._gen_step(flat, observ)
        self.log("val_gen_loss", loss_g, prog_bar=True, sync_dist=True)

    def on_train_epoch_start(self) -> None:
        if (self.current_epoch >= 30 and
                getattr(self.trainer, "last_recon", 1.0) < 1e-3):
            self.adv_scale.fill_(min(0.02, float(self.adv_scale) + 0.002))

class GAINImputer:
    """High-level API for fitting / transforming 3-D arrays where axis-0
    enumerates samples (e.g. countries). Each year-slice is treated as a 
    separate training sample."""

    def __init__(
        self,
        *,
        window_size: int = 5,
        step_size: int = 1,
        hint_rate: float = 0.9,
        alpha: float = 10.0,
        learning_rate: float = 1e-3,
        batch_size: int = 64,
        max_epochs: int = 1000,
        patience: int = 50,
        train_val_split: float = 0.8,
        seed: int = 42,
        hidden_dim: Optional[int] = None,
        verbose: bool = True,
    ) -> None:
        self.window_size = window_size
        self.step_size = step_size
        self.hint_rate = hint_rate
        self.alpha = alpha
        self.lr = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.train_val_split = train_val_split
        self.seed = seed
        self.hidden_dim = hidden_dim
        self.verbose = verbose

        self.model: Optional[GAIN] = None
        self._min: Optional[np.ndarray] = None
        self._max: Optional[np.ndarray] = None

    # ── helpers ────────────────────────────────────────────────────
    def _norm(self, x: np.ndarray) -> np.ndarray:
        """Normalize data to [0,1] range, handling NaN values appropriately."""
        if self._min is None:
            # Calculate min/max per indicator across all countries and years
            self._min = np.nanmin(x, axis=(0, 2), keepdims=True)
            self._max = np.nanmax(x, axis=(0, 2), keepdims=True)

            # Handle cases where all values are NaN for a feature
            if np.any(np.isnan(self._min)):
                if self.verbose:
                    print("Warning: Some features have all NaN values")
                # Replace NaN min/max with 0/1 to prevent NaN propagation
                self._min = np.nan_to_num(self._min, nan=0.0)
                self._max = np.nan_to_num(self._max, nan=1.0)

        rng = self._max - self._min
        # Avoid division by zero for features with constant values
        rng[rng < 1e-10] = 1.0

        # Normalize while preserving NaN values
        normalized = (x - self._min) / rng
        # Extra safety: ensure all values are in [0, 1] range
        normalized = np.clip(normalized, 0, 1)
        return normalized

    def _denorm(self, x: np.ndarray) -> np.ndarray:
        """Denormalize data back to original scale."""
        return x * (self._max - self._min) + self._min

    def _loaders(self, t: torch.Tensor):
        """Create data loaders for training and validation sets."""
        n = t.size(0)
        if n < 2:  # Handle case with a single sample
            # Create a doubled dataset by adding small noise
            t_doubled = torch.cat([t, t + 0.01 * torch.randn_like(t)], dim=0)
            train_ds = TensorDataset(t_doubled)
            val_ds = TensorDataset(t)
        else:
            idx = np.random.RandomState(self.seed).permutation(n)
            n_train = max(1, int(n * self.train_val_split))
            train_ds = TensorDataset(t[idx[:n_train]])
            val_ds = TensorDataset(t[idx[n_train:]])

        return (
            DataLoader(train_ds, batch_size=min(self.batch_size, len(train_ds)),
                       shuffle=True),
            DataLoader(val_ds, batch_size=min(self.batch_size, len(val_ds)),
                       shuffle=False),
        )

    # ── public API ─────────────────────────────────────────────────
    def fit(self, data: np.ndarray):
        """Fit the GAIN model to the data."""
        if self.verbose:
            print(f"Input data shape: {data.shape}")
            print(
                f"Missing values: {np.isnan(data).sum()}/{data.size} ({np.isnan(data).mean()*100:.2f}%)")

        # Check if too many NaNs - GAIN won't learn well with too many missing values
        if np.isnan(data).mean() > 0.8:
            print(
                "Warning: More than 80% of values are missing. Imputation results may be poor.")

        # Check if entire features are missing
        if np.any(np.isnan(data).all(axis=(0, 2))):
            print("Warning: Some features have all missing values.")

        # Normalize data
        norm = self._norm(data)
        assert not np.isnan(norm).all(), "normalisation already produced NaNs"

        if self.verbose:
            print(f"Normalized data - NaN count: {np.isnan(norm).sum()}")

        # Reshape data to treat each year slice as a sample
        W, step = self.window_size, self.step_size
        samples, mapping = [], []
        n_countries, n_indicators, n_years = norm.shape
        for c in range(n_countries):
            for start in range(0, n_years - W + 1, step):
                # (n_indicators * W,)
                window = norm[c, :, start:start+W].reshape(-1)
                samples.append(window)
                mapping.append((c, start))
        reshaped_data = np.asarray(samples, dtype=np.float32)
        self._mapping = mapping

        tensor = torch.tensor(reshaped_data, dtype=torch.float32)
        train_loader, val_loader = self._loaders(tensor)

        # Create model with appropriate input dimension
        self.model = GAIN(
            input_dim=n_indicators * self.window_size,
            hint_rate=self.hint_rate,
            alpha=self.alpha,
            learning_rate=self.lr,
            hidden_dim=self.hidden_dim,
        )

        early_stop = pl.callbacks.EarlyStopping(
            monitor="val_gen_loss", patience=self.patience, mode="min"
        )
        trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            callbacks=[early_stop],
            enable_progress_bar=self.verbose,
            enable_checkpointing=False,
            logger=True,
            log_every_n_steps=1,
        )

        trainer.fit(self.model, train_loader, val_loader)
        if self.verbose:
            print(f"Training completed after {trainer.current_epoch} epochs")

        # Store the original shape info for reshaping during transform
        self.n_indicators = n_indicators
        self.n_years = n_years

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Apply the fitted GAIN model to impute missing values."""
        if self.model is None:
            raise RuntimeError("Model is not trained; call fit() first.")

        # Normalize the data
        norm = self._norm(data)

        # Reshape to year slices
        n_countries, n_indicators, n_years = norm.shape
        W, step = self.window_size, self.step_size

        # 1 ) rebuild the windows
        windows, mapping = [], []
        for c in range(n_countries):
            for start in range(0, n_years - W + 1, step):
                windows.append(norm[c, :, start:start+W].reshape(-1))
                mapping.append((c, start))
        t = torch.tensor(np.asarray(windows, dtype=np.float32))

        # 2 ) impute
        self.model.eval()
        with torch.no_grad():
            imputed_flat = self.model(t).cpu().numpy()
        imputed_windows = imputed_flat.reshape(-1, n_indicators, W)

        # 3 ) fold back with averaging on overlaps
        out = norm.copy()   # still contains NaNs
        counts = np.zeros_like(norm)
        for (c, s), win in zip(mapping, imputed_windows):
            out[c, :, s:s+W] = np.nan_to_num(out[c, :, s:s+W]) + win
            counts[c, :, s:s+W] += 1

        # avoid /0 for untouched cells
        out /= np.where(counts == 0, 1, counts)
        return self._denorm(out)

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fit the model and impute in one step."""
        self.fit(data)
        return self.transform(data)
