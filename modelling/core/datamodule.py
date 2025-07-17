from dataclasses import dataclass, asdict
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import lightning as L

try:
    from ...raw_data.dataset import Dataset as RawDataset
except ImportError:
    from raw_data.dataset import Dataset as RawDataset
    # for local testing without the full package structure


@dataclass
class WindowConfig:
    """
    Hyper-parameters that control how raw panels become model samples.
    """
    # sampling
    window: int = 5
    """Number of past time steps in each sample"""
    horizon: int = 1
    """Number of future time steps to predict"""
    stride: int = 1
    """Step size for moving the window"""

    # optional channels
    include_time_idx: bool = False
    """Whether to include relative time indices (0..window-1)"""
    include_time_values: bool = False
    """Whether to include absolute time values"""
    include_feature_idx: bool = False
    """Whether to include feature indices (0..n_features-1)"""

    def to_dict(self):
        """
        Convert the configuration to a dictionary.
        """
        return asdict(self)


class WindowedTimeSeriesDataset(Dataset):
    """
    Sliding-window view of a :class:`RawDataset`.

    Each sample is a dict with the following keys (all numpy arrays):
    - `x`: The input features of shape (window, n_features).
    - `y`: The target values of shape (horizon, n_features).
    - `time_idx`: Relative time indices of shape (window,).
    - `time_values`: Absolute time values of shape (window,).
    - `feature_idx`: Feature indices of shape (n_features,).
    """

    def __init__(self, dataset: RawDataset, config: WindowConfig):
        super().__init__()
        self.raw_dataset = dataset
        self.config = config

        countries, indicators, years = dataset.shape()
        window, horizon, stride = config.window, config.horizon, config.stride

        self.indices: List[tuple[int, int]] = []
        """List of indices (country_idx, year_idx) for each sample."""
        for country_idx in range(countries):
            for year_idx in range(0, years - window - horizon + 1, stride):
                self.indices.append((country_idx, year_idx))

        if config.include_feature_idx:
            # precompute feature indices
            self.feature_indices = np.arange(indicators, dtype=np.float32)

        if config.include_time_idx:
            # precompute relative time indices
            self.time_indices = np.arange(window, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        country_idx, year_idx = self.indices[idx]
        window, horizon = self.config.window, self.config.horizon

        # shape note: raw_dataset is (indicators, years), so transpose to (years, indicators)
        country_data = self.raw_dataset.data[country_idx].T
        x = country_data[year_idx:year_idx + window]
        """Input features of shape (window, n_features)."""
        y = country_data[year_idx + window:year_idx + window + horizon]
        """Target values of shape (horizon, n_features)."""

        sample = {
            'x': x.astype(np.float32),
            'y': y.astype(np.float32),
        }
        if self.config.include_time_idx:
            sample['time_idx'] = self.time_indices

        if self.config.include_time_values:
            sample['time_values'] = self.raw_dataset.years[year_idx:year_idx +
                                                           window].astype(np.float32)

        if self.config.include_feature_idx:
            sample['feature_idx'] = self.feature_indices

        return sample


class Collator:
    @staticmethod
    def rnn(batch: Sequence[dict]):
        x = torch.tensor(np.stack([b['x']
                         for b in batch]), dtype=torch.float32)
        y = torch.tensor(np.stack([b['y']
                         for b in batch]), dtype=torch.float32)
        return x, y

    @staticmethod
    def informer(batch: Sequence[dict]):
        window = batch[0]["x"].shape[0]
        horizon = batch[0]["y"].shape[0]
        batch_size = len(batch)
        n_features = batch[0]["x"].shape[1]

        enc_x = torch.tensor(np.stack([b["x"]
                             for b in batch]), dtype=torch.float32)
        time_idx = batch[0].get('time_idx', None)
        if time_idx is None:
            time_idx = np.arange(window, dtype=np.float32) / window
        enc_mark = torch.tensor(np.tile(time_idx, (batch_size, 1))[
                                :, :, None], dtype=torch.float32)

        label_len = window // 2
        dec_len = horizon
        dec_known = torch.zeros(
            batch_size, label_len + dec_len, n_features, dtype=torch.float32)
        dec_known[:, :label_len] = enc_x[:, -label_len:]

        dec_time_idx = np.arange(
            label_len + dec_len, dtype=np.float32) / (label_len + dec_len)
        dec_mark = torch.tensor(
            np.tile(dec_time_idx, (batch_size, 1))[:, :, None], dtype=torch.float32)

        tgt = torch.tensor(np.stack([b["y"]
                           for b in batch]), dtype=torch.float32)

        return {
            "x_encoder": enc_x,
            "x_encoder_mark": enc_mark,
            "x_decoder": dec_known,
            "x_decoder_mark": dec_mark,
        }, tgt

    @staticmethod
    def spacetimeformer(batch: Sequence[dict]):
        """
        Collator for Spacetimeformer models.
        Uses the same format as Informer since they have compatible interfaces.
        """
        return Collator.informer(batch)


class TimeSeriesDataModule(L.LightningDataModule):
    """
    Wraps raw dataset into Lightning DataLoaders.
    """

    def __init__(
        self,
        *,
        raw_train: RawDataset,
        raw_val: RawDataset,
        raw_test: RawDataset,
        config: WindowConfig | None = None,
        batch_size: int = 32,
        num_workers: int = 0,
        seed: int = 42,
        persistent_workers: bool = False,
        collate_fn: Callable = Collator.rnn,
        pin_memory: bool = True
    ):
        super().__init__()
        self.raw_train = raw_train
        self.raw_val = raw_val
        self.raw_test = raw_test
        self.config = config or WindowConfig()
        self.batch_size = batch_size
        self.seed = seed
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers and num_workers > 0
        self.collate_fn = collate_fn
        self.pin_memory = pin_memory

    def setup(self, stage: str | None = None):
        self.train_set = WindowedTimeSeriesDataset(
            self.raw_train, self.config
        )
        if self.raw_val is not None:
            self.val_set = WindowedTimeSeriesDataset(
                self.raw_val, self.config
            )
        else:
            self.val_set = None
        if self.raw_test is not None:
            self.test_set = WindowedTimeSeriesDataset(
                self.raw_test, self.config
            )
        else:
            self.test_set = None

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_set is None:
            return None  # type: ignore
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory
        )

    def test_dataloader(self) -> DataLoader:
        if self.test_set is None:
            return None  # type: ignore
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory
        )

    def n_features(self) -> int:
        """
        Returns the number of features in the dataset.
        """
        return self.train_set[0]['x'].shape[1]
