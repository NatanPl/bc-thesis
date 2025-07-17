"""experiment_wrapper.py

Modular tooling for running and orchestrating time-series forecasting experiments.

It contains three layers:

1) `ExperimentRunner`: wires together the fundamental components of a model training.

2) `ExperimentJob`: wraps a single runner plus all meta-data needed to launch
   it (run directory, backend, host, etc.). A job can execute locally,
   as a subprocess, or via SSH.

3) `ExperimentSuite`: a collection of jobs that can be fired off asynchronously
   (locally or remotely), then collated into a tidy pandas `DataFrame` for
   analysis.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Type, Union

import lightning as L
from lightning.pytorch import callbacks as L_callbacks
from lightning.pytorch.loggers import CSVLogger
import pandas as pd
import torch

from modelling.core.datamodule import Collator, TimeSeriesDataModule, WindowConfig
from modelling.core.forecast import ForecastModule

from modelling.models.rnn import RNNmodel
from modelling.models.informer import Informer
from modelling.models.spacetimeformer import SpaceTimeFormer


@dataclass
class ExperimentConfig:
    """Serializable collection of everything that defines a run."""

    # reproducibility
    seed: int = 42

    # DataModule hyper‑params
    window_config: WindowConfig = field(default_factory=WindowConfig)
    batch_size: int = 64
    num_workers: int = 4
    pin_memory: bool = True
    collate: str | Callable = "auto"  # "auto" | "rnn" | "informer" | callable

    # Model hyper‑params
    model_cls: Type[torch.nn.Module] = RNNmodel
    model_kwargs: Dict[str, Any] = field(default_factory=dict)
    learning_rate: float = 1e-3
    loss_fn: Callable = torch.nn.functional.mse_loss

    # Trainer hyper‑params
    max_epochs: int = 100
    accelerator: str = "auto"
    precision: Any = 32
    log_every_n_steps: int = 10
    callbacks: Optional[Sequence[L.Callback]] = None
    trainer_kwargs: Dict[str, Any] = field(default_factory=dict)

    @property
    def horizon(self) -> int:
        return self.window_config.horizon


class ExperimentRunner:
    """Single-run wrapper (DataModule/Model/Trainer)"""

    def __init__(
        self,
        *,
        raw_train,
        raw_val,
        raw_test,
        experiment_config: ExperimentConfig,
        trainer: L.Trainer | None = None,
    ) -> None:
        self.exp_cfg = experiment_config
        L.seed_everything(experiment_config.seed, workers=True)

        # DataModule
        collate_fn = self._pick_collate(
            experiment_config.collate, experiment_config.model_cls)
        self.datamodule = TimeSeriesDataModule(
            raw_train=raw_train,
            raw_val=raw_val,
            raw_test=raw_test,
            config=experiment_config.window_config,
            batch_size=experiment_config.batch_size,
            num_workers=experiment_config.num_workers,
            pin_memory=experiment_config.pin_memory,
            collate_fn=collate_fn,
        )
        self.datamodule.setup()

        # Model - with automatic wrapper handling
        n_features = self.datamodule.n_features()

        # Import wrappers only when needed to avoid circular imports
        from modelling.models.wrappers import InformerWrapper, SpacetimeformerWrapper

        # Map original models to their wrappers for ExperimentRunner compatibility
        model_cls = experiment_config.model_cls
        if model_cls.__name__ == 'Informer':
            model_cls = InformerWrapper
        elif model_cls.__name__ == 'Spacetimeformer':
            model_cls = SpacetimeformerWrapper

        core_model = model_cls(
            n_features=n_features,
            horizon=experiment_config.horizon,
            **experiment_config.model_kwargs,
        )

        # LightningModule
        self.lit_model = ForecastModule(
            model=core_model,
            loss_fn=experiment_config.loss_fn,
            learning_rate=experiment_config.learning_rate,
            horizon=experiment_config.horizon,
        )

        # Trainer
        self.trainer = trainer or self._build_trainer()

    # public methods
    def fit(self):
        self.trainer.fit(self.lit_model, datamodule=self.datamodule)
        return self

    def test(self):
        self.trainer.test(self.lit_model, datamodule=self.datamodule)
        return self

    def predict(self, *args, **kwargs):
        return self.trainer.predict(self.lit_model, *args, **kwargs)

    # internals
    def _build_trainer(self) -> L.Trainer:
        default_cbs = [
            L_callbacks.EarlyStopping(monitor="val_loss", patience=10),
            L_callbacks.ModelCheckpoint(
                monitor="val_loss", filename="{epoch:02d}-{val_loss:.4f}"),
        ]
        callbacks = list(
            self.exp_cfg.callbacks) if self.exp_cfg.callbacks else default_cbs

        # Automatically adjust log_every_n_steps based on dataset size
        train_dataloader = self.datamodule.train_dataloader()
        num_training_batches = len(train_dataloader)
        adjusted_log_every_n_steps = min(
            self.exp_cfg.log_every_n_steps, max(1, num_training_batches))

        return L.Trainer(
            max_epochs=self.exp_cfg.max_epochs,
            accelerator=self.exp_cfg.accelerator,
            precision=self.exp_cfg.precision,
            log_every_n_steps=adjusted_log_every_n_steps,
            callbacks=callbacks,
            logger=CSVLogger(save_dir=".", name="lightning_logs"),
            **self.exp_cfg.trainer_kwargs,
        )

    @staticmethod
    def _pick_collate(specified: str | Callable, model_cls: Type[torch.nn.Module]):
        if callable(specified):
            return specified
        if specified == "auto":
            # Import wrappers to check for them
            from modelling.models.wrappers import InformerWrapper, SpacetimeformerWrapper
            
            # Both Informer and Spacetimeformer need the informer collator
            if (issubclass(model_cls, Informer) or 
                model_cls.__name__ == 'Informer' or
                model_cls.__name__ == 'SpacetimeformerWrapper' or
                model_cls.__name__ == 'InformerWrapper' or
                model_cls.__name__ == 'Spacetimeformer'):
                return Collator.informer
            return Collator.rnn
        if specified == "rnn":
            return Collator.rnn
        if specified == "informer":
            return Collator.informer
        if specified == "spacetimeformer":
            return Collator.spacetimeformer
        raise ValueError(f"Unknown collate spec: {specified}")


class ExperimentJob:
    """Encapsulates one config/run and knows *how* to execute it."""

    def __init__(
        self,
        *,
        config: ExperimentConfig,
        raw_train,
        raw_val,
        raw_test,
        root: Path,
        backend: str = "local",  # local | subprocess | ssh
        remote_host: str | None = None,
        job_id: str | None = None,
    ) -> None:
        self.config = config
        self.raw_train = raw_train
        self.raw_val = raw_val
        self.raw_test = raw_test
        self.backend = backend
        self.remote_host = remote_host
        self.job_id = job_id or uuid.uuid4().hex[:8]
        self.job_dir = root / self.job_id
        self.job_dir.mkdir(parents=True, exist_ok=True)
        # dump config for reproducibility
        (self.job_dir / "config.json").write_text(json.dumps(asdict(config),
                                                             indent=2, default=str))

    # ---------------  execution  ---------------
    async def run_async(self):
        if self.backend == "local":
            await asyncio.to_thread(self._run_local)
        elif self.backend == "subprocess":
            await self._run_subprocess()
        elif self.backend == "ssh":
            await self._run_ssh()
        else:
            raise ValueError(f"Unknown backend {self.backend}")

    # synchronous helper (useful for debugging)
    def run(self):
        asyncio.run(self.run_async())

    # ---------------  backend implementations  ---------------
    def _run_local(self):
        runner = ExperimentRunner(
            raw_train=self.raw_train,
            raw_val=self.raw_val,
            raw_test=self.raw_test,
            experiment_config=self.config,
        )
        runner.fit().test()
        metrics = {k: float(v) for k, v in runner.trainer.callback_metrics.items(
        ) if isinstance(v, (int, float, torch.Tensor))}
        (self.job_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))

    async def _run_subprocess(self):
        # expects that `python -m experiment_wrapper.cli_run config.json` exists on PYTHONPATH
        cmd = ["python", "-m", "experiment_wrapper.cli_run",
               str(self.job_dir / "config.json")]
        proc = await asyncio.create_subprocess_exec(*cmd, cwd=self.job_dir)
        await proc.communicate()

    async def _run_ssh(self):
        if self.remote_host is None:
            raise ValueError("remote_host must be set for ssh backend")
        remote_cmd = f"python -m experiment_wrapper.cli_run {self.job_dir / 'config.json'}"
        proc = await asyncio.create_subprocess_exec("ssh", self.remote_host, remote_cmd)
        await proc.communicate()


class ExperimentSuite:
    """Run many configs (grid or random) and collate results."""

    def __init__(
        self,
        *,
        configs: Sequence[ExperimentConfig],
        raw_train,
        raw_val,
        raw_test,
        root: str | Path = "experiments",
        backend_mapper: Callable[[ExperimentConfig], str] | None = None,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(exist_ok=True, parents=True)
        self.jobs: List[ExperimentJob] = []
        for cfg in configs:
            backend = backend_mapper(cfg) if backend_mapper else "local"
            job = ExperimentJob(
                config=cfg,
                raw_train=raw_train,
                raw_val=raw_val,
                raw_test=raw_test,
                root=self.root,
                backend=backend,
            )
            self.jobs.append(job)

    # -------------------------------------------
    # execution helpers
    # -------------------------------------------

    async def run_all_async(self, max_concurrency: int = 2):
        sem = asyncio.Semaphore(max_concurrency)

        async def _worker(job: ExperimentJob):
            async with sem:
                await job.run_async()

        await asyncio.gather(*(_worker(j) for j in self.jobs))

    def run_all(self, max_concurrency: int = 2):
        asyncio.run(self.run_all_async(max_concurrency))

    # -------------------------------------------
    # results
    # -------------------------------------------

    def collate_results(self) -> pd.DataFrame:
        records: List[Dict[str, Any]] = []
        for job in self.jobs:
            mfile = job.job_dir / "metrics.json"
            if mfile.exists():
                metrics = json.loads(mfile.read_text())
                records.append(
                    {"job_id": job.job_id, **metrics, **asdict(job.config)})
        return pd.DataFrame(records)

    def save_results(self, fname: str | Path = "summary.csv") -> Path:
        df = self.collate_results()
        fpath = self.root / fname
        df.to_csv(fpath, index=False)
        return fpath


if __name__ == "__main__":
    import argparse
    import importlib

    parser = argparse.ArgumentParser(
        description="Run a single experiment via CLI")
    parser.add_argument("config", type=Path,
                        help="Path to config.json written by ExperimentJob")
    args = parser.parse_args()

    cfg_dict = json.loads(Path(args.config).read_text())
    cfg = ExperimentConfig(**cfg_dict)

    # User is responsible for loading data inside cli context – keep names generic
    data_module = importlib.import_module(
        "data.loaders")  # example placeholder
    raw_train, raw_val, raw_test = data_module.load_from_env()

    ExperimentRunner(
        raw_train=raw_train,
        raw_val=raw_val,
        raw_test=raw_test,
        experiment_config=cfg,
    ).fit().test()
