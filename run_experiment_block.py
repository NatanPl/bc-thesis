# run_block.py
"""
Launch a *block* of experiments described in a YAML/JSON file.

Example
-------
$ python run_block.py config/block1.yaml
"""

from __future__ import annotations
import argparse
import json
import yaml
from pathlib import Path

import lightning as L
import pandas as pd
import numpy as np
# from sklearn import impute, preprocessing
from preprocessing.preprocess import Imputer, Scaler

from modelling.experiment_wrapper import ExperimentConfig, ExperimentSuite
from modelling.core.datamodule import TimeSeriesDataModule, WindowConfig
# Adapt if RawDataset sits elsewhere
# ← adjust import if needed
from raw_data.dataset import Dataset as RawDataset
from raw_data.dataset import split_dataset


# --------------------------------------------------------------------------- #
#  Utilities                                                                  #
# --------------------------------------------------------------------------- #

def load_dataset(path: str) -> RawDataset:
    """Loads dataset from storage."""
    return RawDataset.load(path)


def split_country_percentage(dataset: RawDataset, test_ratio: float = 0.2, eval_ratio: float = 0.2, seed: int = 42) -> tuple[RawDataset, RawDataset, RawDataset]:
    """Split dataset into train, validation, and test sets based on country percentages."""

    np.random.seed(seed)  # Set seed for reproducibility

    # Ensure ratios are valid
    if not (0 < test_ratio < 1) or not (0 < eval_ratio < 1):
        raise ValueError("Ratios must be between 0 and 1.")

    # Calculate the number of countries for each split
    num_countries = len(dataset.countries)
    num_test = int(num_countries * test_ratio)
    num_eval = int(num_countries * eval_ratio)

    # Shuffle countries and split
    shuffled_countries = np.random.permutation(dataset.countries)
    test_countries = shuffled_countries[:num_test]
    eval_countries = shuffled_countries[num_test:num_test + num_eval]

    # Split the dataset
    test_dataset, train_dataset = split_dataset(dataset, list(test_countries))
    val_dataset, train_dataset = split_dataset(
        train_dataset, list(eval_countries))

    # Return the datasets
    return train_dataset, val_dataset, test_dataset


def split_country_specified(dataset: RawDataset, test_countries: list[str], eval_countries: list[str]) -> tuple[RawDataset, RawDataset, RawDataset]:
    """Split dataset into train, validation, and test sets based on specified countries."""

    # Ensure countries are in the dataset
    test_countries = [
        country for country in test_countries if country in dataset.countries]

    eval_countries = [
        country for country in eval_countries if country in dataset.countries]

    # Split the dataset
    test_dataset, train_dataset = split_dataset(dataset, test_countries)
    val_dataset, train_dataset = split_dataset(train_dataset, eval_countries)

    # Return the datasets
    return train_dataset, val_dataset, test_dataset


SPLITTERS = {
    "country_percentage": split_country_percentage,
    "country_specified": split_country_specified,
}

# --------------------------------------------------------------------------- #
#  Pre-processing                                                             #
# --------------------------------------------------------------------------- #


class Preprocessor:
    """Composable (fit -> transform) preprocessing pipeline."""

    def __init__(self, *, imputation: str, scaling: str):
        self.imputation = imputation
        self.scaling = scaling
        self.imputer: Imputer | None = None
        self.scaler: Scaler | None = None

        if imputation == "mean":
            self.imputer = Imputer(method="mean")
        elif imputation == "linear":
            self.imputer = Imputer(method="linear")
        elif imputation == "mice":
            self.imputer = Imputer(method="mice")
        elif imputation == "none":
            self.imputer = None
        else:
            raise ValueError(f"Unknown imputation: {imputation}")

        if scaling == "zscore":
            self.scaler = Scaler(method="zscore")
        elif scaling == "robust":
            self.scaler = Scaler(method="robust")
        elif scaling == "none":
            self.scaler = None
        else:
            raise ValueError(f"Unknown scaling: {scaling}")

    def fit(self, dataset: RawDataset) -> None:
        if self.imputer:
            self.imputer.fit(dataset)
        if self.scaler:
            self.scaler.fit(dataset)

    def transform(self, dataset: RawDataset) -> RawDataset:
        dataset = self.apply_imputer(dataset)
        dataset = self.apply_scaler(dataset)
        return dataset

    def apply_imputer(self, dataset: RawDataset) -> RawDataset:
        """Apply imputer to the dataset."""
        if self.imputer:
            dataset = self.imputer.transform(dataset)
        return dataset

    def apply_scaler(self, dataset: RawDataset) -> RawDataset:
        """Apply scaler to the dataset."""
        if self.scaler:
            dataset = self.scaler.transform(dataset)
        return dataset

# --------------------------------------------------------------------------- #
#  Main routine                                                               #
# --------------------------------------------------------------------------- #


def build_experiment_cfg(exp: dict, common: dict) -> ExperimentConfig:
    """Translate one entry from YAML -> ExperimentConfig dataclass."""
    # preprocessing spec --------------------------------------------------
    window_cfg = WindowConfig(window=5, horizon=1, stride=1)
    cfg = ExperimentConfig(
        seed=common.get("seed", 42),
        window_config=window_cfg,
        batch_size=32,
        model_kwargs=dict(hidden_dim=256),              # LSTM baseline
        **exp.get("override", {}),                      # allow per-exp tweaks
    )
    return cfg


def main(cfg: str):
    cfg_path: Path = Path(cfg)
    with cfg_path.open() as fp:
        block = yaml.safe_load(fp) if cfg_path.suffix in (
            ".yml", ".yaml") else json.load(fp)

    # metadata
    block_name = block.get("block_name", cfg_path.stem)
    output_dir = block.get("output_dir", "runs") / block_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # data
    dataset_obj = RawDataset.load(block["dataset"])

    split_method = SPLITTERS.get(block["split"]["method"])
    if not split_method:
        raise ValueError(f"Unknown split method: {block['split']['method']}")
    if split_method == split_country_percentage:
        raw_train, raw_val, raw_test = split_method(
            dataset_obj,
            test_ratio=block["split"].get("test_ratio", 0.2),
            eval_ratio=block["split"].get("val_ratio", 0.1),
            seed=block["split"].get("seed", 42),
        )
    elif split_method == split_country_specified:
        raw_train, raw_val, raw_test = split_method(
            dataset_obj,
            test_countries=block["split"].get("test_countries", []),
            eval_countries=block["split"].get("val_countries", []),
        )
    else:
        raise ValueError(f"Unknown split method: {block['split']['method']}")

    if (raw_train is None or raw_train.is_empty()) and \
       (raw_val is None or raw_val.is_empty()) and \
       (raw_test is None or raw_test.is_empty()):
        raise ValueError(
            "No data available after splitting. Check your dataset and split configuration.")

    defaults = block.get("defaults", {})

    for experiment in block["experiments"]:
        exp_name = experiment['name']
        seed = experiment.get('seed', defaults.get('seed', 42))
        np.random.seed(seed)  # Ensure reproducibility

        # preprocessing
        impute = experiment.get(
            'imputation', defaults.get('imputation', 'mice'))
        scale = experiment.get('scaling', defaults.get('scaling', 'none'))
        preprocessor = Preprocessor(imputation=impute, scaling=scale)
        preprocessor.fit(raw_train)  # Fit on training data
        raw_train = preprocessor.transform(raw_train)
        raw_val = preprocessor.transform(raw_val)
        if experiment.get('impute_test', defaults.get('impute_test', True)):
            raw_test = preprocessor.apply_imputer(raw_test)
        raw_test = preprocessor.apply_scaler(raw_test)

        # window config
        if "window" in experiment:
            window = experiment["window"].get('window',
                                              defaults.get('window', {'window': 5}).get('window', 5))
            horizon = experiment["window"].get('horizon',
                                               defaults.get('window', {'horizon': 1}).get('horizon', 1))
            stride = experiment["window"].get('stride',
                                              defaults.get('window', {'stride': 1}).get('stride', 1))
            window_cfg = WindowConfig(
                window=window, horizon=horizon, stride=stride)
        else:
            window_cfg = WindowConfig(
                window=defaults.get('window', {'window': 5}).get('window', 5),
                horizon=defaults.get(
                    'window', {'horizon': 1}).get('horizon', 1),
                stride=defaults.get('window', {'stride': 1}).get('stride', 1),
            )

        model_name = experiment.get('model', defaults.get('model', 'lstm'))
    # # ----------  build jobs -------------------------------------------------
    # exp_cfgs = []
    # preprocessors = {}
    # for exp in block["experiments"]:
    #     name = exp["name"]
    #     proc = Preprocessor(
    #         imputation=exp["imputation"], scaling=exp["scaling"])
    #     # flatten over countries
    #     proc.fit(raw_train.values.reshape(-1, raw_train.values.shape[-1]))
    #     preprocessors[name] = proc
    #     exp_cfgs.append(build_experiment_cfg(exp, block))

    # # ----------  run --------------------------------------------------------
    # suite = ExperimentSuite(
    #     configs=exp_cfgs,
    #     raw_train=raw_train,                      # *not* pre-processed yet
    #     raw_val=raw_val,
    #     raw_test=raw_test,
    #     root=block.get("output_dir", "runs") / block["block_name"],
    # )

    # # monkey-patch each job with its own pre-processed dataset
    # for job in suite.jobs:
    #     proc = preprocessors[job.config.model_kwargs.get("tag", job.job_id)]
    #     job.raw_train = apply_preprocessor(raw_train, proc)
    #     job.raw_val = apply_preprocessor(raw_val, proc)
    #     job.raw_test = apply_preprocessor(raw_test, proc)

    # suite.run_all(max_concurrency=block.get("max_concurrency", 2))
    # csv_path = suite.save_results("summary.csv")
    # json_path = csv_path.with_suffix(".json")
    # pd.read_csv(csv_path).to_json(json_path, orient="records", indent=2)
    # print(f"✅  All done. Results: {csv_path}  {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="YAML/JSON block description")
    main(parser.parse_args().config_file)
