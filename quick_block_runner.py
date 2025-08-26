from pathlib import Path
import warnings
import argparse
import traceback
import lightning as L
import torch
from modelling.experiment_wrapper import ExperimentConfig, ExperimentRunner
from modelling.core.datamodule import WindowConfig, TimeSeriesDataModule
from modelling.models.rnn import RNNmodel
from modelling.models.informer import Informer
from modelling.models.baseline import run_one_baseline
from modelling.models.spacetimeformer import Spacetimeformer
from raw_data.dataset import Dataset as RawDataset, split_dataset
from preprocessing.preprocess import Imputer, Scaler
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import re

from anova import ExpA, ExpB, ExpC

# Suppress sklearn convergence warnings
warnings.filterwarnings('ignore', category=UserWarning,
                        module='sklearn.impute._iterative')


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
            # print(f"Fitting imputer {self.imputer} with method {self.imputation}")
            self.imputer.fit(dataset)
            # print(f"Imputer fitted: {self.imputer.is_fitted}")
        if self.scaler:
            self.scaler.fit(dataset)

    def transform(self, dataset: RawDataset) -> RawDataset:
        dataset = self.apply_imputer(dataset)
        dataset = self.apply_scaler(dataset)
        return dataset

    def apply_imputer(self, dataset: RawDataset) -> RawDataset:
        """Apply imputer to the dataset."""
        if self.imputer:
            # print(f"Applying imputer: fitted={self.imputer.is_fitted}")
            dataset = self.imputer.transform(dataset)
        return dataset

    def apply_scaler(self, dataset: RawDataset) -> RawDataset:
        """Apply scaler to the dataset."""
        if self.scaler:
            dataset = self.scaler.transform(dataset)
        return dataset


def run_one(model_cls, model_kwargs, tag, train, val, test, seed=42, is_unscaled=False):
    # Adjust hyperparameters based on scaling method

    # More aggressive settings for unscaled data
    max_epochs = 200
    if is_unscaled:
        learning_rate = 1e-12  # Extremely tiny LR for unscaled data
        gradient_clip_val = 0.001  # Ultra aggressive clipping
        batch_size = 2  # Very small batches for stability
    else:
        learning_rate = 1e-4  # Normal LR for scaled data
        gradient_clip_val = 1.0  # Standard clipping
        batch_size = 32

    cfg = ExperimentConfig(
        seed=seed,
        window_config=WindowConfig(),
        batch_size=batch_size,
        max_epochs=max_epochs,
        model_cls=model_cls,
        model_kwargs=model_kwargs,
        log_every_n_steps=1,
        learning_rate=learning_rate,
        precision=32,        # Use full precision
        trainer_kwargs={
            "gradient_clip_val": gradient_clip_val,
            "gradient_clip_algorithm": "norm",  # Use gradient norm clipping
            # "detect_anomaly": is_unscaled,  # Enable anomaly detection for unscaled data
        }
    )
    runner = ExperimentRunner(
        raw_train=train, raw_val=val, raw_test=test, experiment_config=cfg)
    print(
        f"Running {tag} with model {model_cls.__name__} and kwargs {model_kwargs}")
    runner.fit().test()
    metrics = {k: float(v) for k, v in runner.trainer.callback_metrics.items()}
    return (tag, metrics)


def prep_run(train_raw: RawDataset, val_raw: RawDataset, test_raw: RawDataset, imp, scl, model_cls, model_kwargs, tag, mask_percentage: float = 0.0, seed: int = 42, touch_test: bool = False):
    """Prepare and run a single experiment."""
    train = train_raw.copy()
    val = val_raw.copy()
    test = test_raw.copy()

    # Apply artificial masking if specified
    if mask_percentage > 0.0:
        train = artificially_mask_data(train, mask_percentage, seed)
        val = artificially_mask_data(
            val, mask_percentage, seed + 1)  # Different seed for val

    preprocessor = Preprocessor(imputation=imp, scaling=scl)
    preprocessor.fit(train)
    train = preprocessor.transform(train)
    val = preprocessor.transform(val)
    if touch_test:
        test = preprocessor.transform(test)
    else:
        test = preprocessor.apply_scaler(test)

    if train.missing_values_n() > 0 or val.missing_values_n() > 0 or test.missing_values_n() > 0:
        raise Exception("Preprocessing did not remove all missing values!")

    return run_one(model_cls, model_kwargs, tag, train, val, test, seed, is_unscaled=(scl == 'none'))


def artificially_mask_data(dataset: RawDataset, mask_percentage: float, seed: int = 42) -> RawDataset:
    """
    Artificially mask a percentage of non-missing values in the dataset.

    Args:
        dataset: The dataset to mask
        mask_percentage: Percentage of values to mask (0.0 to 1.0)
        seed: Random seed for reproducibility

    Returns:
        Dataset with additional masked values
    """
    if mask_percentage <= 0.0:
        return dataset

    np.random.seed(seed)
    dataset_copy = dataset.copy()

    # Find all non-missing values
    non_missing_mask = ~np.isnan(dataset_copy.data)
    non_missing_indices = np.where(non_missing_mask)

    # Calculate how many to mask
    n_non_missing = len(non_missing_indices[0])
    n_to_mask = int(n_non_missing * mask_percentage)

    if n_to_mask > 0:
        # Randomly select indices to mask
        mask_indices = np.random.choice(
            n_non_missing, size=n_to_mask, replace=False)

        # Apply masking
        for idx in mask_indices:
            i, j, k = non_missing_indices[0][idx], non_missing_indices[1][idx], non_missing_indices[2][idx]
            dataset_copy.data[i, j, k] = np.nan

    print(
        f"Masked {n_to_mask} additional values ({mask_percentage*100:.1f}% of non-missing data)")
    print(f"Total missing values: {dataset_copy.missing_values_n()}")

    return dataset_copy


def run_multiple_seeds(train_raw, val_raw, test_raw, imp, scl, model_cls, model_kwargs, base_tag, seeds, mask_percentage=0.0, touch_test=False):
    """Run experiment with multiple seeds and return both averaged and per-seed results."""
    all_results = []

    for seed in seeds:
        tag = f"{base_tag}_seed{seed}"
        result_tag, metrics = prep_run(
            train_raw, val_raw, test_raw, imp, scl, model_cls, model_kwargs, tag, mask_percentage, seed, touch_test)
        all_results.append(metrics)
        print(f"Seed {seed} results: {metrics}")

    # Calculate averages and standard deviations
    metric_names = all_results[0].keys()
    averaged_results = {}

    for metric in metric_names:
        values = [result[metric] for result in all_results]
        # Filter out NaN values for averaging
        valid_values = [v for v in values if not (
            np.isnan(v) if isinstance(v, float) else False)]

        if valid_values:
            averaged_results[f"{metric}_mean"] = np.mean(valid_values)
            averaged_results[f"{metric}_std"] = np.std(valid_values)
            averaged_results[f"{metric}_n_valid"] = len(valid_values)
            # Store per-seed values for ANOVA
            averaged_results[f"seed_{metric}"] = valid_values
        else:
            averaged_results[f"{metric}_mean"] = np.nan
            averaged_results[f"{metric}_std"] = np.nan
            averaged_results[f"{metric}_n_valid"] = 0
            averaged_results[f"seed_{metric}"] = []

    return base_tag, averaged_results


def run_baseline_multiple_seeds(train_raw, val_raw, test_raw,
                                imp, scl,
                                kind, base_tag, seeds,
                                mask_percentage=0.0, p=2, touch_test=False):
    """
    Mirror run_multiple_seeds but call stat baseline instead of Lightning.
    Deterministic: same metrics every seed; still loop for compatibility.
    """
    all_results = []
    for seed in seeds:
        tag = f"{base_tag}_{kind}_seed{seed}"
        # preprocessing same as prep_run (masking + fit/transform)
        train = train_raw.copy()
        val = val_raw.copy()
        test = test_raw.copy()
        if mask_percentage > 0.0:
            train = artificially_mask_data(train, mask_percentage, seed)
            val = artificially_mask_data(val, mask_percentage, seed + 1)
        preprocessor = Preprocessor(imputation=imp, scaling=scl)
        preprocessor.fit(train)
        train = preprocessor.transform(train)
        val = preprocessor.transform(val)
        test = preprocessor.transform(
            test) if touch_test else preprocessor.apply_scaler(test)
        if train.missing_values_n() or val.missing_values_n() or test.missing_values_n():
            raise RuntimeError("Baseline preprocessing left NaNs.")
        _, metrics = run_one_baseline(kind, tag, train, val, test, p=p)
        all_results.append(metrics)

    # aggregate like run_multiple_seeds
    averaged = {}
    metric_names = all_results[0].keys()
    for m in metric_names:
        vals = [r[m] for r in all_results]
        averaged[f"{m}_mean"] = float(np.mean(vals))
        averaged[f"{m}_std"] = float(np.std(vals, ddof=1))
        averaged[f"{m}_n_valid"] = len(vals)
        averaged[f"seed_{m}"] = vals
    return base_tag, averaged


def experimentA():
    DATA = "data/datasets/oecd_2000_80_t.h5"
    dataset = RawDataset.load(DATA)
    imputation_methods = ["mice", "linear", "mean"]
    scaling_methods = ["zscore", "robust"]
    missingness_percentages = [0.0, 0.1, 0.2, 0.3]
    test_countries = ['CAN', 'CHE', 'DEU', 'FIN', 'FRA', 'GBR', 'HUN']
    remaining_countries = list(set(dataset.countries) - set(test_countries))
    take = len(test_countries)
    val_countries = remaining_countries[:take]
    train_raw, val_raw, test_raw = split_country_specified(
        dataset, test_countries, val_countries)
    results = []
    for imp in imputation_methods:
        for scl in scaling_methods:
            for mask_pct in missingness_percentages:
                experiment_name = f"experimentA_{imp}_{scl}_mask{mask_pct*100:.0f}"
                try:
                    result_tag, averaged_metrics = run_multiple_seeds(
                        train_raw, val_raw, test_raw,
                        imp, scl,
                        RNNmodel, {"cell_type": "lstm",
                                   "hidden_dim": 256, "num_layers": 1},
                        experiment_name, SEEDS, mask_pct
                    )
                    results.append((result_tag, averaged_metrics))

                except Exception as e:
                    print(f"Error running {experiment_name}: {e}")
                    continue

    file = FileName("experimentA")
    report_results(results, file, "experimentA")

    ExpA(results=results).run()


def experimentB():
    DATA70 = "data/datasets/oecd_2000_70.h5"
    DATA80 = "data/datasets/oecd_2000_80.h5"
    DATA90 = "data/datasets/oecd_2000_90.h5"
    imputer = 'linear'
    scaler = 'robust'
    results = []
    for DATA, tag in [(DATA70, '70p'), (DATA80, '80p'), (DATA90, '90p')]:
        dataset = RawDataset.load(DATA)
        countries = list(dataset.countries)
        test_countries = countries[-int((dataset.n_countries() * 0.2)):]
        eval_countries = countries[:int((dataset.n_countries() * 0.2))]
        train_raw, val_raw, test_raw = split_country_specified(
            dataset, test_countries, eval_countries)
        experiment_name = f"experimentB_{tag}"
        try:
            result_tag, averaged_metrics = run_multiple_seeds(
                train_raw, val_raw, test_raw,
                imputer, scaler,
                RNNmodel, {"cell_type": "lstm",
                           "hidden_dim": dataset.n_indicators() * 2, "num_layers": 1},
                experiment_name, SEEDS, mask_percentage=0.0,
                touch_test=True  # Ensure test set is preprocessed
            )
            results.append((result_tag, averaged_metrics))
        except Exception as e:
            print(f"Error running {experiment_name}: {e}")
            continue
    file = FileName("experimentB")
    report_results(results, file, "experimentB")

    ExpB(results=results).run()


def experimentC():
    DATA = "data/datasets/oecd_2000_80.h5"
    models = [
        (Informer, {
            "model_dim": 256,
            "n_heads": 8,
            "d_ff": 512,
            "attention_factor": 3,
            "encoder_layers": 3,
            "decoder_layers": 2,
            "dropout_rate": 0.1,
            "attention": "prob_sparse",
            "activation": "gelu"
        }, "Informer_256"),
        (Spacetimeformer, {
            "label_len": 5,
            "model_dim": 256,
            "n_heads": 8,
            "d_ff": 512,
            "encoder_layers": 3,
            "decoder_layers": 2,
            "dropout_rate": 0.1,
            "attention": "full",
            "activation": "gelu",
            "freq": "a"
        }, "Spacetimeformer_256"),
        (RNNmodel, {"cell_type": "lstm",
         "hidden_dim": 256, "num_layers": 2, "dropout_rate": 0.2}, "LSTM_256"),
        (RNNmodel, {"cell_type": "gru",
         "hidden_dim": 256, "num_layers": 2, "dropout_rate": 0.2}, "GRU_256"),
    ]
    imputer = 'linear'
    scaler = 'robust'
    results = []
    dataset = RawDataset.load(DATA)
    countries = list(dataset.countries)
    test_countries = countries[-int((dataset.n_countries() * 0.2)):]
    eval_countries = countries[:int((dataset.n_countries() * 0.2))]
    train_raw, val_raw, test_raw = split_country_specified(
        dataset, test_countries, eval_countries)

    for model_cls, model_kwargs, model_name in models:
        experiment_name = f"experimentC_{model_name}"
        try:
            result_tag, averaged_metrics = run_multiple_seeds(
                train_raw, val_raw, test_raw,
                imputer, scaler,
                model_cls, model_kwargs,
                experiment_name, SEEDS, mask_percentage=0.0,
                touch_test=True  # Ensure test set is preprocessed
            )
            results.append((result_tag, averaged_metrics))
        except Exception as e:
            print(f"Error running {experiment_name}: {e}")
            continue

    for kind, p in [("persistence", None), ("ar1", None), ("var", 2)]:
        exp_name = f"experimentC_{kind}"
        try:
            tag, metrics = run_baseline_multiple_seeds(
                train_raw, val_raw, test_raw,
                imputer, scaler,
                kind, exp_name, SEEDS,
                mask_percentage=0.0,
                p=p if p is not None else 2,  # p used only for VAR
                touch_test=True
            )
            results.append((tag, metrics))
        except Exception as e:
            print(f"Error running {exp_name}: {e}")
            continue

    print(results)

    file = FileName("experimentC")
    report_results(results, file, "experimentC")

    ExpC(results=results).run()


def FileName(experiment_name):
    from datetime import datetime
    current_time = datetime.now().strftime("%m%d_%H%M%S")
    return f'runs/results_{experiment_name}_{current_time}.json'


def report_results(results, file_name, experiment_name):
    import json
    # write results to a file in JSON format
    # check if directory exists, if not create it
    # Change file extension to .json
    if file_name.endswith('.txt'):
        file_name = file_name[:-4] + '.json'
    elif not file_name.endswith('.json'):
        file_name += '.json'
    
    Path(file_name).parent.mkdir(parents=True, exist_ok=True)
    
    # Structure the data for JSON
    json_data = {
        "experiment_name": experiment_name,
        "results": []
    }
    
    for result_tag, metrics in results:
        json_data["results"].append({
            "tag": result_tag,
            "metrics": metrics
        })
    
    with open(file_name, 'w') as f:
        json.dump(json_data, f, indent=2)


SEEDS = [42, 123, 456, 973, 1023, 9124, 2150, 7562, 4855, 3]

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run quick block experiments")
    parser.add_argument("-a", action="store_true",
                        help="Run experiment A")
    parser.add_argument("-b", action="store_true",
                        help="Run experiment B")
    parser.add_argument("-c", action="store_true",
                        help="Run experiment C")
    args = parser.parse_args()
    if args.a:
        try:
            experimentA()
        except Exception as e:
            print(f"Error running experiment A: {e}")
            import traceback
            traceback.print_exc()
    if args.b:
        try:
            experimentB()
        except Exception as e:
            print(f"Error running experiment B: {e}")
            import traceback
            traceback.print_exc()
    if args.c:
        try:
            experimentC()
        except Exception as e:
            print(f"Error running experiment C: {e}")
            import traceback
            traceback.print_exc()
