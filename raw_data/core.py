# =============================================================================
# Core API module for economic development forecasting.
#
# This module serves as the main API interface between backend processing
# modules and UI implementations. It coordinates data downloading, processing,
# model creation, # and prediction functionality by integrating all backend
# modules together.
# =============================================================================
import multiprocessing as mp
from pathlib import Path
import math
import random
import numpy as np
from typing import Iterable
from raw_data._core.getters import Getters
from raw_data._core.downloader import Downloader

import raw_data.country_data_t
from raw_data.documentation_structs import *
from raw_data.dataset import Dataset

import datetime
import os
import time

import pandas as pd
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


class Core(Getters, Downloader):
    """
    Core class for managing economic forecasting operations.

    This class serves as the central coordinator for all operations, including:
    - Data downloading and management
    - Country and indicator selection
    - Dataset creation and manipulation
    - Gap filling in datasets
    - Model creation, training and prediction

    It acts as the main API interface between the backend modules and UI implementations.
    """

    def __init__(
        self,
        folders: dict[str, Path] = None,
        multiprocessing: bool = True,
    ):
        """Initialize the Core API interface.

        Args:
            folders: Dictionary of folder paths for configuration, data, reports, models, and logs
            multiprocessing: Whether to use multiprocessing for data operations
        """
        if folders is None:
            print("No folders provided, using default paths.")
            folders = {
                "config": Path("config"),
                "data": Path("data"),
            }
        # Folders for storage
        for attribute, key in (("folder_config", "config"), ("folder_data", "data")):
            setattr(self, attribute, folders.get(key, Path(key)))
            getattr(self, attribute).mkdir(exist_ok=True)

        self.countries_data = object_collection_t(
            countries=True, folder_path=self.folder_config
        )
        self.indicators_data = object_collection_t(
            indicators=True, folder_path=self.folder_config
        )

        self.min_year = 0
        self.max_year = 0
        self.selected_countries = set()
        self.selected_indicators = set()
        self.selected_years = (0, 0)
        self.available_countries = set()
        self.available_indicators = set()

        self.multiprocessing = multiprocessing

        self.lookback_period = 5

        if self.multiprocessing:
            self.initialize_pooling()

        self.update_available_information()

    def initialize_pooling(self):
        """Initialize the multiprocessing pool for parallel data operations."""
        self.pool_initialized = True
        mp.freeze_support()
        self.Pool = mp.Pool(20, maxtasksperchild=3)
        self.Manager = mp.Manager()
        self.Queue = self.Manager.Queue()

    def update_available_information(self) -> None:
        print(self.folder_data)
        """
        Update internal lists of available countries and indicators based on existing data files.

        This method scans the data directories and updates:
        - available_countries
        - available_indicators
        - min_year and max_year for which data is available
        """
        self.available_countries.clear()
        self.available_indicators.clear()
        self.min_year = 9999
        self.max_year = 0
        self.countries_data.load_objects()
        self.indicators_data.load_objects()
        data_files = set()
        for dir in ["raw_data", "mod_data"]:
            for file in (self.folder_data / dir).glob("*.h5"):
                country_code = file.stem
                if "_mask" in country_code:
                    continue
                if country_code not in self.countries_data.objects.keys():
                    print(
                        f"Country {country_code} not in countries.json, but in {dir}."
                    )
                    continue
                data_files.add(country_code)

        for country_code in data_files:
            self.available_countries.add(country_code)
            data_object = country_data_t(country_code, self.folder_data)
            data_object.load_data()
            if data_object.data_raw is None:
                continue
            self.countries_data.objects[country_code].load_data(data_object)
            self.min_year = min(
                self.min_year, self.countries_data.objects[country_code].min_year(
                )
            )
            self.max_year = max(
                self.max_year, self.countries_data.objects[country_code].max_year(
                )
            )
            self.available_indicators.update(
                set(
                    self.countries_data.objects[
                        country_code
                    ].list_indicators_with_data()
                )
            )

        self.updated = True

    def create_dataset(self, countries: Iterable[str], indicators: Iterable[str], years: tuple[int, int], show_modified: bool = False) -> Dataset:
        data = np.zeros(
            (
                len(countries),
                len(indicators),
                years[1] - years[0] + 1,
            )
        )
        if show_modified:
            mod_array = np.full_like(data, False, dtype=bool)
        countries = pd.Index(sorted(list(countries)))
        indicators = pd.Index(sorted(list(indicators)))
        years = pd.Index(
            range(years[0], years[1] + 1), name="Year"
        )
        for i, country in enumerate(countries):
            for j, indicator in enumerate(indicators):
                for k, year in enumerate(years):
                    data_object: country_data_t = self.countries_data.objects[country].data_object
                    try:
                        data[i, j, k] = data_object.data.loc[indicator, year]
                    except Exception:
                        data[i, j, k] = np.nan
                    if show_modified:
                        # check if the data_object.data_mod exists
                        if data_object.data_mod is None:
                            continue
                        if not np.isnan(data_object.data_mod.loc[indicator, year]):
                            mod_array[i, j, k] = True
        if show_modified:
            return Dataset(data, countries, indicators, years, mod_array)
        return Dataset(data, countries, indicators, years)

    def create_selected_dataset(self) -> Dataset:
        """
        Create a Dataset object containing the selected data.

        Returns:
            Dataset: A dataset object containing the selected countries, indicators and time periods
                    or an empty Dataset if no selections have been made
        """
        if (
            not self.selected_countries
            or not self.selected_indicators
            or not self.selected_years
        ):
            return Dataset(None, None, None, None)
        countries = [country.code for country in self.selected_countries]
        indicators = [indicator.code for indicator in self.selected_indicators]
        return self.create_dataset(
            countries, indicators, self.selected_years
        )

    def resolve_country_code(self, country_name: str) -> str:
        """
        Convert a country name to its corresponding code.

        Args:
            country_name: Full name of the country

        Returns:
            Country code string, or empty string if not found
        """
        for country in self.countries_data.objects.values():
            if country.name == country_name:
                return country.code
        return ""

    def close(self) -> bool:
        """
        Clean up resources and save configuration.

        This method should be called when shutting down the application.

        Returns:
            True if shutdown was successful
        """
        if self.pool_initialized:
            self.Pool.close()
            self.Pool.join()
            self.Pool.terminate()
            self.Pool = None
            self.pool_initialized = False
        self.countries_data.save_objects()
        self.indicators_data.save_objects()
        return True

    def prepare_modeling_data(self,
                              holdout_years: bool = False, holdout_countries: bool = False,
                              ratio_train: float = 0.8, ratio_eval: float = 0.2, ratio_test: float = 0.0,
                              test_years: int = 5, test_countries: list[str] = None) -> tuple[
        tuple[np.ndarray, np.ndarray],
        tuple[np.ndarray, np.ndarray],
        tuple[np.ndarray, np.ndarray],
        set[str]
    ]:
        """
        Split the dataset into training, testing, and validation sets based on the specified parameters.

        Args:
            holdout_years: Whether to use holdout years for the final testing
            holdout_countries: Whether to use holdout countries for the final testing
            ratio_train: Ratio of data to use for training
            ratio_eval: Ratio of data to use for validation during training
            ratio_test: Ratio of data to use for final testing after the training is done
            test_years: Number of years from the end of the dataset to holdout for the final testing
            test_countries: List of country codes to use for testing

        Returns:
            Tuple of training, evaluation, and testing data sets as input-output pairs as well as the set of indicators used
        """
        def concat_data(datasets: list[Dataset], X=None, y=None) -> tuple[np.ndarray, np.ndarray]:
            if not datasets:
                return X, y  # Return input X, y if datasets is empty

            for data in datasets:
                data_X, data_y, _ = data.prepare_training_data(
                    self.lookback_period)
                if X is None:
                    X, y = data_X, data_y
                elif data_X.size > 0:  # Only concatenate if data_X is not empty
                    # Check if shapes are compatible for concatenation
                    # Compare all dimensions except the first one
                    if X.shape[1:] == data_X.shape[1:]:
                        X = np.concatenate((X, data_X), axis=0)
                        y = np.concatenate((y, data_y), axis=0)
                    else:
                        print(
                            f"Warning: Incompatible shapes for concatenation: {X.shape} and {data_X.shape}")

            return X, y

        if (
            not self.selected_countries
            or not self.selected_indicators
            or not self.selected_years
        ):
            # sanity check for selected data
            return None

        if ratio_train + ratio_eval + ratio_test != 1:
            raise ValueError(
                "Training, evaluation, and testing ratios must sum to 1")

        test_data = []
        train_X, train_y, eval_X, eval_y, test_X, test_y = None, None, None, None, None, None

        all_countries = [country.code for country in self.selected_countries]
        all_indicators = [
            indicator.code for indicator in self.selected_indicators]
        all_years = list(
            range(self.selected_years[0], self.selected_years[1] + 1))

        if holdout_countries:
            if test_countries is None:
                raise ValueError(
                    "Country test list must be provided when using country-based testing.")
            # sanity check that given test countries are in selected countries
            if any(country not in all_countries for country in test_countries):
                raise ValueError(
                    "Test countries must be in the selected dataset")
            all_countries = [
                country for country in all_countries if country not in test_countries]

            test_data.append(self.create_dataset(
                test_countries, all_indicators, (all_years[0], all_years[-1])))

        if holdout_years:
            if test_years == 0:
                raise ValueError(
                    "Test years cannot be zero when using year-based testing.")
            if test_years < self.lookback_period:
                raise ValueError(
                    "Test years must be greater or equal to lookback period")
            if test_years > len(all_years) - self.lookback_period:
                raise ValueError("Test years exceed the available data range")
            test_years_list = all_years[-test_years:]
            all_years = all_years[:-test_years]
            test_year_range = (test_years_list[0], test_years_list[-1])

            test_data.append(self.create_dataset(
                all_countries, all_indicators, test_year_range))

        if test_data:
            test_X, test_y = concat_data(test_data)

        if ratio_test > 0:
            # calculate the current test data size
            test_data_count = 0 if test_X is None else test_X.shape[0]
            # estimate the remaining data count
            remaining_data_count = len(
                all_years) * len(all_countries) - test_data_count
            actual_ratio = test_data_count / \
                (test_data_count + remaining_data_count)
            if actual_ratio < ratio_test:
                # calculate the number of additional countries needed to reach the desired ratio
                needed_ratio = ratio_test - actual_ratio
                needed_countries = math.floor(
                    needed_ratio * len(all_countries))
                # select the needed countries from the remaining countries
                random.shuffle(all_countries)
                selected_countries = all_countries[:needed_countries]
                all_countries = all_countries[needed_countries:]
                # add them to the test data
                data = self.create_dataset(
                    selected_countries, all_indicators, (all_years[0], all_years[-1]))
                test_X, test_y = concat_data([data], test_X, test_y)

        # Create the training and evaluation datasets
        country_size_ratio = (
            self.selected_years[1] - self.selected_years[0] + 1) / len(all_years)
        train_eval_ratio = ratio_train / (ratio_train + ratio_eval)
        train_country_count = math.floor(len(all_countries) * train_eval_ratio)
        random.shuffle(all_countries)
        train_countries = all_countries[:train_country_count]
        eval_countries = all_countries[train_country_count:]
        train_data = self.create_dataset(
            train_countries, all_indicators, (all_years[0], all_years[-1]))
        eval_data = self.create_dataset(
            eval_countries, all_indicators, (all_years[0], all_years[-1]))
        train_X, train_y, _ = train_data.prepare_training_data(
            self.lookback_period)
        eval_X, eval_y, _ = eval_data.prepare_training_data(
            self.lookback_period)

        # Return empty arrays instead of None when no data
        if test_X is None:
            test_X = np.array([])
            test_y = np.array([])

        return (train_X, train_y), (eval_X, eval_y), (test_X, test_y), set(all_indicators)

    def selected_data_complete(self) -> bool:
        """
        Check if the selected data is complete and can be used for modeling.

        Returns:
            True if the selected data contains all the data
        """
        if not self.selected_countries or not self.selected_indicators or not self.selected_years:
            # print("selected_data_incomplete: no countries, indicators or years selected")
            return False
        for country in self.selected_countries:
            # check if the data object is loaded
            if not country.data_object:
                # print("selected_data_incomplete: data object not loaded")
                return False
            data_object: country_data_t = country.data_object
            # check if dataframe exists
            if data_object.data is None:
                # print("selected_data_incomplete: data object data is None")
                return False
            data: pd.DataFrame = data_object.data
            # check that the indices are present
            if any(indicator.code not in data.index for indicator in self.selected_indicators):
                # print("selected_data_incomplete: indicator not in data")
                return False
            # check that the years are present
            if any(year not in data.columns for year in range(self.selected_years[0], self.selected_years[1] + 1)):
                # print("selected_data_incomplete: year not in data")
                return False
            # check that the actual data is not missing
            if any(np.isnan(data.loc[indicator.code, year]) for indicator in self.selected_indicators for year in range(self.selected_years[0], self.selected_years[1] + 1)):
                # print("selected_data_incomplete: data is missing")
                return False
        # print("selected_data_complete: all checks passed")
        return True
