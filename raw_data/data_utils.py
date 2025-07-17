from pathlib import Path
import pandas as pd
import numpy as np
import json


class DataUtils:
    """
    A utility class to handle common operations like file handling, indicator validation,
    and dataset processing.
    """

    @staticmethod
    def load_hdf(file_path: Path, key: str):
        """
        Load data from an HDF5 file.

        :param file_path: Path to the HDF5 file.
        :param key: Key within the HDF5 file.
        :return: Loaded DataFrame or None if the file does not exist.
        """
        if file_path.exists():
            try:
                return pd.read_hdf(file_path, key)
            except (KeyError, TypeError) as e:
                print("An error occurred while reading the HDF file:", e)
                return None
        return None

    @staticmethod
    def save_hdf(data: pd.DataFrame, file_path: Path, key: str):
        """
        Save a DataFrame to an HDF5 file.

        :param data: DataFrame to save.
        :param file_path: Path to the HDF5 file.
        :param key: Key within the HDF5 file.
        """
        file_path.parent.mkdir(parents=True, exist_ok=True)
        data.to_hdf(file_path, key=key, mode="w", format="table")

    @staticmethod
    def validate_indicators(
        indicators: list[str], valid_indicators: set[str]
    ) -> list[str]:
        """
        Validate and filter a list of indicators against a set of valid indicators.

        :param indicators: List of indicator codes to validate.
        :param valid_indicators: Set of valid indicator codes.
        :return: List of valid indicators.
        """
        return [indicator for indicator in indicators if indicator in valid_indicators]

    @staticmethod
    def generate_empty_dataset(indicators: list[str], years: list[int]) -> pd.DataFrame:
        """
        Generate an empty dataset with indicators as rows and years as columns.

        :param indicators: List of indicators.
        :param years: List of years.
        :return: Empty DataFrame with the specified rows and columns.
        """
        return pd.DataFrame(index=indicators, columns=years, dtype=float)

    @staticmethod
    def load_json(file_path: Path) -> dict:
        """
        Load JSON data from a file.

        :param file_path: Path to the JSON file.
        :return: Loaded JSON data as a dictionary.
        """
        if file_path.exists():
            with open(file_path, "r") as file:
                return json.load(file)
        return {}

    @staticmethod
    def save_json(data: dict, file_path: Path):
        """
        Save a dictionary as JSON to a file.

        :param data: Dictionary to save.
        :param file_path: Path to the JSON file.
        """
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w") as file:
            json.dump(data, file)

    @staticmethod
    def sort_dataframe(data: pd.DataFrame) -> pd.DataFrame:
        """
        Sort a DataFrame by its index and columns.

        :param data: DataFrame to sort.
        :return: Sorted DataFrame.
        """
        return data.sort_index(axis=0).sort_index(axis=1)
