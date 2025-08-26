import numpy as np
import pandas as pd
import pickle
import json
import os
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

# Optional imports for HDF5 support
try:
    import h5py
    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False


class Dataset:
    """
    Dataset class encapsulating 3-dimensional numpy array with additional metadata.

    This class provides a structured way to store and access economic indicator data
    with the following dimensions:
    - First dimension (axis 0): Countries
    - Second dimension (axis 1): Indicators
    - Third dimension (axis 2): Years/Time periods

    It also provides utility methods to check data quality, extract subsets,
    and prepare data for model training.
    """

    def __init__(
        self,
        data: np.ndarray,
        countries: pd.Index,
        indicators: pd.Index,
        years: pd.Index,
        modified: np.ndarray = None
    ):
        """
        Initialize a new Dataset object with data and metadata.

        :param data: 3-dimensional numpy array with shape (n_countries, n_indicators, n_years)
        :param countries: pandas Index with country codes
        :param indicators: pandas Index with indicator codes
        :param years: pandas Index with years
        """
        self.data: np.ndarray = data
        """[countries, indicators, years]"""
        self.countries: pd.Index = countries
        self.indicators: pd.Index = indicators
        self.years: pd.Index = years
        if modified is not None:
            self.is_modified: np.ndarray = modified
        if any(dim is None for dim in [data, countries, indicators, years]):
            return
        if self.data.shape != (len(countries), len(indicators), len(years)):
            raise ValueError(
                f"Data shape mismatch: expected ({len(countries)}, {len(indicators)}, {len(years)}) but got {self.data.shape}"
            )

    def is_empty(self) -> bool:
        """
        Check if the dataset is empty.

        Returns:
            bool: True if any dimension of the dataset is empty, False otherwise
        """
        return any(
            dim is None or dim.size == 0
            for dim in [self.data, self.countries, self.indicators, self.years]
        )

    def any_nan(self) -> bool:
        """
        Check if any value in the dataset is NaN.

        This is useful to verify data completeness before analysis.

        Returns:
            bool: True if any value is NaN, False otherwise
        """
        return np.isnan(self.data).any()

    def n_countries(self) -> int:
        """
        Return the number of countries in the dataset.

        Returns:
            int: Count of countries
        """
        return 0 if self.countries is None else len(self.countries)

    def n_indicators(self) -> int:
        """
        Return the number of indicators in the dataset.

        Returns:
            int: Count of economic indicators
        """
        return 0 if self.indicators is None else len(self.indicators)

    def n_years(self) -> int:
        """
        Return the number of years in the dataset.

        Returns:
            int: Count of time periods
        """
        return len(self.years)

    def missing_countries(self) -> pd.Index:
        """
        Return the countries with some missing data.

        Identifies countries that have at least one NaN value across any
        indicator or year.

        Returns:
            pd.Index: Index of countries with missing data
        """
        return self.countries[np.isnan(self.data).any(axis=(1, 2))]

    def missing_indicators(self) -> pd.Index:
        """
        Return the indicators with some missing data.

        Identifies indicators that have at least one NaN value across any
        country or year.

        Returns:
            pd.Index: Index of indicators with missing data
        """
        return self.indicators[np.isnan(self.data).any(axis=(0, 2))]

    def missing_years(self) -> pd.Index:
        """
        Return the years with some missing data.

        Identifies years/time periods that have at least one NaN value across any
        country or indicator.

        Returns:
            pd.Index: Index of years with missing data
        """
        return self.years[np.isnan(self.data).any(axis=(0, 1))]

    def presence_heatmap(self) -> np.ndarray:
        """
        Return a 2D heatmap of presence of data in the dataset.

        Creates a matrix where each cell represents the count of non-NaN values
        for a specific indicator (row) and year (column) across all countries.

        Returns:
            np.ndarray: 2D array with shape (n_indicators, n_years)
        """
        if self.is_empty():
            return np.ndarray((0, 0))
        return np.sum(~np.isnan(self.data), axis=0)

    def prepare_training_data(
        self, lookback_period: int = 5
    ) -> tuple[np.ndarray, np.ndarray, set[str]]:
        """
        Prepare training data for LSTM model.

        Formats the dataset into sequences suitable for time series forecasting models,
        particularly LSTM networks. Creates sliding windows of historical data (X)
        and their corresponding next values (y) for each country and indicator.

        :param lookback_period: Number of years to include in each input sequence
        :return: Tuple of (X, y, set) where:
                 - X is a 3D array with shape (n_samples, lookback_period, n_indicators)
                 - y is a 2D array with shape (n_samples, n_indicators)
                 - set of indicator codes to denote the dataset used
        """

        if self.is_empty():
            return np.ndarray((0, 0, 0)), np.ndarray((0, 0)), set()

        X = []
        y = []
        indicators_set = self.indicator_set()
        # Iterate through each country
        for country in range(self.n_countries()):
            # Create sliding window samples for this country
            for i in range(self.n_years() - lookback_period):
                # Input sequence: lookback_period consecutive years of all indicators
                X.append(self.data[country, :, i: i + lookback_period].T)
                # Target: values for all indicators in the next year
                y.append(self.data[country, :, i + lookback_period])

        return np.array(X), np.array(y), indicators_set

    def shape(self) -> tuple[int, int, int]:
        """
        Return the shape of the dataset as a tuple (n_countries, n_indicators, n_years).

        Returns:
            tuple: Shape of the dataset
        """
        return self.data.shape

    def size_n(self) -> int:
        """
        Return the total number of values in the dataset.

        Returns:
            int: Total count of values
        """
        return np.prod(self.data.shape)

    def missing_values_n(self) -> int:
        """
        Return the number of missing (NaN) values in the dataset.

        Returns:
            int: Count of missing values
        """
        return np.isnan(self.data).sum()

    def missing_ratio(self) -> float:
        """
        Return the ratio of missing values to total values in the dataset.

        Returns:
            float: Ratio of missing values
        """
        return self.missing_values_n() / self.size_n()

    def missing_countries_per_indicator(self) -> dict:
        """
        Return a dictionary with the list of countries not having at least one value for each indicator.
        """
        indicator_missing = {}
        for i, indicator in enumerate(self.indicators):
            # Get the countries with missing values for this indicator
            missing_countries = self.countries[np.isnan(
                self.data[:, i, :]).all(axis=1)]
            if missing_countries.size > 0:
                indicator_missing[indicator] = missing_countries.tolist()
        return indicator_missing

    def country_report(self, country: str) -> dict:
        """
        Generate a report on data completeness for given country.

        Args:
            country: Country code to generate report for
        """

        if country not in self.countries:
            raise ValueError(f"Country '{country}' not found in dataset")

        country_idx = self.countries.get_loc(country)
        country_data = self.data[country_idx]

        missing_values = np.isnan(country_data).sum(axis=1)
        missing_ratio = missing_values / self.n_years() * self.n_indicators()

        return {
            "missing_values": missing_values,
            "missing_ratio": missing_ratio,
        }

    def indicator_report(self, indicator: str) -> dict:
        """
        Generate a report on data completeness for given indicator.

        Args:
            indicator: Indicator code to generate report for
        """

        if indicator not in self.indicators:
            raise ValueError(f"Indicator '{indicator}' not found in dataset")

        indicator_idx = self.indicators.get_loc(indicator)
        indicator_data = self.data[:, indicator_idx]

        missing_values = np.isnan(indicator_data).sum(axis=1)
        missing_ratio = missing_values / self.n_years() * self.n_countries()

        return {
            "missing_values": missing_values,
            "missing_ratio": missing_ratio,
        }

    def year_report(self, year: int) -> dict:
        """
        Generate a report on data completeness for given year.

        Args:
            year: Year to generate report for
        """

        if year not in self.years:
            raise ValueError(f"Year '{year}' not found in dataset")

        year_idx = self.years.get_loc(year)
        year_data = self.data[:, :, year_idx]

        missing_values = np.isnan(year_data).sum(axis=1)
        missing_ratio = missing_values / self.n_countries() * self.n_indicators()

        return {
            "missing_values": missing_values,
            "missing_ratio": missing_ratio,
        }

    def indicator_set(self) -> set:
        """
        Return a set of the indicator codes in the dataset.

        Returns:
            set: Set of indicator codes
        """
        return set(self.indicators)

    def extract_time_series(self):
        """
        Returns a list of time series for each indicator and year, suitable for line plots.
        Is accompanied by is_modified array to help highlight modified data.
        """
        time_series = []
        if self.is_empty():
            return time_series

        years = list(self.years)

        for c, country in enumerate(self.countries):
            for i, indicator in enumerate(self.indicators):
                data_ts = self.data[c, i, :]
                dic = {
                    "country": country,
                    "indicator": indicator,
                    "years": years,
                    "data": data_ts,
                }
                if hasattr(self, "is_modified"):
                    dic["is_modified"] = self.is_modified[c, i, :]

                time_series.append(dic)
        return time_series

    def indicator_variance(self, indicator: str) -> float:
        """
        Calculate the variance of a specific indicator across all countries and years.

        Args:
            indicator: Indicator code to check

        Returns:
            float: Variance value
        """
        if indicator not in self.indicators:
            raise ValueError(f"Indicator '{indicator}' not found in dataset")

        indicator_idx = self.indicators.get_loc(indicator)
        return np.nanvar(self.data[:, indicator_idx, :])

    def indicator_variance_per_country(self, indicator: str) -> pd.Series:
        """
        Calculate the variance of a specific indicator for each country.

        Args:
            indicator: Indicator code to check

        Returns:
            pd.Series: Series with country codes as index and variance values as values
        """
        if indicator not in self.indicators:
            raise ValueError(f"Indicator '{indicator}' not found in dataset")

        indicator_idx = self.indicators.get_loc(indicator)
        variances = np.nanvar(self.data[:, indicator_idx, :], axis=1)

        return pd.Series(variances, index=self.countries)

    def indicator_index(self, indicator: str) -> int:
        """
        Get the index of a specific indicator in the dataset.

        Args:
            indicator: Indicator code to find

        Returns:
            int: Index of the indicator
        """
        if indicator not in self.indicators:
            raise ValueError(f"Indicator '{indicator}' not found in dataset")

        return self.indicators.get_loc(indicator)

    def indicator_completeness(self, indicator: str) -> float:
        """
        Calculate the completeness of a specific indicator across all countries and years.

        Args:
            indicator: Indicator code to check

        Returns:
            float: Completeness ratio (0.0 to 1.0), where 1.0 means fully complete
        """
        if indicator not in self.indicators:
            raise ValueError(f"Indicator '{indicator}' not found in dataset")

        indicator_idx = self.indicators.get_loc(indicator)
        total_values = self.n_countries() * self.n_years()
        missing_values = np.isnan(self.data[:, indicator_idx, :]).sum()

        return (total_values - missing_values) / total_values

    def indicator_completeness_per_country(self, indicator: str) -> pd.Series:
        """
        Calculate the completeness of a specific indicator for each country.

        Args:
            indicator: Indicator code to check

        Returns:
            pd.Series: Series with country codes as index and completeness ratios as values, where 1.0 means fully complete
        """
        if indicator not in self.indicators:
            raise ValueError(f"Indicator '{indicator}' not found in dataset")

        indicator_idx = self.indicators.get_loc(indicator)
        total_years = self.n_years()
        missing_values = np.isnan(self.data[:, indicator_idx, :]).sum(axis=1)
        completeness = (total_years - missing_values) / total_years

        return pd.Series(completeness, index=self.countries)

    def extract_indicator(self, indicator: str) -> np.ndarray:
        """
        Extract the time series data for a specific indicator across all countries.

        Args:
            indicator: Indicator code to extract

        Returns:
            np.ndarray: 2D array with shape (n_countries, n_years)
        """
        if indicator not in self.indicators:
            raise ValueError(f"Indicator '{indicator}' not found in dataset")

        indicator_idx = self.indicators.get_loc(indicator)
        return self.data[:, indicator_idx, :]

    def extract_country(self, country: str) -> np.ndarray:
        """
        Extract the time series data for a specific country across all indicators.

        Args:
            country: Country code to extract

        Returns:
            np.ndarray: 2D array with shape (n_indicators, n_years)
        """
        if country not in self.countries:
            raise ValueError(f"Country '{country}' not found in dataset")

        country_idx = self.countries.get_loc(country)
        return self.data[country_idx, :, :]

    def to_pickle(self, filepath: str) -> None:
        """
        Serialize the dataset to a pickle file.

        Args:
            filepath: Path where the dataset will be saved
        """
        with open(filepath, 'wb') as f:
            pickle.dump({
                'data': self.data,
                'countries': self.countries,
                'indicators': self.indicators,
                'years': self.years,
                'is_modified': self.is_modified if hasattr(self, 'is_modified') else None
            }, f)

    @classmethod
    def from_pickle(cls, filepath: str) -> 'Dataset':
        """
        Load a dataset from a pickle file.

        Args:
            filepath: Path to the pickle file

        Returns:
            Dataset: Loaded dataset
        """
        with open(filepath, 'rb') as f:
            data_dict = pickle.load(f)
            return cls(
                data=data_dict['data'],
                countries=data_dict['countries'],
                indicators=data_dict['indicators'],
                years=data_dict['years'],
                modified=data_dict.get('is_modified')
            )

    def to_hdf5(self, filepath: str) -> None:
        """
        Serialize the dataset to HDF5 format.

        This is an efficient format for large numerical datasets.
        Requires h5py package to be installed.

        Args:
            filepath: Path where the dataset will be saved
        """
        if not HDF5_AVAILABLE:
            raise ImportError(
                "h5py package is required for HDF5 serialization")
        dirpath = os.path.dirname(filepath)
        if dirpath and not os.path.exists(dirpath):
            os.makedirs(dirpath, exist_ok=True)
        with h5py.File(filepath, 'w') as f:
            # Store data array
            f.create_dataset('data', data=self.data)

            # Store metadata as attributes or datasets
            f.create_dataset('countries', data=np.array(
                self.countries.tolist(), dtype='S'))
            f.create_dataset('indicators', data=np.array(
                self.indicators.tolist(), dtype='S'))
            f.create_dataset('years', data=self.years)

            if hasattr(self, 'is_modified'):
                f.create_dataset('is_modified', data=self.is_modified)

    @classmethod
    def from_hdf5(cls, filepath: str) -> 'Dataset':
        """
        Load a dataset from an HDF5 file.

        Args:
            filepath: Path to the HDF5 file

        Returns:
            Dataset: Loaded dataset
        """
        if not HDF5_AVAILABLE:
            raise ImportError(
                "h5py package is required for HDF5 serialization")

        with h5py.File(filepath, 'r') as f:
            data = f['data'][()]

            # Convert bytes to strings for countries and indicators
            countries = pd.Index([s.decode('utf-8')
                                 for s in f['countries'][()]])
            indicators = pd.Index([s.decode('utf-8')
                                  for s in f['indicators'][()]])
            years = pd.Index(f['years'][()])

            modified = None
            if 'is_modified' in f:
                modified = f['is_modified'][()]

            return cls(
                data=data,
                countries=countries,
                indicators=indicators,
                years=years,
                modified=modified
            )

    def save(self, filepath: str, format: str = 'pickle') -> None:
        """
        Save the dataset in the specified format.

        Args:
            filepath: Path where the dataset will be saved
            format: Format to use ('pickle', 'json', or 'hdf5')
        """
        format = format.lower()

        # Add appropriate extension if not provided
        extensions = {'pickle': '.pkl', 'json': '.json', 'hdf5': '.h5'}
        if not any(filepath.endswith(ext) for ext in extensions.values()):
            filepath += extensions[format]

        if format == 'pickle':
            self.to_pickle(filepath)
        elif format == 'json':
            self.to_json(filepath)
        elif format == 'hdf5':
            self.to_hdf5(filepath)
        else:
            raise ValueError(
                f"Unsupported format: {format}. Must be one of: pickle, json, hdf5")

    @classmethod
    def load(cls, filepath: str, format: str = None) -> 'Dataset':
        """
        Load a dataset from the specified file.

        If format is not specified, it will be inferred from the file extension.

        Args:
            filepath: Path to the file to load
            format: Optional format override ('pickle', 'json', or 'hdf5')

        Returns:
            Dataset: Loaded dataset
        """
        if format is None:
            # Infer format from file extension
            if filepath.endswith('.pkl'):
                format = 'pickle'
            elif filepath.endswith('.json'):
                format = 'json'
            elif filepath.endswith('.h5') or filepath.endswith('.hdf5'):
                format = 'hdf5'
            else:
                raise ValueError(
                    f"Could not infer format from file extension: {filepath}")

        format = format.lower()
        if format == 'pickle':
            return cls.from_pickle(filepath)
        elif format == 'json':
            return cls.from_json(filepath)
        elif format == 'hdf5':
            return cls.from_hdf5(filepath)
        else:
            raise ValueError(
                f"Unsupported format: {format}. Must be one of: pickle, json, hdf5")

    def copy(self) -> 'Dataset':
        """
        Create a deep copy of the dataset.

        Returns:
            Dataset: A new instance of the dataset with copied data and metadata
        """
        return Dataset(
            data=self.data.copy(),
            countries=self.countries.copy(),
            indicators=self.indicators.copy(),
            years=self.years.copy(),
            modified=self.is_modified.copy() if hasattr(self, 'is_modified') else None
        )


def exclude_indicators_from_dataset(dataset: Dataset, indicators: list[str]) -> Dataset:
    """
    Exclude specific indicators from the dataset.

    Args:
        dataset: Dataset object to modify
        indicators: List of indicator codes to exclude

    Returns:
        Dataset: New dataset with specified indicators excluded
    """
    if not isinstance(dataset, Dataset):
        raise ValueError("dataset must be an instance of Dataset")

    if not isinstance(indicators, list):
        raise ValueError("indicators must be a list of strings")

    # Get indices of indicators to exclude
    exclude_indices = [dataset.indicator_index(
        ind) for ind in indicators if ind in dataset.indicators]

    # Create new data array excluding specified indicators
    new_data = np.delete(dataset.data, exclude_indices, axis=1)

    # Create new indicators index excluding specified indicators
    new_indicators = dataset.indicators.drop(indicators)

    return Dataset(new_data, dataset.countries, new_indicators, dataset.years)


def split_dataset(dataset: Dataset, countries_to_take: list[str]) -> tuple[Dataset, Dataset]:
    """
    Split the dataset into two datasets based on specified countries.

    Args:
        dataset: Dataset object to split
        countries_to_take: List of country codes to include in the first dataset

    Returns:
        tuple: Two Dataset objects (first_dataset, second_dataset)
    """
    if not isinstance(dataset, Dataset):
        raise ValueError("dataset must be an instance of Dataset")

    if not isinstance(countries_to_take, list):
        raise ValueError("countries_to_take must be a list of strings")

    # Get indices of countries to take
    take_indices = [dataset.countries.get_loc(
        country) for country in countries_to_take if country in dataset.countries]

    # Create new data arrays for both datasets
    first_data = dataset.data[take_indices]
    second_data = np.delete(dataset.data, take_indices, axis=0)

    # Create new countries indices for both datasets
    first_countries = dataset.countries[take_indices]
    second_countries = dataset.countries.drop(countries_to_take)

    return (Dataset(first_data, first_countries, dataset.indicators, dataset.years),
            Dataset(second_data, second_countries, dataset.indicators, dataset.years))
