from raw_data.dataset import Dataset, exclude_indicators_from_dataset
import numpy as np
from preprocessing.gain import GAINImputer
from sklearn.preprocessing import RobustScaler, StandardScaler
import pandas as pd
from itertools import combinations
from statsmodels.tsa.stattools import adfuller
import warnings


class Scaler:
    def __init__(self, method: str = 'zscore', **kwargs):
        """
        Initialize the scaler with a specific method.

        Args:
            method: The scaling method to use ('robust', 'zscore')
            kwargs: Additional parameters for the scaling method
        """
        method.lower()
        self._scaler_classes = {
            'robust': RobustScaler,
            'zscore': StandardScaler
        }
        if method not in self._scaler_classes:
            raise ValueError(
                f"Unknown scaling method: {method}. Available methods: {list(self._scaler_classes.keys())}")
        self.scaler_class = self._scaler_classes[method]

        self.kwargs = kwargs
        self.is_fitted = False
        self.scalers = {}

    def fit(self, dataset: Dataset):
        for indicator in dataset.indicators:
            if indicator not in self.scalers:
                self.scalers[indicator] = self.scaler_class(**self.kwargs)
            self.scalers[indicator].fit(
                dataset.extract_indicator(indicator).reshape(-1, 1))

        self.is_fitted = True

    def transform(self, dataset: Dataset, inplace: bool = False) -> Dataset:
        """
        Transform the dataset using fitted scalers.

        Args:
            dataset: Dataset object with data to transform
            inplace: If True, modify the dataset in-place, otherwise return a new instance

        Returns:
            Dataset: Transformed dataset (either the same object or a new instance)
        """
        if not self.scalers:
            raise ValueError(
                "Scalers not fitted. Call fit() before transform().")

        transformed_data = dataset.data.copy()

        for indicator, scaler in self.scalers.items():
            if indicator in dataset.indicators:
                idx = dataset.indicator_index(indicator)
                # Extract indicator data
                indicator_data = dataset.extract_indicator(indicator)
                # Reshape for scikit-learn's 2D format
                original_shape = indicator_data.shape
                reshaped_data = indicator_data.reshape(-1, 1)
                # Transform only non-NaN values
                mask = ~np.isnan(reshaped_data)
                reshaped_data[mask] = scaler.transform(
                    reshaped_data[mask].reshape(-1, 1)).flatten()
                # Update the data
                transformed_data[:, idx, :] = reshaped_data.reshape(
                    original_shape)

        if inplace:
            dataset.data = transformed_data
            return dataset
        else:
            return Dataset(
                data=transformed_data,
                countries=dataset.countries,
                indicators=dataset.indicators,
                years=dataset.years,
            )

    def fit_transform(self, dataset: Dataset) -> Dataset:
        """
        Fit the scalers and transform the dataset.

        Args:
            dataset: Dataset object with data to fit and transform

        Returns:
            Dataset: Transformed dataset
        """
        self.fit(dataset)
        return self.transform(dataset)

    def inverse_transform(self, dataset: Dataset, inplace: bool = False) -> Dataset:
        """
        Inverse transform the dataset using fitted scalers.

        Args:
            dataset: Dataset object with data to inverse transform
            inplace: If True, modify the dataset in-place, otherwise return a new instance

        Returns:
            Dataset: Inverse transformed dataset (either the same object or a new instance)
        """
        if not self.scalers:
            raise ValueError(
                "Scalers not fitted. Call fit() before inverse_transform().")

        transformed_data = dataset.data.copy()
        is_modified = np.zeros_like(dataset.data, dtype=bool)
        if hasattr(dataset, "is_modified"):
            is_modified = dataset.is_modified.copy()

        for indicator, scaler in self.scalers.items():
            if indicator in dataset.indicators:
                idx = dataset.indicator_index(indicator)
                # Extract indicator data
                indicator_data = dataset.extract_indicator(indicator)
                # Reshape for scikit-learn's 2D format
                original_shape = indicator_data.shape
                reshaped_data = indicator_data.reshape(-1, 1)
                # Inverse transform only non-NaN values
                mask = ~np.isnan(reshaped_data)
                reshaped_data[mask] = scaler.inverse_transform(
                    reshaped_data[mask].reshape(-1, 1)).flatten()
                # Update the data
                transformed_data[:, idx, :] = reshaped_data.reshape(
                    original_shape)
                # Mark as not modified
                is_modified[:, idx, :] = False

        if inplace:
            dataset.data = transformed_data
            dataset.is_modified = is_modified
            return dataset
        else:
            return Dataset(
                data=transformed_data,
                countries=dataset.countries,
                indicators=dataset.indicators,
                years=dataset.years,
                modified=is_modified
            )

    def to_pickle(self, filepath: str) -> None:
        """
        Save the scaler to a pickle file.

        Args:
            filepath: Path to save the scaler
        """
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    def from_pickle(self, filepath: str) -> None:
        """
        Load the scaler from a pickle file.

        Args:
            filepath: Path to load the scaler from
        """
        import pickle
        with open(filepath, 'rb') as f:
            loaded_scaler = pickle.load(f)
            self.scalers = loaded_scaler.scalers
            self.is_fitted = loaded_scaler.is_fitted


class Preprocessor:
    @staticmethod
    def remove_constant(dataset: Dataset, variance_threshold: float = 0.1, country_threshold: float = 0.25) -> Dataset:
        """
        Remove indicators with constant values (variance below the threshold), in at least a certain percentage of countries.

        Args:
            dataset: The dataset to process
            variance_threshold: Minimum variance threshold to count an indicator as active
            country_threshold: Minimum percentage of countries that must have non-constant values for the indicator to be kept
        Returns:
            Dataset with constant indicators removed
        """
        indicators_to_discard = []
        total_countries = dataset.n_countries()

        for indicator in dataset.indicators:
            variances = dataset.indicator_variance_per_country(indicator)
            # Count how many countries have variance above the threshold
            active_countries = sum(
                1 for variance in variances if variance > variance_threshold)
            # If the number of active countries is below the threshold, discard the indicator
            if (active_countries / total_countries) < country_threshold:
                indicators_to_discard.append(indicator)

        return exclude_indicators_from_dataset(dataset, indicators_to_discard)

    @staticmethod
    def remove_missing_total(dataset: Dataset, threshold: float = 0.5) -> Dataset:
        """
        Remove indicators with missing values above the threshold(percentage of missing values of all the values)

        Args:
            dataset: The dataset to process
            threshold: Maximum percentage of missing values allowed for an indicator

        Returns:
            Dataset with high-missing indicators removed
        """
        to_remove = []
        for indicator in dataset.indicators:
            indicator_completeness = dataset.indicator_completeness(indicator)
            missing_ratio = 1 - indicator_completeness
            if missing_ratio > threshold:
                to_remove.append(indicator)
        return exclude_indicators_from_dataset(dataset, to_remove)

    @staticmethod
    def remove_missing_per_country(dataset: Dataset, threshold: float = 0.4) -> Dataset:
        """
        Remove indicators that have a missingness ratio above the threshold for any country
        Args:
            dataset: The dataset to process
            threshold: Maximum percentage of missing values allowed for an indicator in any country
        Returns:
            Dataset with indicators that have high missingness in any country removed
        """
        to_remove = []
        for indicator in dataset.indicators:
            indicator_completeness = dataset.indicator_completeness_per_country(
                indicator)
            missing_ratios = [
                1 - completeness for completeness in indicator_completeness]
            if any(ratio > threshold for ratio in missing_ratios):
                to_remove.append(indicator)
        return exclude_indicators_from_dataset(dataset, to_remove)

    @staticmethod
    def remove_variance(dataset: Dataset, threshold: float = 0.01) -> Dataset:
        """
        Remove indicators with variance below the threshold

        Args:
            dataset: The dataset to process
            threshold: Minimum variance threshold to keep an indicator

        Returns:
            Dataset with low-variance indicators removed
        """
        indicators_to_discard = []

        for indicator in dataset.indicators:
            variance = dataset.indicator_variance(indicator)
            if variance == 0 or variance < threshold:
                indicators_to_discard.append(indicator)

        return exclude_indicators_from_dataset(dataset, indicators_to_discard)

    @staticmethod
    def stationarity(
        dataset: Dataset,
        adf_alpha: float = 0.05,
        inplace: bool = True,
        min_obs: int = 3,
    ) -> Dataset:
        data = dataset.data if inplace else dataset.data.copy().astype("float64")

        n_countries, n_indicators, n_years = data.shape

        for indicator_idx in range(n_indicators):
            for country_idx in range(n_countries):
                time_series = data[country_idx, indicator_idx, :].copy()

                # Option 1: non-positive values - plain delta
                if (time_series <= 0).any():
                    data[country_idx, indicator_idx, :] = np.diff(
                        time_series, prepend=np.nan)
                    continue

                time_series = time_series.astype("float64")
                time_series = np.log(time_series)

                # guard against nan/inf values from log
                bad_mask = ~np.isfinite(time_series)
                if bad_mask.any():
                    # if there are bad values, replace them with NaN
                    time_series[bad_mask] = np.nan

                nan_mask = np.isnan(time_series)
                time_series_non_nan = time_series[~nan_mask]

                # Check if there are enough non-NaN observations
                if len(time_series_non_nan) < min_obs or np.var(time_series_non_nan) < 1e-7:
                    # Not enough data to perform stationarity test, keep log values
                    data[country_idx, indicator_idx, :] = time_series
                    continue
                with warnings.catch_warnings():
                    warnings.filterwarnings("error", category=RuntimeWarning)
                    try:
                        p_value = adfuller(time_series_non_nan)[1]
                        if p_value > adf_alpha:
                            # If p-value is greater than alpha, series is non-stationary
                            # Apply first difference to make it stationary
                            time_series = np.diff(
                                time_series, prepend=np.nan)
                    except (RuntimeWarning, ValueError):
                        # adfuller raised a warning or error, treat as non-stationary
                        time_series = np.diff(time_series, prepend=np.nan)

                data[country_idx, indicator_idx, :] = time_series

        if inplace:
            dataset.data = data
            return dataset
        else:
            return Dataset(
                data=data,
                countries=dataset.countries,
                indicators=dataset.indicators,
                years=dataset.years,
            )

    @staticmethod
    def z_score(dataset: Dataset, inplace: bool = True) -> Dataset:
        """
        Apply z-score normalization to the dataset.

        Args:
            dataset: Dataset object with data to normalize
            inplace: If True, modify the dataset in -place, otherwise return a new instance
        Returns:
            Dataset: Normalized dataset(either the same object or a new instance)
        """
        from scipy.stats import zscore
        data = dataset.data.copy() if not inplace else dataset.data
        n_countries, n_indicators, n_years = data.shape

        for indicator_idx in range(n_indicators):
            for country_idx in range(n_countries):
                time_series = data[country_idx, indicator_idx, :]
                # Apply z-score normalization
                normalized_series = zscore(time_series, nan_policy='omit')
                # Replace NaNs with 0
                normalized_series = np.nan_to_num(normalized_series)
                data[country_idx, indicator_idx, :] = normalized_series

        if inplace:
            dataset.data = data
            return dataset
        else:
            return Dataset(
                data=data,
                countries=dataset.countries,
                indicators=dataset.indicators,
                years=dataset.years,
            )

    @staticmethod
    def _produce_correlation_matrix(dataset: Dataset, corr_method='pearson', min_periods=3) -> np.ndarray:
        """
        Produce a correlation matrix for the dataset.

        Args:
            dataset: The dataset to process
            corr_method: Correlation method to use (default is 'pearson')
            min_periods: Minimum number of observations required for correlation

        Returns:
            Correlation matrix as a 2D numpy array
        """
        def fisher_mean(corr_stack):
            """corr_stack: (n_countries, p, p) array of Pearson ρ"""
            z = np.arctanh(np.clip(corr_stack, -0.9999, 0.9999))
            z_mean = np.nanmean(z, axis=0)
            return np.tanh(z_mean)

        # iterate over the countries in the dataset, create correlation matrices for each one
        correlation_matrices = []
        for i in range(dataset.n_countries()):
            # data for the country
            data = dataset.data[i]
            # Transpose the data so indicators are columns (features)
            df = pd.DataFrame(data.T, columns=dataset.indicators)

            # Calculate the correlation matrix with NaN handling
            # min_periods ensures we have enough data points for meaningful correlation
            corr = df.corr(method=corr_method, min_periods=min_periods)

            correlation_matrices.append(corr)

        # average the correlation matrices across all countries
        avg_corr = fisher_mean(
            np.stack(correlation_matrices))   # p × p ndarray
        np.fill_diagonal(avg_corr, 1.0)
        return avg_corr

    @staticmethod
    def correlation_prune(dataset: Dataset, original_dataset: Dataset, threshold: float = 0.8, min_periods=3, corr_method='pearson') -> Dataset:
        """
        Remove indicators that are highly correlated with each other.

        Args:
            dataset: The dataset to process
            threshold: Correlation threshold above which indicators will be considered redundant

        Returns:
            Dataset with highly correlated indicators removed
        """
        from scipy.cluster.hierarchy import linkage, fcluster

        avg_corr = Preprocessor._produce_correlation_matrix(
            dataset, corr_method=corr_method, min_periods=min_periods)

        # hierarchical clustering on distance = 1-|ρ|
        dist = 1 - abs(avg_corr)
        Z = linkage(dist[np.triu_indices_from(
            dist, k=1)], method="complete")
        labels = fcluster(Z, t=1-threshold, criterion="distance")

        to_keep = set()
        for label in np.unique(labels):
            members = np.where(labels == label)[0]
            if len(members) == 0:
                continue
            if len(members) == 1:
                # only one member, keep it
                to_keep.add(members[0])
                continue

            # For clusters with multiple members, choose the one with highest completeness
            best_member = None
            best_completeness = -1
            for member_idx in members:
                indicator = dataset.indicators[member_idx]
                completeness = original_dataset.indicator_completeness(
                    indicator)
                if completeness > best_completeness:
                    best_completeness = completeness
                    best_member = member_idx
            if best_member is not None:
                to_keep.add(best_member)

        # Convert to indicators to remove
        to_remove = []
        for i, indicator in enumerate(dataset.indicators):
            if i not in to_keep:
                to_remove.append(indicator)

        return exclude_indicators_from_dataset(dataset, to_remove)

    @staticmethod
    def correlation_summary(correlation_matrix: np.ndarray, threshold: float) -> list[tuple[str, str]]:
        """
        Summarize pairs of indicators with correlation above the threshold.
        """
        correlation_matrix = np.abs(correlation_matrix)
        # remove self-correlations
        np.fill_diagonal(correlation_matrix, 0)
        maximum = np.max(correlation_matrix)
        mean = np.mean(correlation_matrix)
        return (maximum, mean)

    @staticmethod
    def repeated_correlation_prune(dataset: Dataset, original_dataset: Dataset, threshold: float = 0.8, min_periods=3, corr_method='pearson') -> Dataset:
        """
        Single pass of greedy correlation pruning on hierarchical clusters can leave correlation pairs above threshold in between clusters, need to repeat pruning until no more pairs above threshold are found.
        """

        corr_matrix = Preprocessor._produce_correlation_matrix(
            dataset, corr_method=corr_method, min_periods=min_periods)
        maximum, mean = Preprocessor.correlation_summary(
            corr_matrix, threshold)
        n_features = dataset.n_indicators()
        print(
            f"Initial correlation matrix: max={maximum:.3f}, mean={mean:.3f}, features={n_features} (threshold={threshold})")

        i = 0
        while True:
            prev_count = dataset.n_indicators()
            dataset = Preprocessor.correlation_prune(
                dataset, original_dataset, threshold, min_periods, corr_method)
            new_count = dataset.n_indicators()
            if new_count == prev_count:
                break
            corr_matrix = Preprocessor._produce_correlation_matrix(
                dataset, corr_method=corr_method, min_periods=min_periods)
            maximum, mean = Preprocessor.correlation_summary(
                corr_matrix, threshold)
            n_features = dataset.n_indicators()
            print(
                f"Iteration {i}: max={maximum:.3f}, mean={mean:.3f}, features={n_features} (threshold={threshold})")

            i += 1

        return dataset


class Imputer:
    def __init__(self, method: str = 'mice', **kwargs):
        """
        Initialize the imputer with a specific method.

        Args:
            method: The imputation method to use ('mean', 'linear', 'mice')
            kwargs: Additional parameters for the imputation method
        """
        self.method = method.lower()
        self.kwargs = kwargs
        self.is_fitted = False
        self.imputer = None

    def fit(self, dataset: Dataset):
        self.imputer = None
        match self.method:
            case 'mice':
                from sklearn.experimental import enable_iterative_imputer
                from sklearn.impute import IterativeImputer
                self.imputer = IterativeImputer(**self.kwargs)

                original_shape = dataset.data.shape
                reshaped = dataset.data.reshape(-1, original_shape[-1])
                self.imputer.fit(reshaped)
            case 'linear':
                pass
            case 'mean':
                from sklearn.impute import SimpleImputer
                self.imputer = SimpleImputer(strategy='mean', **self.kwargs)
                original_shape = dataset.data.shape
                reshaped = dataset.data.reshape(-1, original_shape[-1])
                self.imputer.fit(reshaped)
            case _:
                raise ValueError(
                    f"Unknown imputation method: {self.method}.")

        print(f"Imputer fitted with method: {self.method}")
        self.is_fitted = True

    def transform(self, dataset: Dataset) -> Dataset:
        if not self.is_fitted:
            raise ValueError(
                "Imputer not fitted. Call fit() before transform().")
        
        if self.imputer is None and self.method != 'linear':
            raise ValueError(
                f"Imputer is None for method {self.method}. This should not happen.")

        match self.method:
            case 'mice':
                import scipy.sparse
                from typing import Any
                original_shape = dataset.data.shape
                reshaped = dataset.data.reshape(-1, original_shape[-1])
                reshaped = self.imputer.transform(reshaped)  # type: ignore
                # Convert sparse matrix to dense if needed
                if scipy.sparse.issparse(reshaped):
                    reshaped = reshaped.toarray()  # type: ignore
                dataset.data = np.asarray(reshaped).reshape(original_shape)
            case 'linear':
                original_shape = dataset.data.shape
                n_countries, n_indicators, n_years = original_shape

                result = np.empty_like(dataset.data, dtype=np.float64)

                for country_idx in range(n_countries):
                    for indicator_idx in range(n_indicators):
                        # Extract the time series for the current country and indicator
                        time_series = dataset.data[country_idx,
                                                   indicator_idx, :]

                        # Perform linear interpolation
                        interpolated_series = pd.Series(time_series).interpolate(
                            method='linear', limit_direction='both')

                        # Store the result back in the array
                        result[country_idx, indicator_idx,
                               :] = interpolated_series.values  # type: ignore
                dataset.data = result
            case 'mean':
                import scipy.sparse
                original_shape = dataset.data.shape
                reshaped = dataset.data.reshape(-1, original_shape[-1])
                reshaped = self.imputer.transform(reshaped)  # type: ignore
                # Convert sparse matrix to dense if needed
                if scipy.sparse.issparse(reshaped):
                    reshaped = reshaped.toarray()  # type: ignore
                dataset.data = np.asarray(reshaped).reshape(original_shape)
            case _:
                raise ValueError(
                    f"Unknown imputation method: {self.method}.")
        return dataset

    def fit_transform(self, dataset: Dataset) -> Dataset:
        self.fit(dataset)
        return self.transform(dataset)


def apply_preprocessing_step(dataset: Dataset, func, func_kwargs: dict, description_template: str, prev_count: int) -> tuple[Dataset, int]:
    """
    Applies a preprocessing function to the dataset and logs the results.

    Args:
        dataset: The dataset to preprocess
        func: The preprocessing function to apply
        func_kwargs: Keyword arguments to pass to the function
        description_template: F-string template for the description (will be formatted with func_kwargs)
        prev_count: Previous indicator count for calculating removed indicators

    Returns:
        Tuple of (processed_dataset, new_indicator_count)
    """
    processed_dataset: Dataset = func(dataset, **func_kwargs)
    indicator_count = processed_dataset.n_indicators()
    removed_count = prev_count - indicator_count

    # Format the description template with the function kwargs
    description = description_template.format(**func_kwargs)
    print(f"{description} Removed {removed_count} indicators, {indicator_count} remain.")
    return processed_dataset, indicator_count


def apply_correlation_pruning(dataset: Dataset, imputer='mice', threshold=0.8) -> Dataset:
    dataset_copy = dataset.copy()

    # Impute missing values
    imp = Imputer(method=imputer)
    dataset_copy = imp.fit_transform(dataset_copy)

    # stationarise
    dataset_copy = Preprocessor.stationarity(dataset_copy)

    # normalise
    dataset_copy = Preprocessor.z_score(dataset_copy)

    # Check for correlation
    dataset_copy = Preprocessor.repeated_correlation_prune(
        dataset_copy, dataset, threshold=threshold)

    return dataset_copy
