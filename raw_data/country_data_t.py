from typing import List, Optional, Union
import itertools
from raw_data.data_utils import DataUtils
from raw_data.wb_api import query_api
from pathlib import Path
import numpy as np
import pandas as pd
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


class country_data_t:
    """
    Class to handle data for a single country.

    This class provides methods for loading, saving, updating, and manipulating
    economic data for a specific country. It maintains three versions of the data:
    - raw data: as downloaded from the API
    - modified data: with gaps filled or corrections made
    - data: a superimposition of modified data over raw data

    It also keeps track of which data has been queried using a mask dataframe.
    """

    def __init__(
        self,
        country: str,
        parent_path: Path = Path("data"),
        raw_directory: Path = Path("raw_data"),
        mod_directory: Path = Path("mod_data"),
    ) -> None:
        """
        Initialize a country_data_t object.

        Parameters
        ----------

        country : str
            Country code (e.g., 'USA', 'GBR')
        parent_path : Path
            Parent directory where data will be stored
        raw_directory : Path
            Relative path to store raw data within parent_path
        mod_directory : Path
            Relative path to store modified data within parent_path
        """
        print(
            f'country_data_t initialized for {country}, {parent_path.absolute()}')
        self.country = country
        "Country code"
        self.raw_directory = parent_path / raw_directory
        "Directory for raw data"
        self.mod_directory = parent_path / mod_directory
        "Directory for modified data"

        self.data_raw: Optional[pd.DataFrame] = None
        "Data as downloaded from the API, can have missing values"
        self.data_mod: Optional[pd.DataFrame] = None
        "Data with missing values filled in"
        self.mask: Optional[pd.DataFrame] = None
        """Mask for queried data, useful for when the data is not available for a certain year. 
        
        'Has this data been queried from the API?'"""
        self.data: Optional[pd.DataFrame] = None
        "Superimposition of modified data on top of raw data"

    def load_data(self) -> None:
        """
        Load raw data, modified data, and mask from disk.

        This method loads the raw data, modified data, and mask from HDF files.
        If the mask does not exist but raw data does, it creates a mask based on
        non-NA values in the raw data and saves it.

        After loading, it calls superimpose_data() to create the combined dataset.
        """
        self.data_raw = DataUtils.load_hdf(
            self.raw_directory / f"{self.country}.h5", "data"
        )
        self.data_mod = DataUtils.load_hdf(
            self.mod_directory / f"{self.country}.h5", "data"
        )
        self.mask = DataUtils.load_hdf(
            self.raw_directory / f"{self.country}_mask.h5", "mask"
        )

        if self.mask is None and self.data_raw is not None:
            self.mask = self.data_raw.notna()
            DataUtils.save_hdf(
                self.mask, self.raw_directory /
                f"{self.country}_mask.h5", "mask"
            )

        self.superimpose_data()

    def save_data(self) -> None:
        """
        Save raw data, modified data, and mask to disk.

        This method saves:
        - Raw data to {raw_directory}/{country}.h5
        - Mask to {raw_directory}/{country}_mask.h5
        - Modified data to {mod_directory}/{country}.h5

        Each dataset is only saved if it exists (not None).
        """
        if self.data_raw is not None:
            DataUtils.save_hdf(
                self.data_raw, self.raw_directory /
                f"{self.country}.h5", "data"
            )
            DataUtils.save_hdf(
                self.mask, self.raw_directory /
                f"{self.country}_mask.h5", "mask"
            )
        if self.data_mod is not None:
            DataUtils.save_hdf(
                self.data_mod, self.mod_directory /
                f"{self.country}.h5", "data"
            )

    def _query_and_update_indicator(self, indicator: str, years: List[int]) -> None:
        """Queries the API for a given indicator and years, and updates the data and mask."""
        indicator_data = query_api([self.country], [indicator], years)
        for year in years:
            self.data_raw.loc[indicator,
                              year] = indicator_data.loc[indicator, year]
            self.mask.loc[indicator, year] = True

    def _query_and_update_new_indicator(self, indicator: str, years: List[int]) -> None:
        """Queries the API for a new indicator and years, and updates the data and mask."""
        indicator_data = query_api([self.country], [indicator], years)
        self.data_raw = pd.concat([self.data_raw, indicator_data])
        self.mask = pd.concat(
            [
                self.mask,
                pd.DataFrame(index=[indicator],
                             columns=years, dtype=bool, data=True),
            ]
        )

    def _create_initial_data(self, indicators: List[str], years: List[int]) -> None:
        """Creates initial data and mask if they don't exist."""
        new_data = pd.DataFrame()
        for batch in itertools.batched(indicators, 50):
            data = query_api([self.country], batch, years)
            new_data = pd.concat([new_data, data])
        self.data_raw = new_data
        self.mask = pd.DataFrame(index=indicators, columns=years, dtype=bool)
        self.mask[:] = True

    def _update_existing_data(self, indicators: List[str], years: List[int]) -> None:
        """Updates existing data and mask for given indicators and years."""
        for indicator in indicators:
            if indicator in self.mask.index.values:
                if missing_years := [
                    year
                    for year in years
                    if year not in self.data_raw.columns
                    or not self.mask.loc[indicator, year]
                ]:
                    self._query_and_update_indicator(indicator, missing_years)
            else:
                self._query_and_update_new_indicator(indicator, years)

    def update_data(self, indicators: List[str], years: List[int]) -> None:
        """
        Update data by querying the API for specified indicators and years.

        This method queries the World Bank API for the given indicators and years
        for this country. If the data already exists and has been queried (according
        to the mask), it won't query it again. For new indicators or years not
        previously queried, it will retrieve the data and update the dataframes.

        Parameters
        ----------

        indicators : List[str]
            List of indicator codes to query
        years : List[int]
            List of years to query data for
        """
        years = [int(year) for year in years]  # making sure years are integers
        if self.data_raw is None:  # no data exists yet
            self._create_initial_data(indicators, years)
        else:
            self._update_existing_data(indicators, years)
        self.data_raw = DataUtils.sort_dataframe(self.data_raw)
        self.mask = DataUtils.sort_dataframe(self.mask)

        pd.set_option("future.no_silent_downcasting", True)
        self.mask = self.mask.replace([pd.NA, np.nan, None], False)

        self.save_data()
        self.superimpose_data()

    def download_missing(self) -> None:
        """
        Download data that has been marked as not downloaded in the mask.

        This method checks the mask for entries marked as False (not downloaded),
        groups them by indicator, and calls update_data() for each indicator
        with its missing years.
        """
        missing_data = self.downloaded_mask == False
        missing_data = missing_data.stack()
        missing_data = missing_data[missing_data]
        missing_data = missing_data.reset_index()
        missing_data = missing_data.drop(columns=0)
        missing_data.columns = ["indicator", "year"]
        missing_data = (
            missing_data.groupby("indicator")["year"].apply(list).reset_index()
        )
        for i in range(len(missing_data)):
            self.update_data(
                [missing_data.loc[i, "indicator"]], missing_data.loc[i, "year"]
            )

    def ensure_continuous(self) -> None:
        """
        Ensure data is available for all years between the min and max years.

        This method identifies any missing years within the range of available years
        and queries the API to fill in those gaps. It then downloads any remaining
        missing data."""
        if self.data_raw is None:
            return
        min_year = self.data_raw.columns.min()
        max_year = self.data_raw.columns.max()
        if max_year - min_year < 2:
            return
        missing_years = []
        missing_years.extend(
            year
            for year in range(min_year, max_year)
            if year not in self.data_raw.columns.values
        )
        if missing_years:
            self.update_data(self.data_raw.index.values, missing_years)
            self.download_missing()

    def superimpose_data(self) -> None:
        """
        Create a combined dataset by superimposing modified data over raw data.

        This method merges raw_data and mod_data to create a complete dataset,
        using values from mod_data where available, and falling back to raw_data
        otherwise. The result is stored in the data attribute.
        """
        if self.data_raw is None:
            self.data = None
        elif self.data_mod is None:
            self.data = self.data_raw.copy()
        else:
            # fill in missing values in raw data with modified data, overwriting raw data with modified data if both are available
            self.data = self.data_mod.combine_first(self.data_raw)

    def extract_for_dataset(self, indicators: List[str], years: List[int]) -> pd.DataFrame:
        """
        Extract a subset of data for specified indicators and years.

        This method creates a dataframe containing only the specified indicators and years
        from the combined dataset. Missing values remain as NaN.

        Parameters
        ----------

        indicators : List[str]
            List of indicator codes to extract
        years : List[int]
            List of years to extract data for

        Returns
        -------

        pd.DataFrame
            DataFrame with the extracted data, having indicators as index and years as columns
        """
        # prepare an empty dataframe with indicators as rows and years as columns
        dataset_data = DataUtils.generate_empty_dataset(indicators, years)
        if self.data is None:
            return dataset_data
        for indicator in indicators:
            if indicator not in self.data.index.values:
                continue
            for year in years:
                if year not in self.data.columns.values:
                    continue
                dataset_data.loc[indicator,
                                 year] = self.data.loc[indicator, year]

        return dataset_data

    def input_modified_data(self, indicator, year, value) -> None:
        """
        Input modified data for a specific indicator and year.

        This method inputs a new value for a specific indicator and year in the modified data.

        Parameters
        ----------

        indicator : str
            Indicator code to input data for
        year : int
            Year to input data for
        value : float
            Value to input for the indicator
        """
        if self.data_mod is None:
            self.data_mod = pd.DataFrame(
                index=self.data_raw.index, columns=self.data_raw.columns, dtype=float, data=np.nan)
        self.data_mod.loc[indicator, year] = value
        self.data[indicator, year] = value
