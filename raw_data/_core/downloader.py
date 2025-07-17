from raw_data._core.base import CoreBase
from raw_data.documentation_structs import country_t
from raw_data.country_data_t import country_data_t
from raw_data.download_process import download_process


def error_callback(e):
    """Callback function for handling errors in worker processes.

    Args:
        e: The error raised by the worker process
    """
    print(f"error in worker: {e}")


class Downloader(CoreBase):
    def download_data_batch(
        self,
        country_codes: list[str],
        indicators: list[str],
        years: tuple[int, int],
    ):
        """
        Download data for multiple countries in batch mode.

        Args:
            country_codes: List of country codes to download data for
            indicators: List of indicator codes to download
            years: Tuple with (start_year, end_year) range to download
        """
        if self.multiprocessing:
            self.download_data_multiprocessing_batch(
                country_codes, indicators, years)
        else:
            for country_code in country_codes:
                print(f"Downloading data for {country_code}...")
                self.download_data(country_code, indicators, years)
                print(f"Data for {country_code} downloaded.")

    def download_data(
        self,
        country_code: str,
        indicators: list[str],
        years: tuple[int, int],
    ):
        """
        Download data for a single country.

        Args:
            country_code: Country code to download data for
            indicators: List of indicator codes to download
            years: Tuple with (start_year, end_year) range to download
        """
        country: country_t = self.countries_data.objects[country_code]
        country.update_data(indicators, list(range(years[0], years[1] + 1)))
        if hasattr(self, 'Queue'):
            self.Queue.put_nowait(f"Data for {country_code} downloaded.")

    def download_data_multiprocessing_batch(
        self,
        country_codes: list[str],
        indicators: list[str],
        years: tuple[int, int],
    ):
        """
        Download data for multiple countries using multiprocessing.

        Args:
            country_codes: List of country codes to download data for
            indicators: List of indicator codes to download
            years: Tuple with (start_year, end_year) range to download
        """
        print("Downloading data for countries:", country_codes)
        for country_code in country_codes:
            self.download_data_multiprocessing(country_code, indicators, years)

    def download_data_multiprocessing(
        self,
        country_code: str,
        indicators: list[str],
        years: tuple[int, int],
    ) -> None:
        """
        Download data for a single country using multiprocessing.

        Args:
            country_code: Country code to download data for
            indicators: List of indicator codes to download
            years: Tuple with (start_year, end_year) range to download
        """
        country = self.countries_data.objects[country_code].data_object
        data_object_existed = country is not None
        if not data_object_existed:
            country = country_data_t(country_code)

        self.Pool.apply_async(
            download_process,
            args=(country, indicators, years, self.Queue, country_code),
            error_callback=error_callback,
        )

    def download_everything(self) -> None:
        """
        Download all possible data for countries and indicators in configuration files.

        This method attempts to download complete data for all defined countries and indicators
        for the entire time range specified in MIN_YEAR and MAX_YEAR.
        """
        countries = [country.code for country in self.get_all_countries()]
        indicators = [
            indicator.code for indicator in self.get_all_indicators()]
        self.download_data_batch(
            countries,
            indicators,
            (self.MIN_YEAR, self.MAX_YEAR),
        )

    def ensure_homogeneous(self) -> None:
        """
        Ensure data homogeneity across all countries.

        Downloads any missing indicators for countries that might be missing data
        that exists for other countries to ensure consistent indicator coverage.
        """
        self.update_available_information()
        countries = [
            country.code for country in self.get_available_countries()]
        indicators = [
            indicator.code for indicator in self.get_available_indicators()]
        years = (self.min_year, self.max_year)
        self.download_data_multiprocessing_batch(countries, indicators, years)
        self.update_available_information()
