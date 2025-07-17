
from raw_data._core.base import CoreBase

from raw_data.documentation_structs import country_t, indicator_t
from typing import Iterable


class Getters(CoreBase):
    def get_all_countries(self) -> Iterable[country_t]:
        """
        Return all country objects defined in configuration.

        Returns:
            Dictionary of country objects, keyed by country code
        """
        return self.countries_data.objects.values()

    def get_all_indicators(self) -> Iterable[indicator_t]:
        """
        Return all indicator objects defined in configuration.

        Returns:
            Dictionary of indicator objects, keyed by indicator code
        """
        return self.indicators_data.objects.values()

    def get_available_countries(self) -> list[country_t]:
        """
        Return countries that have data available.

        Returns:
            List of country objects that have data files
        """
        return [
            country
            for country in self.get_all_countries()
            if country.code in self.available_countries
        ]

    def get_available_indicators(self) -> list[indicator_t]:
        """
        Return indicators that have data available.

        Returns:
            List of indicator objects that have data in at least one country file
        """
        return [
            indicator
            for indicator in self.get_all_indicators()
            if indicator.code in self.available_indicators
        ]
