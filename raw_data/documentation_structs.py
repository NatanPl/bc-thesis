# =============================================================================Documentation Structures Module for Economic Development Forecasting
# Metadata Module
#
# This module provides classes for organizing, categorizing, and managing metadata
# about countries and economic indicators. It serves as a layer between raw data
# and the analysis/forecasting components, providing structured access to country
# datasets along with rich metadata.
#
# The module helps track data availability, quality, and coverage across different
# countries and time periods, which is crucial for effective economic forecasting.
# =============================================================================
import json
from pathlib import Path
import wbgapi as wb

from raw_data.country_data_t import country_data_t


def save_json(obj, path):
    with open(path, "w") as file:
        file.write('{\n')
        items = list(obj.items())
        for i, (key, value) in enumerate(items):
            comma = ',' if i < len(items) - 1 else ''
            key_str = json.dumps(key)
            value_str = json.dumps(value, separators=(',', ':'))
            file.write(f'{key_str}: {value_str}{comma}\n')
        file.write('}\n')


class object_collection_t:
    """
    Collection manager for country or indicator objects.

    This class manages collections of similar objects (either countries or indicators),
    providing methods to load, save, and manipulate these collections. It acts as 
    a container and persistence manager for the documentation objects.

    Attributes:
        path: Path to the JSON file where the collection is stored
        objects: Dictionary of objects keyed by their codes
        categories: Set of all categories used across objects in the collection
        is_countries: Flag indicating if this collection contains countries
        is_indicators: Flag indicating if this collection contains indicators
    """
    path: Path
    objects: dict
    categories: set
    is_countries: bool
    is_indicators: bool
    countries_path = Path("countries.json")
    indicators_path = Path("indicators.json")

    def __init__(
        self,
        countries: bool = False,
        indicators: bool = False,
        folder_path: Path = Path("config"),
    ) -> None:
        """
        Initialize a new collection.

        Args:
            countries: If True, this is a collection of countries
            indicators: If True, this is a collection of indicators
            folder_path: Directory where the JSON files are stored
        """
        if countries and indicators:
            raise ValueError(
                "Collection cannot be both countries and indicators.")

        self.objects = {}
        self.categories = set()
        self.is_countries = countries
        self.is_indicators = indicators
        if countries:
            self.path = folder_path / self.countries_path
            self.is_countries = True
        if indicators:
            self.path = folder_path / self.indicators_path
            self.is_indicators = True

    def add_object(self, obj: object) -> None:
        """
        Add an object to the collection.

        Args:
            obj: The country or indicator object to add
        """
        self.objects[obj.code] = obj

    def update_categories(self) -> None:
        """
        Update the set of unique categories across all objects in the collection.

        This method aggregates all categories from all objects to maintain a master
        list of categories for filtering and organization.
        """
        for item in self.objects:
            for category in self.objects[item].categories:
                self.categories.add(category)

    def load_objects(self) -> None:
        """
        Load objects from the JSON file.

        This method reads the collection's JSON file and instantiates appropriate
        objects (either country_t or indicator_t) for each entry.
        """
        with open(self.path, "r") as file:
            dictionary = json.load(file)
            for key in dictionary.keys():
                # Create country or indicator object based on collection type
                obj = (
                    country_t(self, key, "")
                    if self.is_countries
                    else indicator_t(self, key, "")
                )
                obj.load_from_dict(dictionary[key])
                obj.parent = self
                self.add_object(obj)
        self.update_categories()

    def save_objects(self) -> None:
        """
        Save all objects to the JSON file.

        This method serializes all objects in the collection to a JSON file
        for persistence between program executions.
        """
        if self.objects == {}:
            return
        dictionary = {
            key: self.objects[key].save_to_dict() for key in self.objects.keys()
        }
        save_json(dictionary, self.path)


class base_documentation_t:
    """
    Base class for documentation objects (countries and indicators).

    This abstract class provides common functionality for both country
    and indicator documentation objects, including metadata handling
    and category management.

    Attributes:
        code: Unique identifier code for the object
        name: Human-readable name of the object
        parent: Reference to the parent collection
        categories: List of categories assigned to this object
    """
    code: str
    name: str
    parent: object_collection_t
    categories: list
    in_use: bool
    reason_for_disuse: str

    def __init__(
        self, object_collection: object_collection_t, code: str, name: str
    ) -> None:
        """
        Initialize a base documentation object.

        Args:
            object_collection: Parent collection this object belongs to
            code: Unique identifier code
            name: Human-readable name
        """
        self.code = code
        self.name = name
        self.parent = object_collection
        self.categories = []
        self.in_use = True
        self.reason_for_disuse = ""

    def save(self) -> None:
        """Save this object via the parent collection."""
        self.parent.save_objects()

    def refresh(self) -> None:
        """Reload all objects in the parent collection."""
        self.parent.load_objects()

    def save_to_dict(self, dic) -> dict:
        """
        Serialize this object to a dictionary.

        Returns:
            A dictionary representation of this object, excluding parent reference
            and empty/non-persistent fields, as well as the actual 
        """
        if self.categories == []:
            dic.pop("categories", None)
        dic.pop("data", None)
        dic.pop("data_object", None)
        if dic["in_use"] is True:
            # Only save reason for disuse if not in use, to minimize the JSON
            dic.pop("reason_for_disuse", None)
            dic.pop("in_use", None)
        return dic

    def get_attribute_dict(self) -> dict:
        """
        Get a dictionary of object attributes for serialization.

        Returns:
            Dictionary of object attributes (excluding parent reference)
        """
        return {
            key: getattr(self, key)
            for key in self.__dict__.keys()
            if key != "parent"
        }

    def load_from_dict(self, dic: dict) -> None:
        """
        Deserialize from a dictionary.

        Args:
            dic: Dictionary containing object attributes
        """
        for key in dic:
            setattr(self, key, dic[key])

    def add_category(self, category: str) -> None:
        """
        Add a category to this object and save changes.

        Args:
            category: Category string to add
        """
        if category not in self.categories:
            self.categories.append(category)
            self.save()

    def remove_category(self, category: str) -> None:
        """
        Remove a category from this object and save changes.

        Args:
            category: Category string to remove
        """
        if category in self.categories:
            self.categories.remove(category)
            self.save()

    def discontinue(self, reason: str = "") -> None:
        """
        Mark this object as not in use, with an optional reason.

        Args:
            reason: Explanation for why the object is discontinued
        """
        self.in_use = False
        self.reason_for_disuse = reason


class country_t(base_documentation_t):
    """
    Class for country data and metadata management.

    This class extends base_documentation_t to handle country-specific
    functionality, including data loading/saving and analysis methods
    to evaluate data completeness and quality.

    Attributes:
        data_object: Reference to the actual country data
        in_use: Flag indicating if this country is currently in use
        reason_for_disuse: Explanation if the country is not in use
    """

    def __init__(
        self, object_collection: object_collection_t, code: str, name: str
    ) -> None:
        """
        Initialize a country documentation object.

        Args:
            object_collection: Parent collection
            code: Country code (e.g., 'USA')
            name: Country name
        """
        super().__init__(object_collection, code, name)
        self.data_object: country_data_t = None  # Will hold the actual data

    def save(self) -> None:
        """Save both metadata and the actual data."""
        super().save()
        self.data_object.save_data()

    def save_to_dict(self) -> dict:
        """
        Serialize this country to a dictionary.

        Makes sure data is saved separately before serializing metadata.

        Returns:
            Dictionary representation of this country's metadata
        """
        if self.data_object is not None:
            self.data_object.save_data()
        dic = super().get_attribute_dict()
        # place for subclass specific attribute handling

        return super().save_to_dict(dic)

    def load_data(self, data: country_data_t) -> None:
        """
        Load the country data from a data object.

        Args:
            data: The country_data_t object containing actual data
        """
        self.data_object = data
        # self.update_data_density()
        # self.to_json()

    def offload_data(self) -> None:
        """
        Remove the reference to the data object to free memory.

        This can be used to reduce memory usage when data is not needed.
        """
        self.data_object = None

    def update_data(self, indicators: list[str], years: list[int]) -> None:
        """
        Update the country data for specified indicators and years.

        If the data object doesn't exist yet, it creates one.

        Args:
            indicators: List of indicator codes to update
            years: List of years to fetch data for
        """
        if self.data_object is None:
            self.data_object = country_data_t(self.code)
        self.data_object.update_data(indicators, years)

    def min_year(self) -> int:
        """
        Get the earliest year for which data is available.

        Returns:
            The minimum year as an integer, or 9999 if no data is available
        """
        if self.data_object is None:
            return 9999
        if self.data_object.data_raw is None:
            return 9999

        return min(self.data_object.data_raw.columns.values)

    def max_year(self) -> int:
        """
        Get the latest year for which data is available.

        Returns:
            The maximum year as an integer, or 0 if no data is available
        """
        if self.data_object is None:
            return 0
        if self.data_object.data_raw is None:
            return 0

        return max(self.data_object.data_raw.columns.values)

    # def update_data_density(self) -> None:
    #     if self.data is None:
    #         return
    #     self.data_density = {}
    #     for year in self.data.columns.values:
    #         for indicator in self.data.index.values:
    #             if not pd.isna(self.data.loc[indicator, year]):
    #                 if year in self.data_density.keys():
    #                     self.data_density[year] += 1
    #                 else:
    #                     self.data_density[year] = 1
    #     self.update_data_range()

    # def update_data_range(self) -> None:
    #     min_year = int(min(self.data_density.keys()).replace("YR", ""))
    #     max_year = int(max(self.data_density.keys()).replace("YR", ""))
    #     self.data_range = (min_year, max_year)

    def discontinue(self, reason: str = "") -> None:
        """
        Mark this country as not in use, with an optional reason.

        Args:
            reason: Explanation for why the country is discontinued
        """
        super().discontinue(reason)
        self.save_to_dict()

    def list_indicators_with_data(self) -> list:
        """
        Get a list of indicators that have at least some data points.

        Returns:
            List of indicator codes that have non-null data
        """
        return [
            indicator
            for indicator in self.data_object.data_raw.index.values
            if not self.data_object.data_raw.loc[indicator].isna().all()
        ]

    def list_indicators_with_no_data(self) -> list:
        """
        Get a list of indicators that have no data points.

        Returns:
            List of indicator codes that have only null data
        """
        return [
            indicator
            for indicator in self.data_object.data_raw.index.values
            if self.data_object.data_raw.loc[indicator].isna().all()
        ]

    def list_indicators_with_quantity(self) -> list:
        """
        Get indicators with the count of available data points.

        Returns:
            List of (indicator_code, data_count) tuples, sorted by count descending
        """
        l = []
        for indicator in self.data_object.data_raw.index.values:
            data = self.data_object.data_raw.loc[indicator]
            l.append((indicator, data.count()))
        return sorted(l, key=lambda x: x[1], reverse=True)

    def indicator_availability_in_range(self, year_from: int, year_to: int) -> list:
        """
        Calculate data availability ratios for each indicator in a year range.

        This helps assess how complete the data is for different indicators
        within the specified time period.

        Args:
            year_from: Start year of the range
            year_to: End year of the range

        Returns:
            List of (indicator_code, availability_ratio) tuples, sorted by ratio descending
        """
        year_range = year_to - year_from + 1
        l = []
        for indicator in self.data_object.data_raw.index.values:
            data = self.data_object.data_raw.loc[indicator, year_from:year_to]
            l.append((indicator, data.count() / year_range))
        return sorted(l, key=lambda x: x[1], reverse=True)

    def score_country(self, year_from: int, year_to: int) -> float:
        """
        Calculate an overall data quality score for this country.

        The score represents the average availability ratio across all indicators
        for the specified time period. Higher scores indicate more complete data.

        Args:
            year_from: Start year of the range
            year_to: End year of the range

        Returns:
            A float between 0 and 1 representing overall data completeness
        """
        data = self.indicator_availability_in_range(year_from, year_to)
        return sum(x[1] for x in data) / len(data)


class indicator_t(base_documentation_t):
    """
    Class for economic indicator metadata management.

    This class extends base_documentation_t to handle indicator-specific
    functionality and metadata about economic indicators.

    Attributes:
        indicator_description: Detailed description of the indicator
        indicator_data_range: Range of values the indicator typically takes
        indicator_year_range: Years for which the indicator is defined
        indicator_periodicity: How often the indicator is measured (in years)
    """
    indicator_description: str = ""
    indicator_data_range: tuple = None
    indicator_year_range: tuple = None
    indicator_periodicity: int = None

    def ensure_validity(self) -> None:
        """
        Ensure that the indicator has a valid description.

        If the description is missing, it tries to fetch it from the
        World Bank API using the indicator code.
        """
        if self.indicator_description == "":
            # Fetch indicator metadata from World Bank API
            self.indicator_description = wb.series.metadata(self.code)[
                "value"
            ]

    def save_to_dict(self) -> dict:
        """
        Serialize this indicator to a dictionary.

        Returns:
            Dictionary representation of this indicator's metadata
        """
        dic = super().get_attribute_dict()
        return super().save_to_dict(dic)

    def discontinue(self, reason=""):
        super().discontinue(reason)
        self.save_to_dict()
