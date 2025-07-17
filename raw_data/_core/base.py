from pathlib import Path

from raw_data.documentation_structs import country_t, indicator_t, object_collection_t

import multiprocessing as mp


class CoreBase:
    """
    Sort of .h file for the Core class.
    """
    MIN_YEAR = 1960
    MAX_YEAR = 2020

    folder_config: Path
    folder_data: Path
    folder_models: Path
    folder_logs: Path
    folder_reports: Path

    countries_data: object_collection_t
    indicators_data: object_collection_t
    available_countries: set[country_t]
    available_indicators: set[indicator_t]

    min_year: int
    max_year: int
    selected_countries: set[country_t]
    selected_indicators: set[indicator_t]
    selected_years: tuple[int, int]

    multiprocessing: bool
    debug: bool
    pool_initialized: bool

    Pool: mp.Pool
    Queue: mp.Queue
    Manager: mp.Manager  # type: ignore

    def update_available_information(self): pass

    def get_available_countries(self) -> set[country_t]: pass

    def get_available_indicators(self) -> set[indicator_t]: pass

    def get_all_countries(self) -> set[country_t]: pass
    def get_all_indicators(self) -> set[indicator_t]: pass
