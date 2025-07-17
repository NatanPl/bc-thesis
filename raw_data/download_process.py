# =============================================================================
# Multiprocessing Data Download Module
#
# This module contains functionality for downloading economic indicator data
# in parallel processes. It's designed to work within a
# multiprocessing framework to efficiently fetch data for multiple countries
# simultaneously from the World Bank API.
#
# The module provides a worker function that can be executed in
# separate processes, communicating progress back to the main application
# through a multiprocessing Queue.
# =============================================================================

import sys
from raw_data.country_data_t import country_data_t
from multiprocessing import Queue


def download_process(
    country_object: country_data_t,
    indicators: list[str],
    years: tuple[int, int],
    queue: Queue,
    country_code: str,
):
    """
    Worker function for downloading country data in a separate process.

    This function is designed to be run in a separate process as part of a multiprocessing
    pool. It downloads specified economic indicators for a given country and date range,
    providing progress updates through a Queue.

    Args:
        country_object: Country data object with methods to fetch and store data
        indicators: List of economic indicator codes to download (e.g., 'NY.GDP.MKTP.CD')
        years: Tuple containing (start_year, end_year) (inclusive) for the data range
        queue: Multiprocessing Queue for sending progress messages to the main process
        country_code: country code as used by the World Bank for logging and identification purposes

    Note:
        The function uses sys.stdout.flush() to ensure that print statements are 
        immediately visible in multiprocessing environments where stdout buffering 
        might occur.
    """
    # Log entry into the process
    print(f"Entered download_process for {country_code}")
    sys.stdout.flush()  # Ensure print output is visible immediately

    # Send initial status message to the main process if queue is provided
    if queue:
        queue.put_nowait(f"Downloading data for {country_code}.")

    # Generate a list of years from the start and end years and update the country data
    country_object.update_data(indicators, list(range(years[0], years[1] + 1)))

    # Log completion of download
    print(f"Data for {country_code} downloaded.")
    sys.stdout.flush()  # Ensure print output is visible immediately

    # Send completion status message to the main process if queue is available
    if queue:
        queue.put_nowait(f"Data for {country_code} downloaded.")
