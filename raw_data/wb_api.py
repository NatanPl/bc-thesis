# =============================================================================
# World Bank API Interface Module
#
# This module provides functionality to interact with the World Bank API
# using the wbgapi library. It allows querying economic indicators for
# specified countries over defined time periods.
# =============================================================================

# External library imports
import wbgapi as wb  # World Bank API wrapper
import pandas as pd  # Data manipulation library


def query_api(countries, indicators, years) -> pd.DataFrame:
    """
    Query the World Bank API for the specified countries, indicators 
    and years.
    
    Parameters:
        countries (list): List of country codes (e.g., ['USA', 'CHN'])
        indicators (list): List of World Bank indicator codes (e.g., ['NY.GDP.MKTP.CD'])
        years (list): List of years to query data for (e.g., ['2010', '2020'])
        
    Returns:
        pd.DataFrame: DataFrame containing the queried data, or an empty DataFrame if the query fails
    """
    try:
        # Execute the API query through the World Bank API wrapper
        data = wb.data.DataFrame(
            series=indicators, economy=countries, time=years
        )
    except Exception as e:
        # Handle any API errors gracefully, report the country for which the query failed
        print(
            f"An error occurred while querying the World Bank API for "
            f"{countries}: {e}"
        )
        return pd.DataFrame()  # Return empty DataFrame on error

    # Post-processing of the returned data
    # If only one indicator was requested, use it as the index
    if len(indicators) == 1:
        data.index = [indicators[0]]

    # If only one year was requested, use it as the column name
    if len(years) == 1:
        data.columns = [years[0]]

    # Clean up column names: The API returns years prefixed with "YR"
    # This loop removes that prefix for cleaner data representation
    for col in data.columns:
        if isinstance(col, str) and col.startswith("YR"):
            data.rename(columns={col: int(col[2:])}, inplace=True)

    return data  # Return the processed DataFrame
