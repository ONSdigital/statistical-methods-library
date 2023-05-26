"""
Functions to work with periods. Periods are in "yyyymm" format i.e. 4 digit
year and 2 digit month. The terms year and month here do not necessarily have
to be actual years and months although the period calculations assume 12
periods in a year. In particular, because these functions are intended to work
with data where the periods are used to identify sample periods rather than to
act as calendar dates (e.g. accounting periods in economic survey data) no
calendar related arithmetic is performed (e.g. leap years etc) with all
calculations being strictly period based.

For Copyright information, please see LICENCE.
"""

from pyspark.sql import Column
from pyspark.sql.functions import lpad


def calculate_previous_period(period: Column, relative: int):
    """
    Calculate the previous period.

    Args:
        period - The column containing the period.
        relative: The mount of periods to subtract.

    Returns:
    A column containing the previous period.
    """

    period = period.cast("integer")
    return lpad(
        (
            period
            - relative
            - 88 * (relative // 12 + (period % 100 <= relative % 12).cast("integer"))
        ).cast("string"),
        6,
        "0",
    )


def calculate_next_period(period: Column, relative: int) -> Column:
    """
    Calculate the next period.

    Args:
        period - The column containing the period.
        relative: The mount of periods to add.

    Returns:
    A column containing the next period.
    """

    period = period.cast("integer")
    return lpad(
        (
            period
            + relative
            + 88
            * (relative // 12 + ((period % 100) + (relative % 12) > 12).cast("integer"))
        ).cast("string"),
        6,
        "0",
    )
