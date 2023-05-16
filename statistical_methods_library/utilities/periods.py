from pyspark.sql import Column

def calculate_previous_period(period: Column, relative: int):
    period = period.cast("integer")
    return (
        period
        - relative
        - 88 * (relative // 12 + (period % 100 <= relative % 12).cast("integer"))
    ).cast("string")

def calculate_next_period(period: Column, relative: int) -> Column:
    period = period.cast("integer")
    return (
        period
        + relative
        + 88
        * (relative // 12 + ((period % 100) + (relative % 12) > 12).cast("integer"))
    ).cast("string")
