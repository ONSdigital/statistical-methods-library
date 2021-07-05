from pyspark.sql import DataFrame
from pyspark.sql.functions import col, lit, count


def estimation_by_expansion(
    input_df: DataFrame,
    strata_col: str,
    period_col: str,
    sample_inclusion_marker_col: str,
    death_marker_col: str,
    h_marker_col: str,
) -> DataFrame:
    group_cols = [period_col, strata_col]
    if death_marker_col not in input_df.columns:
        prepped_df = input_df.withColumn(death_marker_col, lit(0))
        prepped_df = prepped_df.withColumn(h_marker_col, lit(0))
    else:
        prepped_df = input_df

    working_df = prepped_df.groupBy(group_cols)\
        .agg(count(col(sample_inclusion_marker_col) == 1)).alias("sample_count")\
        .count().alias("population_count")\
        .agg(count(col(death_marker_col) == 1)).alias("death_count")
