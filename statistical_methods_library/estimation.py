from pyspark.sql import DataFrame
from pyspark.sql.functions import col, lit, count, expr


def estimation_by_expansion(
    input_df: DataFrame,
    strata_col: str,
    period_col: str,
    sample_inclusion_marker_col: str,
    death_marker_col: str,
    h_col: str,
) -> DataFrame:
    group_cols = [period_col, strata_col]
    if death_marker_col not in input_df.columns:
        prepped_df = input_df.withColumn(death_marker_col, lit(0))
        prepped_df = prepped_df.withColumn(h_col, lit(0))
    else:
        prepped_df = input_df

    return prepped_df.groupBy(group_cols)\
        .agg(count(col(sample_inclusion_marker_col) == 1)).alias("sample_count")\
        .agg(count(col(sample_inclusion_marker_col))).alias("population_count")\
        .agg(count(col(death_marker_col) == 1)).alias("death_count")\
        .withColumn(
            "design_weight", expr("(population_count / sample_count) * (1 + (" + h_col + " * (death_count/(sample_count - death_count))))")
        )


def estimation_calibration_calculation(
        input_df: DataFrame,
        group_col: str,
        period_col: str,
        sample_inclusion_marker_col: str,
        auxiliary_col: str
) -> DataFrame:
    group_cols = [period_col, group_col]
    input_df.groupBy(group_cols) \
        .agg(count(col(sample_inclusion_marker_col) == 1)).alias("sample_count") \
        .agg(count(col(sample_inclusion_marker_col))).alias("population_count") \
        .agg(sum(col(auxiliary_col))).alias("sum_auxiliary") \
        .agg(expr("sum(" + auxiliary_col + " * population_count / sample_count)")).alias("sum_auxiliary_design")\
        .withColumn("calibration_weight", expr("sum_auxiliary / sum_auxiliary_design"))


def estimation_by_ratio(
        input_df: DataFrame,
        strata_col: str,
        period_col: str,
        sample_inclusion_marker_col: str,
        auxiliary_col: str,
        death_marker_col: str,
        h_marker_col: str,
) -> DataFrame:
    df_with_design = estimation_by_expansion(input_df, strata_col, period_col, sample_inclusion_marker_col, death_marker_col, h_marker_col)
    df_with_calibration = estimation_calibration_calculation(input_df, strata_col, period_col, sample_inclusion_marker_col, auxiliary_col)


def estimation_by_combined_ratio(
        input_df: DataFrame,
        strata_col: str,
        period_col: str,
        sample_inclusion_marker_col: str,
        calibration_group_col: str,
        auxiliary_col: str,
        death_marker_col: str,
        h_marker_col: str,
) -> DataFrame:
    df_with_design = estimation_by_expansion(input_df, strata_col, period_col, sample_inclusion_marker_col, death_marker_col, h_marker_col)
    df_with_calibration = estimation_calibration_calculation(input_df, calibration_group_col, period_col, sample_inclusion_marker_col, auxiliary_col)
