"""
This module performs Winsorisation. Currently only One-sided Winsorisation is
implemented.
"""

import typing

from pyspark.sql import DataFrame
from pyspark.sql.functions import col, expr, lit, when


class ValidationError(Exception):
    """Error raised when data frame validation fails."""

    pass


def one_sided_winsorise(
    input_df: DataFrame,
    reference_col: str,
    period_col: str,
    grouping_col: str,
    target_col: str,
    design_col: str,
    l_value_col: str,
    outlier_col: str,
    calibration_col: typing.Optional[str] = None,
    auxiliary_col: typing.Optional[str] = None,
):
    if not isinstance(input_df, DataFrame):
        raise TypeError("input_df must be an instance of pyspark.sql.DataFrame.")

    ratio_cols = [auxiliary_col, calibration_col]
    if any(ratio_cols) and not all(ratio_cols):
        raise TypeError(
            "Both or neither of auxiliary_col and calibration_col must be specified."
        )

    expected_cols = {
        reference_col,
        period_col,
        grouping_col,
        target_col,
        design_col,
        l_value_col,
    }

    if calibration_col is not None:
        expected_cols.add(calibration_col)

    if auxiliary_col is not None:
        expected_cols.add(auxiliary_col)

    for col_name in expected_cols:
        if not isinstance(col_name, str):
            raise TypeError("Provided column names must be strings.")

        if not col_name:
            raise ValueError("Provided column names must not be empty.")

    missing_cols = expected_cols - set(input_df.columns)
    if missing_cols:
        raise ValidationError(f"Missing columns: {', '.join(c for c in missing_cols)}")

    for col_name in expected_cols:
        if input_df.filter(col(col_name).isNull()).count() > 0:
            raise ValidationError(f"Column {col_name} must not contain null values.")

    col_list = [
        col(reference_col).alias("reference"),
        col(period_col).alias("period"),
        col(grouping_col).alias("grouping"),
        col(target_col).alias("target"),
        col(design_col).alias("design"),
        col(l_value_col).alias("l_value"),
    ]

    if auxiliary_col is not None:
        col_list.append(col(auxiliary_col).alias("auxiliary"))

    else:
        col_list.append(lit(1.0).alias("auxiliary"))

    if calibration_col is not None:
        col_list.append(col(calibration_col).alias("calibration"))

    else:
        col_list.append(lit(1.0).alias("calibration"))

    group_cols = ["period", "grouping"]
    df = input_df.select(col_list)

    return (
        df.join(
            (
                df.withColumn("target_design", expr("target * design"))
                .withColumn("aux_design", expr("auxiliary * design"))
                .groupBy(group_cols)
                .agg({"target_design": "sum", "aux_design": "sum"})
                .withColumn(
                    "ratio_sum_target_sum_aux",
                    col("sum(target_design)") / col("sum(aux_design)"),
                )
            ),
            group_cols,
        )
        .withColumn("winsorisation_value", expr("ratio_sum_target_sum_aux * auxiliary"))
        .withColumn(
            "k_value",
            expr("winsorisation_value + (l_value/((design * calibration) -1))"),
        )
        .withColumn(
            "modified_target",
            when(
                "target > k_value",
                expr(
                    """
                        (target/(design*calibration))
                        + (k_value - (k_value/(design*calibration)))
                    """
                ),
            ).otherwise(col("target")),
        )
        .withColumn("outlier", expr("modified_target/target"))
        .select(
            col("reference").alias(reference_col),
            col("period").alias(period_col),
            col("grouping").alias(grouping_col),
            col("outlier").alias(outlier_col),
        )
    )
