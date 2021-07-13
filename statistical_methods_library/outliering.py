"""
This module performs outliering. Currently 1-sided Winsorisation is
implemented.
"""
import typing

from pyspark.sql import DataFrame
from pyspark.sql.functions import col, expr, lit, when


def one_sided_winsorisation(
    input_df: DataFrame,
    reference_col: str,
    period_col: str,
    grouping_col: str,
    target_col: str,
    design_weight_col: str,
    l_value_col: str,
    outlier_weight_col: str,
    calibration_weight_col: typing.Optional[str] = None,
    auxiliary_col: typing.Optional[str] = None,
):
    col_list = [
        col(reference_col).alias("reference"),
        col(period_col).alias("period"),
        col(grouping_col).alias("grouping"),
        col(target_col).alias("target"),
        col(design_weight_col).alias("design"),
        col(l_value_col).alias("l_value"),
    ]

    if auxiliary_col is not None:
        col_list.append(col(auxiliary_col).alias("auxiliary"))

    else:
        col_list.append(lit(1.0).alias("auxiliary"))

    if calibration_weight_col is not None:
        col_list.append(col(calibration_weight_col).alias("calibration"))

    else:
        col_list.append(lit(1.0).alias("calibration"))

    group_cols = ["period", "grouping"]
    df = input_df.select(col_list)
    return (
        df.join(
            df.groupBy(group_cols).withColumn(
                "ratio_sum_target_sum_aux",
                expr("sum(target * design)/sum(auxiliary * design)"),
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
            col("outlier").alias(outlier_weight_col),
        )
    )
