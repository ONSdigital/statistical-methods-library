import typing

from pyspark.sql import DataFrame
from pyspark.sql.functions import col, lit


def estimation(
    input_df: DataFrame,
    period_col: str,
    strata_col: str,
    sample_inclusion_marker_col: str,
    death_marker_col: typing.Optional[str] = None,
    h_col: typing.Optional[str] = None,
    auxiliary_col: typing.Optional[str] = None,
    calibration_group_col: typing.Optional[str] = None,
) -> DataFrame:
    col_list = [
        col(period_col).alias("period"),
        col(strata_col).alias("strata"),
        col(sample_inclusion_marker_col).alias("sample_marker"),
    ]

    if death_marker_col is not None:
        col_list += [
            col(death_marker_col).alias("death_marker"),
            col(h_col).alias("h_value"),
        ]

    else:
        col_list += [lit(0.0).alias("death_marker"), lit(0.0).alias("h_value")]

    if auxiliary_col is not None:
        col_list.append(col(auxiliary_col).alias("auxiliary"))

    if calibration_group_col is not None:
        col_list.append(col(calibration_group_col).alias("calibration_group"))

    working_df = input_df.select(col_list)
    design_df = working_df.groupBy(["period", "strata"]).selectExpr(
        "period",
        "strata",
        "SUM(sample_marker) as sample_count",
        "SUM(death_marker) as death_count",
        "(COUNT(sample_marker) / sample_count) AS unadjusted_design_weight",
        """
            unadjusted_design_weight * (1 + (h_value * (death_count
            / (sample_count - death_count)))) AS design_weight
        """,
    )

    def calibration_calculation(df: DataFrame, group_col: str) -> DataFrame:
        group_cols = ["period", group_col]
        return df.groupBy(group_cols).selectExpr(
            """
                SUM(auxiliary) / SUM(auxiliary * unadjusted_design_weight)
                AS calibration_weight
            """
        )

    if "auxiliary" in working_df.columns:
        working_df = working_df.join(design_df, ["period", "strata"])
        if "calibration_group" in working_df.columns:
            calibration_df = calibration_calculation(working_df, "calibration_group")
        else:
            calibration_df = calibration_calculation(working_df, "strata")

        return working_df.join(calibration_df, ["period", "strata"]).select(
            col("period").alias(period_col),
            col("strata").alias(strata_col),
            col("design_weight"),
            col("calibration_weight"),
        )

    else:
        return design_df.select(
            col("period").alias(period_col),
            col("strata").alias(strata_col),
            col("design_weight"),
        )
