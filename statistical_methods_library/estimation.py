"""
Estimates design and calibration weights based on Expansion and Ratio estimation.
"""

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
    """
    Perform estimation of design and calibration weights using Expansion and
    Ratio estimation.

    ###Arguments
    * input_df: The input data frame.
    * period_col: The name of the column containing the period information for
      the contributor.
    * strata_col: The name of the column containing the strata of the contributor.
    * sample_inclusion_marker_col: The name of the column containing a marker
      for whether to include the contributor in the sample or only in the
      population. This column must only contain values of 0 or 1 where 0 means
      to exclude the contributor from the sample and 1 means the contributor
      will be included in the sample count.
    * death_marker_col: The name of the column containing a marker for whether
      the contributor is dead. This column must only contain the values 0
      meaning the contributor is not dead and 1 meaning that the contributor is dead.
    * h_col: The name of the column containing the h value for the strata.
    * auxiliary_col: The name of the column containing the auxiliary value for
      the contributor.
    * calibration_group_col: The name of the column containing the calibration
      group for the contributor.

    ###Returns
    A data frame containing:

    * `period_col`
    * `strata_col`
    * `design_weight`
    * `calibration_weight`

    ###Notes

    Either both or neither of `death_marker_col` and `h_col` must be specified.
    If they are then the design weight is adjusted using birth-death
    adjustment, otherwise it is not.

    If `auxiliary_col` is specified then one of Separate Ratio or Combined Ratio
    estimation is performed. This depends on whether `calibration_group_col`
    is specified. If so then Combined Ratio estimation is performed, otherwise
    Separate Ratio estimation is performed. If `auxiliary_col` is not
    specified then only Expansion estimation is performed and specifying
    `calibration_group_col` raises an error.

    All specified columns must be fully populated. If not an error is raised.
    """

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
    # Perform Expansion estimation. If we've got a death marker and h value
    # then we'll use these, otherwise they'll be 0 and thus the
    # calculation for design weight just multiplies the unadjusted design
    # weight by 1.
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

    # The ratio calculation for calibration weight is the same for Separate
    # and Combined estimation with the exception of the grouping.
    def calibration_calculation(df: DataFrame, group_col: str) -> DataFrame:
        group_cols = ["period", group_col]
        return df.groupBy(group_cols).selectExpr(
            """
                SUM(auxiliary) / SUM(auxiliary * unadjusted_design_weight)
                AS calibration_weight
            """
        )

    if "auxiliary" in working_df.columns:
        # We can perform some sort of ratio estimation since we have an
        # auxiliary value.
        working_df = working_df.join(design_df, ["period", "strata"])
        if "calibration_group" in working_df.columns:
            # We have a calibration group so perform Combined Ratio estimation.
            calibration_df = calibration_calculation(working_df, "calibration_group")

        else:
            # No calibration group so perform Separate Ratio estimation.
            calibration_df = calibration_calculation(working_df, "strata")

        return working_df.join(calibration_df, ["period", "strata"]).select(
            col("period").alias(period_col),
            col("strata").alias(strata_col),
            col("design_weight"),
            col("calibration_weight"),
        )

    else:
        # No auxiliary values so only perform Expansion estimation.
        return design_df.select(
            col("period").alias(period_col),
            col("strata").alias(strata_col),
            col("design_weight"),
        )
