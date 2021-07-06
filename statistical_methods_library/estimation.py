"""
Estimates design and calibration weights based on Expansion and Ratio estimation.
"""

import typing

from pyspark.sql import DataFrame
from pyspark.sql.functions import col, count, first, lit, sum


class ValidationError(Exception):
    """
    Error raised when validating the input data frame.
    """

    pass


def estimate(
    input_df: DataFrame,
    period_col: str,
    strata_col: str,
    sample_marker_col: str,
    death_marker_col: typing.Optional[str] = None,
    h_value_col: typing.Optional[str] = None,
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
    * sample_marker_col: The name of the column containing a marker
      for whether to include the contributor in the sample or only in the
      population. This column must only contain values of 0 or 1 where 0 means
      to exclude the contributor from the sample and 1 means the contributor
      will be included in the sample count.
    * death_marker_col: The name of the column containing a marker for whether
      the contributor is dead. This column must only contain the values 0
      meaning the contributor is not dead and 1 meaning that the contributor is dead.
    * h_value_col: The name of the column containing the h value for the strata.
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

    Either both or neither of `death_marker_col` and `h_value_col` must be specified.
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

    # --- Validate params ---
    if not isinstance(input_df, DataFrame):
        raise TypeError("input_df must be an instance of pyspark.sql.DataFrame")

    death_cols = (death_marker_col, h_value_col)
    if any(death_cols) and not all(death_cols):
        raise TypeError(
            "Either both or none of death_marker_col and h_value_col must be specified."
        )

    if calibration_group_col is not None and auxiliary_col is None:
        raise TypeError(
            "If calibration_group_col is specified then auxiliary_col must be provided."
        )

    expected_cols = [
        period_col,
        strata_col,
        sample_marker_col,
    ]
    if death_marker_col is not None:
        expected_cols += [death_marker_col, h_value_col]
    if auxiliary_col is not None:
        expected_cols.append(auxiliary_col)

    if calibration_group_col is not None:
        expected_cols.append(calibration_group_col)

    # Check to see if the column names are of the correct types, not empty and
    # do not contain nulls.
    for col_name in expected_cols:
        if not isinstance(col_name, str):
            raise TypeError("All column names provided in params must be strings.")

        if col_name == "":
            raise ValueError(
                "Column name strings provided in params must not be empty."
            )

        if input_df.filter(col(col_name).isNull()).count() > 0:
            raise ValidationError(
                f"Input column {col_name} must not contain null values."
            )

    # Check to see if any required columns are missing from the dataframe.
    missing_cols = set(expected_cols) - set(input_df.columns)
    if missing_cols:
        raise ValidationError(f"Missing columns: {', '.join(c for c in missing_cols)}")

    # As per the documentation, death marker and sample marker columns must
    # only contain 0 or 1.
    for col_name in (sample_marker_col, death_marker_col):
        if input_df.filter((col(col_name) != 0) & (col(col_name) != 1)).count() > 0:
            raise ValidationError(
                f"Input column {col_name} must only contain values of 0 or 1."
            )

    # --- prepare our working data frame ---
    col_list = [
        col(period_col).alias("period"),
        col(strata_col).alias("strata"),
        col(sample_marker_col).alias("sample_marker"),
    ]

    if death_marker_col is not None:
        col_list += [
            col(death_marker_col).alias("death_marker"),
            col(h_value_col).alias("h_value"),
        ]

    else:
        col_list += [lit(0.0).alias("death_marker"), lit(0.0).alias("h_value")]

    if auxiliary_col is not None:
        col_list.append(col(auxiliary_col).alias("auxiliary"))

    if calibration_group_col is not None:
        col_list.append(col(calibration_group_col).alias("calibration_group"))

    working_df = input_df.select(col_list)
    # --- Expansion estimation ---
    # If we've got a death marker and h value then we'll use these, otherwise
    # they'll be 0 and thus the calculation for design weight just multiplies
    # the unadjusted design weight by 1.
    # Due to the fact that sample and death markers are either 0 or 1, summing
    # those columns gives the number of contributors in the sample and the
    # number of dead contributors respectively. There's only ever 1 h value
    # per strata so we can just take the first one in that period and strata,
    # and every contributor must have a sample marker so counting this column
    # gives us the total population.
    design_df = (
        working_df.groupBy(["period", "strata"])
        .agg(
            sum(col("sample_marker")),
            sum(col("death_marker")),
            first(col("h_value")),
            count(col("sample_marker")),
        )
        .withColumn(
            "unadjusted_design_weight",
            col("count(sample_marker)") / col("sum(sample_marker)"),
        )
        .withColumn(
            "design_weight",
            (
                col("unadjusted_design_weight")
                * (
                    1
                    + (
                        col("first(h_value)")
                        * col("sum(death_marker)")
                        / (col("sum(sample_marker)") - col("sum(death_marker)"))
                    )
                )
            ),
        )
    )

    # --- Ratio estimation ---
    # Note: if we don't have the columns for this then only Expansion
    # estimation is performed.

    # The ratio calculation for calibration weight is the same for Separate
    # and Combined estimation with the exception of the grouping.
    def calibration_calculation(df: DataFrame, group_col: str) -> DataFrame:
        group_cols = ["period", group_col]
        return (
            df.withColumn(
                "aux_design", col("auxiliary") * col("unadjusted_design_weight")
            )
            .groupBy(group_cols)
            .agg({"auxiliary": "sum", "aux_design": "sum"})
            .withColumn(
                "calibration_weight", col("sum(auxiliary)") / col("sum(aux_design)")
            )
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
        # No auxiliary values so return the results of Expansion estimation.
        return design_df.select(
            col("period").alias(period_col),
            col("strata").alias(strata_col),
            col("design_weight"),
        )
