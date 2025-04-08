"""
Estimate design weights and calibration factors based on Expansion and Ratio estimation.
For Copyright information, please see LICENCE.
"""

import typing

from pyspark.sql import DataFrame
from pyspark.sql.functions import col, count, first, lit, sum, when
from pyspark.sql.types import BooleanType, DecimalType, StringType

from statistical_methods_library.utilities import validation


def estimate(
    input_df: DataFrame,
    unique_identifier_col: str,
    period_col: str,
    strata_col: str,
    sample_marker_col: str,
    adjustment_marker_col: typing.Optional[str] = None,
    h_value_col: typing.Optional[str] = None,
    out_of_scope_full: typing.Optional[bool] = None,
    auxiliary_col: typing.Optional[str] = None,
    calibration_group_col: typing.Optional[str] = None,
    unadjusted_design_weight_col: typing.Optional[str] = None,
    design_weight_col: typing.Optional[str] = "design_weight",
    calibration_factor_col: typing.Optional[str] = "calibration_factor",
) -> DataFrame:
    """
    Perform Horvitz-Thompson estimation.

    Design weights and calibration factors are estimated using Expansion and
    Ratio estimation.

    Args:
        input_df: The input data frame.
        unique_identifier_col: The name of the column containing the unique
          identifier for a contributor.
        period_col: The name of the column identifying the period. Values in
          this column are merely for grouping purposes and thus do not need
          to conform to the conventions or data type documented for periods
          in the `utilities.periods` module.
        strata_col: The name of the column containing the stratum.
        sample_marker_col: The name of the column containing a boolean marker
          where True means to include the contributor in the sample otherwise
          the contributor is just included in the population.
        adjustment_marker_col: The name of the column marking a contributor as
          in scope ("I"), out of scope ("O") or dead ("D").
        h_value_col: The name of the column containing the boolean h value for
          the stratum. The value must be the same per stratum and period.
        out_of_scope_full: The type of out of scope adjustment to use when an
          `out_of_scope_marker_col` is provided. True causes the out of
          scope adjustment to be used on both sides of the adjustment
          fraction whereas False causes the out of scopeadjustment only on
          the denominator of the adjustment fraction.
        auxiliary_col: The name of the column containing the auxiliary value.
          Providing this column causes Ratio Estimation to be performed,
          otherwise only Expansion Estimation takes place.
        calibration_group_col: The name of the column containing the calibration
          group. This column can only be provided when `auxiliary_col` is also
          provided and it causes Combined Ratio Estimation to be performed,
          otherwise Separate Ratio Estimation takes place.
        unadjusted_design_weight_col: The name of the column which will contain
          the unadjusted design weight. The column isn't output unless this
          column is specified and design weight adjustment is performed.
        design_weight_col: The name of the column which will contain the
          design weight. If birth-death adjustment is being performed this
          will contain the adjusted value.
        calibration_factor_col: The name of the column which will contain the
          calibration factor if Ratio Estimation is performed.

    Returns:
    A data frame containing the estimated weights. The exact columns depend on
    the type of estimation performed as specified by the provided arguments. The
    data frame contains a row for each period, stratum and (if Combined Ratio
    Estimation is performed) calibration group combination in the input data.

    Either both or neither of `adjustment_marker_col` and `h_value_col` must be specified.
    If they are then the design weight is adjusted using birth-death
    adjustment, otherwise it is not.
    If `out_of_scope_full` is also specified, out of scope adjustment
    is performed during birth-death adjustment.

    All specified input columns must be fully populated.
    Since `design_weight_col` and `calibration_factor_col` are both
    output columns this does not apply to them, and any values they contain prior
    to calling the method will be ignored.
    """

    death_cols = (adjustment_marker_col, h_value_col)
    if any(death_cols) and not all(death_cols):
        raise TypeError(
            "Either both or none of death_marker_col and h_value_col must be specified."
        )

    if out_of_scope_full is not None and not all(death_cols):
        raise TypeError(
            "For out of scope, adjustment_marker_col and h_value_col must be specified."
        )

    if calibration_group_col is not None and auxiliary_col is None:
        raise TypeError(
            "If calibration_group_col is specified then auxiliary_col must be provided."
        )

    # --- Validate params ---
    input_params = {
        "unique_identifier": unique_identifier_col,
        "period": period_col,
        "strata": strata_col,
        "sample_marker": sample_marker_col,
    }

    optional_params = {
        "adjustment_marker": adjustment_marker_col,
        "h_value": h_value_col,
        "auxiliary": auxiliary_col,
        "calibration_group": calibration_group_col,
    }

    input_params.update({k: v for k, v in optional_params.items() if v is not None})

    type_mapping = {
        "sample_marker": BooleanType,
        "adjustment_marker": StringType,
        "h_value": BooleanType,
        "auxiliary": DecimalType,
    }

    aliased_df = validation.validate_dataframe(
        input_df, input_params, type_mapping, ["unique_identifier", "period"]
    )

    # h values must not change within a stratum
    if h_value_col is not None:
        validation.validate_one_value_per_group(
            input_df, [period_col, strata_col], h_value_col
        )

    # Values for the marker column used for birth-death and out of scope adjustment.
    # I - In Scope, O - Out Of Scope, D - Dead
    all_adjustment_markers = {"I", "O", "D"}
    death_adjustment_markers = {"I", "D"}

    if adjustment_marker_col is not None:
        validation.validate_no_matching_rows(
            input_df,
            ((~col(sample_marker_col)) & (col(adjustment_marker_col) != "I")),
            "Unsampled responders must only contain an 'I' marker.",
        )
        if out_of_scope_full is not None:
            validation.validate_no_matching_rows(
                input_df,
                (~col(adjustment_marker_col).isin(all_adjustment_markers)),
                f"The {adjustment_marker_col} must only contain 'I', 'O' or 'D'.",
            )
        else:
            validation.validate_no_matching_rows(
                input_df,
                (~col(adjustment_marker_col).isin(death_adjustment_markers)),
                f"The {adjustment_marker_col} must only contain 'I' or 'D'.",
            )

    # --- prepare our working data frame ---
    working_df = aliased_df.withColumn(
        "sample_marker", col("sample_marker").cast(DecimalType())
    )
    if adjustment_marker_col is None:
        working_df = working_df.withColumn("adjustment_marker", lit("I"))
        working_df = working_df.withColumn("h_value", lit(0))
    else:
        working_df = working_df.withColumn(
            "h_value", col("h_value").cast(DecimalType())
        )

    def count_conditional(cond):
        return sum(when(cond, 1).otherwise(0))

    # --- Expansion estimation ---
    # If we've got an adjustment marker and h value then we'll use these, otherwise
    # they'll be 0 and thus the calculation for design weight just multiplies
    # the unadjusted design weight by 1. adjustment marker is counted based on
    # marker provided. Due to the fact that sample is either 0 or 1
    # (after converting bool to int), summing this column gives
    # the number of contributors in the sample. There's only ever
    # 1 h value per strata, so we can just take the first one in that period and strata,
    # and every contributor must have a sample marker so counting this column
    # gives us the total population.

    design_df = (
        working_df.groupBy(["period", "strata"])
        .agg(
            sum(col("sample_marker")).alias("sample_sum"),
            count_conditional(col("adjustment_marker") == "D").alias("death_marker"),
            first(col("h_value")),
            count_conditional(col("adjustment_marker") == "O").alias(
                "out_of_scope_marker"
            ),
            count(col("sample_marker")).alias("sample_count"),
        )
        .withColumn(
            "unadjusted_design_weight",
            col("sample_count") / col("sample_sum"),
        )
    )

    if out_of_scope_full is True or out_of_scope_full is None:
        design_df = design_df.withColumn(
            "out_of_scope_marker_numerator", col("out_of_scope_marker")
        )
        design_df = design_df.withColumn(
            "out_of_scope_marker_denominator", col("out_of_scope_marker")
        )
    else:
        design_df = design_df.withColumn("out_of_scope_marker_numerator", lit(0))
        design_df = design_df.withColumn(
            "out_of_scope_marker_denominator", col("out_of_scope_marker")
        )

    design_df = design_df.withColumn(
        "design_weight",
        (
            col("unadjusted_design_weight")
            * (
                1
                + (
                    col("first(h_value)")
                    * (col("death_marker") + col("out_of_scope_marker_numerator"))
                    / (
                        col("sample_sum")
                        - col("death_marker")
                        - col("out_of_scope_marker_denominator")
                    )
                )
            )
        ),
    ).drop(
        "sample_sum",
        "death_marker",
        "first(h_value)",
        "out_of_scope_marker_numerator",
        "out_of_scope_marker_denominator",
        "sample_count",
    )

    # --- Ratio estimation ---
    # Note: if we don't have the columns for this then only Expansion
    # estimation is performed.

    # The ratio calculation for calibration factor is the same for Separate
    # and Combined estimation except for the grouping.
    def calibration_calculation(df: DataFrame, group_col: str) -> DataFrame:
        group_cols = ["period", group_col]
        return (
            df.withColumn(
                "aux_design",
                col("auxiliary")
                * col("unadjusted_design_weight")
                * col("sample_marker"),
            )
            .groupBy(group_cols)
            .agg({"auxiliary": "sum", "aux_design": "sum"})
            .withColumn(
                "calibration_factor", col("sum(auxiliary)") / col("sum(aux_design)")
            )
        )

    return_col_list = [
        col("period").alias(period_col),
        col("strata").alias(strata_col),
    ]
    if auxiliary_col is not None:
        # We can perform some sort of ratio estimation since we have an
        # auxiliary value.
        working_df = working_df.join(design_df, ["period", "strata"])
        if calibration_group_col is not None:
            # We have a calibration group so perform Combined Ratio estimation.
            return_col_list.append(
                col("calibration_group").alias(calibration_group_col)
            )
            estimated_df = (
                working_df.select(
                    "period",
                    "strata",
                    "calibration_group",
                    "unadjusted_design_weight",
                    "design_weight",
                )
                .distinct()
                .join(
                    calibration_calculation(working_df, "calibration_group"),
                    ["period", "calibration_group"],
                )
            )

        else:
            # No calibration group so perform Separate Ratio estimation.
            estimated_df = design_df.join(
                calibration_calculation(working_df, "strata"), ["period", "strata"]
            )

        return_col_list += [
            col("design_weight").alias(design_weight_col),
            col("calibration_factor").alias(calibration_factor_col),
        ]

    else:
        # No auxiliary values so return the results of Expansion estimation.
        return_col_list.append(col("design_weight").alias(design_weight_col))
        estimated_df = design_df

    if unadjusted_design_weight_col is not None:
        return_col_list.append(
            col("unadjusted_design_weight").alias(unadjusted_design_weight_col)
        )

    return estimated_df.select(return_col_list)
