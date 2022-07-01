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
    unique_identifier_col: str,
    period_col: str,
    strata_col: str,
    sample_marker_col: str,
    death_marker_col: typing.Optional[str] = None,
    h_value_col: typing.Optional[str] = None,
    out_of_scope_marker_col: typing.Optional[str] = None,
    out_of_scope_full: typing.Optional[bool] = None,
    auxiliary_col: typing.Optional[str] = None,
    calibration_group_col: typing.Optional[str] = None,
    unadjusted_design_weight_col: typing.Optional[str] = None,
    design_weight_col: typing.Optional[str] = "design_weight",
    calibration_weight_col: typing.Optional[str] = "calibration_weight",
) -> DataFrame:
    """
    Perform estimation of design and calibration weights using Expansion and
    Ratio estimation.

    ###Arguments
    * `input_df`: The input data frame.
    * `unique_identifier_col`: The name of the column containing the unique identifier
      for the contributors.
    * `period_col`: The name of the column containing the period information for
      the contributor.
    * `strata_col`: The name of the column containing the strata of the contributor.
    * `sample_marker_col`: The name of the column containing a marker
      for whether to include the contributor in the sample or only in the
      population. This column must only contain values of 0 or 1 where 0 means
      to exclude the contributor from the sample and 1 means the contributor
      will be included in the sample count.
    * `death_marker_col`: The name of the column containing a marker for whether
      the contributor is dead. This column must only contain the values 0
      meaning the contributor is not dead and 1 meaning that the contributor is dead.
    * `h_value_col`: The name of the column containing the h value for the strata.
    * `out_of_scope_marker_col`: The name of the column containing a marker for
      whether the contributor is out of scope. This column must only contain
      the values 0 meaning the contributor is not out of scope and 1 meaning
      that the contributor is out of scope.
    * out_of_scope_full: A parameter that specifies what type of out of scope
      to run when an `out_of_scope_marker_col` is provided. True specifies
      that the out of scope is used on both sides of the adjustment fraction.
      False specifies that the out of scope is used only on the denominator of
      the adjustment fraction.
    * auxiliary_col: The name of the column containing the auxiliary value for
      the contributor.
    * calibration_group_col: The name of the column containing the calibration
      group for the contributor.
    * unadjusted_design_weight_col: The name of the column which will contain
      the unadjusted design weight for the contributor. The column isn't
      output unless a name is provided.
    * design_weight_col: The name of the column which will contain the
      design weight for the contributor. Defaults to `design_weight`.
    * calibration_weight_col: The name of the column which will containthe
      calibration weight for the contributor. Defaults to `calibration_weight`.

    ###Returns

    A data frame containing the estimated weights. The exact columns depend on
    the type of estimation performed as specified below.

    ####Common Columns

    In all cases the data frame will contain:

    * `period_col`
    * `strata_col`
    * `design_weight_col`

    ####Ratio Estimation

    In the case of either Separate or Combined Ratio Estimation, the data frame
    will also contain the column specified by `calibration_weight_col`.

    ####Combined Ratio Estimation

    When Combined Ratio Estimation is performed, the data frame will also
    contain the column specified by `calibration_weight_col`.

    ###Notes

    Either both or neither of `death_marker_col` and `h_value_col` must be specified.
    If they are then the design weight is adjusted using birth-death
    adjustment, otherwise it is not. In addition, since birth-death adjustment
    is per-stratum, the `h_value_col` must not change within a given period and
    stratum.

    If `out_of_scope_marker_col` is specified the `out_of_scope_full`
    parameter must also be set. In addition `death_marker_col` and `h_value_col`
    must be provided.

    If `auxiliary_col` is specified then one of Separate Ratio or Combined Ratio
    estimation is performed. This depends on whether `calibration_group_col`
    is specified. If so then Combined Ratio estimation is performed, otherwise
    Separate Ratio estimation is performed. If `auxiliary_col` is not
    specified then only Expansion estimation is performed and specifying
    `calibration_group_col` raises an error.

    `unadjusted_design_weight_col` is only in the output if a column name is specified.

    All specified input columns must be fully populated. If not an error is
    raised. Since `design_weight_col` and `calibration_weight_col` are both
    output columns this does not apply to them, and any values they contain prior
    to calling the method will be ignored.
    """

    # --- Validate params ---
    if not isinstance(input_df, DataFrame):
        raise TypeError("input_df must be an instance of pyspark.sql.DataFrame")

    death_cols = (death_marker_col, h_value_col)
    if any(death_cols) and not all(death_cols):
        raise TypeError(
            "Either both or none of death_marker_col and h_value_col must be specified."
        )

    # Not the same as death_cols because when out_of_scope_full is false the
    # all fails.
    out_of_scope_cols = (out_of_scope_marker_col, out_of_scope_full)
    if out_of_scope_cols.count(None) == 1:
        raise TypeError(
            "Either both or none of out_of_scope_marker_col "
            + "and out_of_scope_full must be specified."
        )
    if any(out_of_scope_cols) and not any(death_cols):
        raise TypeError(
            "For out of scope, death_marker_col and h_value_col must be specified."
        )

    if calibration_group_col is not None and auxiliary_col is None:
        raise TypeError(
            "If calibration_group_col is specified then auxiliary_col must be provided."
        )

    expected_cols = [
        unique_identifier_col,
        period_col,
        strata_col,
        sample_marker_col,
    ]
    if death_marker_col is not None:
        expected_cols += [death_marker_col, h_value_col]

    if out_of_scope_marker_col is not None:
        expected_cols.append(out_of_scope_marker_col)

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

    duplicate_check = input_df.select(unique_identifier_col, period_col)
    if duplicate_check.distinct().count() != duplicate_check.count():
        raise ValidationError("Duplicate contributors in a period")

    # As per the documentation, death marker and sample marker columns must
    # only contain 0 or 1.
    marker_cols = [sample_marker_col]
    if death_marker_col is not None:
        marker_cols.append(death_marker_col)
    if out_of_scope_marker_col is not None:
        marker_cols.append(out_of_scope_marker_col)

    for col_name in marker_cols:
        if input_df.filter((col(col_name) != 0) & (col(col_name) != 1)).count() > 0:
            raise ValidationError(
                f"Input column {col_name} must only contain values of 0 or 1."
            )

    # h values must not change within a stratum
    if h_value_col is not None and (
        input_df.select(period_col, strata_col).distinct().count()
        != input_df.select(period_col, strata_col, h_value_col).distinct().count()
    ):
        raise ValidationError("The h value must be the same per period and stratum.")

    # death(death_marker=1) count must be less than sample(sample_marker=1)
    if (
        death_marker_col is not None
        and (
            input_df.groupBy([period_col, strata_col])
            .agg(sum(col(death_marker_col)), sum(col(sample_marker_col)))
            .fillna(0, ["sum(sample_inclusion_marker)"])
            .filter(col("sum(death_marker)") > col("sum(sample_inclusion_marker)"))
            .count()
        )
        >= 1
    ):
        raise ValidationError("The death count must be less than sample count.")

    # --- prepare our working data frame ---
    col_list = [
        col(period_col).alias("period"),
        col(strata_col).alias("strata"),
        col(sample_marker_col).alias("sample_marker"),
    ]

    if death_marker_col is not None and h_value_col is not None:
        col_list += [
            col(death_marker_col).alias("death_marker"),
            col(h_value_col).alias("h_value"),
        ]

    else:
        col_list += [lit(0).alias("death_marker"), lit(0.0).alias("h_value")]

    if out_of_scope_marker_col is not None:
        col_list.append(
            col(out_of_scope_marker_col).alias("out_of_scope_marker_denominator")
        )
        if out_of_scope_full:
            col_list.append(
                col(out_of_scope_marker_col).alias("out_of_scope_marker_numerator")
            )
        else:
            col_list.append(lit(0).alias("out_of_scope_marker_numerator"))
    else:
        col_list.append(lit(0).alias("out_of_scope_marker_numerator"))
        col_list.append(lit(0).alias("out_of_scope_marker_denominator"))

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
            first(col("h_value").cast("integer")).alias("first(h_value)"),
            sum(col("out_of_scope_marker_numerator")),
            sum(col("out_of_scope_marker_denominator")),
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
                        * (
                            col("sum(death_marker)")
                            + col("sum(out_of_scope_marker_numerator)")
                        )
                        / (
                            col("sum(sample_marker)")
                            - col("sum(death_marker)")
                            - col("sum(out_of_scope_marker_denominator)")
                        )
                    )
                )
            ),
        )
        .drop(
            "sum(sample_marker)",
            "sum(death_marker)",
            "first(h_value)",
            "sum(out_of_scope_marker_numerator)",
            "sum(out_of_scope_marker_denominator)",
            "count(sample_marker)",
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
                "aux_design",
                col("auxiliary")
                * col("unadjusted_design_weight")
                * col("sample_marker"),
            )
            .groupBy(group_cols)
            .agg({"auxiliary": "sum", "aux_design": "sum"})
            .withColumn(
                "calibration_weight", col("sum(auxiliary)") / col("sum(aux_design)")
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
            col("calibration_weight").alias(calibration_weight_col),
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
