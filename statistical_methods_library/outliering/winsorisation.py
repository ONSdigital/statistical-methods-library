"""
Perform Winsorisation on a data frame.

Currently only One-sided Winsorisation is implemented.
For Copyright information, please see LICENCE.
"""

import typing
from enum import Enum

from pyspark.sql import DataFrame
from pyspark.sql.functions import col, expr, lit, when
from pyspark.sql.types import DecimalType

from statistical_methods_library.utilities import validation


class Marker(Enum):
    """Values for the marker column created during winsorisation."""

    WINSORISED = "W"
    """The value has been winsorised."""

    FULLY_ENUMERATED = "NW_FE"
    """The value has not been winsorised because it makes up 100% of its stratum.
    (design_weight = 1 and calibration_factor = 1)"""

    DESIGN_CALIBRATION = "NW_AG"
    """The value has not been winsorised because design * calibration is <= 1."""


def outlier(
    input_df: DataFrame,
    reference_col: str,
    period_col: str,
    grouping_col: str,
    target_col: str,
    design_col: str,
    l_value_col: str,
    outlier_col: typing.Optional[str] = "outlier_weight",
    calibration_col: typing.Optional[str] = None,
    auxiliary_col: typing.Optional[str] = None,
    marker_col: typing.Optional[str] = "winsorisation_marker",
) -> DataFrame:
    """
    Perform One-sided Winsorisation.

    Args:
        input_df: The input data frame.
        reference_col: The name of the column to reference a unique contributor.
        period_col: The name of the column identifying the period. Values in
          this column are merely for grouping purposes and thus do not need
          to conform to the conventions or data type documented for periods
          in the `utilities.periods` module.
        grouping_col: The name of the column containing the grouping information
          for a contributor.
        target_col: The name of the column containing the target variable.
        design_col: The name of the column containing the design weight.
        l_value_col: The name of the column containing the l value.
        outlier_col: The name of the column which will contain the calculated
          outlier weight.
        calibration_col: The name of the column containing the calibration factor
          if Ratio Winsorisation is to be performed.
        auxiliary_col: The name of the column containing the auxiliary values if
          Ratio Winsorisation is to be performed.
        marker_col: The name of the column which will contain the
          marker for winsorisation (one of the values defined by the `Marker`
          class).

    Returns:
    A new data frame containing the columns as described by the arguments
    `reference_col`, `period_col`, `grouping_col`, `outlier_col` and `marker_col`.

    The provided columns containing input values must be fully
    populated. In addition there must only be one l-value per stratum and
    period.

    Both or neither of `calibration_col` and `auxiliary_col` must be
    specified. If these columns are specified then they are used in
    calculations so Ratio Winsorisation is performed. If not then Expansion
    Winsorisation is performed.
    """

    ratio_cols = [auxiliary_col, calibration_col]
    if any(ratio_cols) and not all(ratio_cols):
        raise TypeError(
            "Both or neither of auxiliary_col and calibration_col must be specified."
        )

    # Validate params
    input_params = {
        "reference": reference_col,
        "period": period_col,
        "grouping": grouping_col,
        "target": target_col,
        "design": design_col,
    }

    optional_params = {
        "l_value": l_value_col,
        "calibration": calibration_col,
        "auxiliary": auxiliary_col,
    }

    input_params.update({k: v for k, v in optional_params.items() if v is not None})

    type_mapping = {
        "target": DecimalType,
        "design": DecimalType,
        "l_value": DecimalType,
        "calibration": DecimalType,
        "auxiliary": DecimalType,
    }

    aliased_df = validation.validate_dataframe(
        input_df, input_params, type_mapping, ["reference", "period"]
    )
    validation.validate_one_value_per_group(
        input_df, [period_col, grouping_col], l_value_col
    )
    validation.validate_no_matching_rows(
        input_df,
        (col(design_col) < 1),
        f"Column {design_col} must not contain values smaller than one.",
    )
    validation.validate_no_matching_rows(
        input_df,
        (col(l_value_col) < 0),
        f"Column {l_value_col} must not contain negative values.",
    )
    if calibration_col is not None:
        validation.validate_no_matching_rows(
            input_df,
            (col(calibration_col) <= 0),
            f"Column {calibration_col} must not contain zero or negative values.",
        )

    # If we don't have a calibration factor and auxiliary value then set to 1.
    # This cancels out the ratio part of Winsorisation which means that
    # Expansion Winsorisation is performed.
    pre_marker_df = aliased_df
    if auxiliary_col is None:
        pre_marker_df = pre_marker_df.withColumn(
            "auxiliary", lit(1).cast(DecimalType())
        )

    if calibration_col is None:
        pre_marker_df = pre_marker_df.withColumn(
            "calibration", lit(1).cast(DecimalType())
        )

    group_cols = ["period", "grouping"]
    # Separate out rows that are not to be winsorised and mark appropriately.
    df = pre_marker_df.withColumn(
        "design_calibration", expr("design * calibration")
    ).withColumn(
        "marker",
        when(
            (col("design") == 1) & (col("calibration") >= 1),
            lit(Marker.FULLY_ENUMERATED.value),
        ).otherwise(
            when(col("design_calibration") <= 1, lit(Marker.DESIGN_CALIBRATION.value))
        ),
    )

    not_winsorised_df = df.filter(col("marker").isNotNull()).withColumn(
        "outlier", lit(1).cast(DecimalType())
    )
    to_be_winsorised_df = df.filter(col("marker").isNull())

    # The design ratio needs to be calculated by grouping whereas the outlier
    # weight calculation is per contributor.
    # If the outlier weight can't be calculated due to the target value being a zero,
    # then default the outlier weight to 1.
    return (
        to_be_winsorised_df.join(
            (
                to_be_winsorised_df.withColumn("target_design", expr("target * design"))
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
            expr("winsorisation_value + (l_value/(design_calibration - 1))"),
        )
        .withColumn(
            "modified_target",
            when(
                (col("target") > col("k_value")),
                expr(
                    """
                        (target/design_calibration)
                        + (k_value - (k_value/design_calibration))
                    """
                ),
            ).otherwise(col("target")),
        )
        .withColumn("outlier", expr("modified_target/target"))
        .fillna(1.0, ["outlier"])
        .withColumn("marker", lit(Marker.WINSORISED.value))
        .drop(
            "target_design",
            "aux_design",
            "ratio_sum_target_sum_aux",
            "sum(target_design)",
            "sum(aux_design)",
            "winsorisation_value",
            "k_value",
            "modified_target",
        )
        .unionByName(not_winsorised_df)
        .select(
            col("reference").alias(reference_col),
            col("period").alias(period_col),
            col("grouping").alias(grouping_col),
            col("outlier").alias(outlier_col),
            col("marker").alias(marker_col),
        )
    )
