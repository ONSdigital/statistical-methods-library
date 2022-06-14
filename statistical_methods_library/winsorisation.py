"""
This module performs Winsorisation.

Currently only One-sided Winsorisation is implemented.
"""

import typing
from enum import Enum

from pyspark.sql import DataFrame
from pyspark.sql.functions import col, expr, lit, when


class ValidationError(Exception):
    """Error raised when data frame validation fails."""

    pass


class Marker(Enum):
    """Values for the marker column created during winsorisation."""

    WINSORISED = "W"
    """The value has been winsorised."""

    FULLY_ENUMERATED = "NW_FE"
    """The value has not been winsorised because it makes up 100% of its stratum.
    (design_weight = 1 and calibration_weight = 1)"""

    DESIGN_CALIBRATION = "NW_AG"
    """The value has not been winsorised because design * calibration is <= 1."""


def one_sided_winsorise(
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
):
    """
    Perform One-sided Winsorisation.

    ###Arguments
    * input_df: The input data frame.
    * reference_col: The name of the column to reference a unique contributor.
    * period_col: The name of the column containing the period information for
      a contributor.
    * grouping_col: The name of the column containing the grouping information
      for a contributor.
    * target_col: The name of the column containing the target variable.
    * design_col: The name of the column containing the design weight.
    * l_value_col: The name of the column containing the l value.
    * outlier_col: The name of the column which will contain the calculated
      outlier weight. Defaults to `outlier_weight`.
    * calibration_col: The name of the column containing the calibration weight
      if Ratio Winsorisation is to be performed.
    * auxiliary_col: The name of the column containing the auxiliary values if
      Ratio Winsorisation is to be performed.
    * marker_col: The name of the column which will contain the
      marker for winsorisation. Defaults to `winsorisation_marker`.

    ###Returns
    A new data frame containing:

    * `reference_col`
    * `period_col`
    * `outlier_col`
    * `marker_col`

    ###Notes

    All of the provided columns containing input values must be fully
    populated. Otherwise an error is raised.

    If a stratum contains multiple l-values in the same period then an error
    will be raised.

    Both or neither of `calibration_col` and `auxiliary_col` must be
    specified. Specifying one without the other raises an error.

    If these columns are specified then they are used in calculations so
    Ratio Winsorisation is performed. If not then Expansion Winsorisation is
    performed.

    `marker_col` will contain one of the marker constants defined in the
    `Marker` enum.
    """

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

    # If we don't have a calibration weight and auxiliary value then set to 1.
    # This cancels out the ratio part of Winsorisation which means that
    # Expansion Winsorisation is performed.
    if auxiliary_col is not None:
        col_list.append(col(auxiliary_col).alias("auxiliary"))

    else:
        col_list.append(lit(1.0).alias("auxiliary"))

    if calibration_col is not None:
        col_list.append(col(calibration_col).alias("calibration"))

    else:
        col_list.append(lit(1.0).alias("calibration"))

    group_cols = ["period", "grouping"]
    pre_marker_df = input_df.select(col_list)

    if (
        input_df.select([col(grouping_col), col(period_col), col(l_value_col)])
        .distinct()
        .groupBy([grouping_col, period_col])
        .count()
        .filter("count > 1")
        .count()
        > 0
    ):
        raise ValidationError("Differing L Values within a grouping in the same period")

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
        "outlier", lit(1.0)
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
            col("outlier").alias(outlier_col),
            col("marker").alias(marker_col),
        )
    )
