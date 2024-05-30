"""
Perform link-based imputation on a data frame.

This module provides the engine and other core aspects of imputation with
ratio calculation being handled by provided callables.

For Copyright information, please see LICENCE.
"""
from decimal import Decimal
from enum import Enum
from functools import reduce
from typing import Optional, Union

from pyspark.sql import Column, DataFrame
from pyspark.sql.functions import col, expr, first, lit, when
from pyspark.sql.types import DecimalType, StringType

from statistical_methods_library.utilities.periods import (
    calculate_next_period,
    calculate_previous_period,
)
from statistical_methods_library.utilities.validation import validate_dataframe

from .ratio_calculators import RatioCalculator, ratio_of_means_construction


class Marker(Enum):
    """Values for the marker column created during imputation."""

    RESPONSE = "R"
    """The value is a response."""

    FORWARD_IMPUTE_FROM_RESPONSE = "FIR"
    """The value has been forward imputed from a response."""

    BACKWARD_IMPUTE = "BI"
    """The value has been backward imputed from a response,
    backward imputation from construction is not permitted."""

    CONSTRUCTED = "C"
    """The value is constructed."""

    FORWARD_IMPUTE_FROM_CONSTRUCTION = "FIC"
    """The value has been forward imputed from a constructed value."""

    MANUAL_CONSTRUCTION = "MC"
    """The value is manual construction."""

    FORWARD_IMPUTE_FROM_MANUAL_CONSTRUCTION = "FIMC"
    """The value has been forward imputed from a manual construction."""


def impute(
    *,
    input_df: DataFrame,
    reference_col: str,
    period_col: str,
    grouping_col: str,
    target_col: str,
    auxiliary_col: str,
    forward_backward_ratio_calculator: RatioCalculator,
    construction_ratio_calculator: Optional[
        RatioCalculator
    ] = ratio_of_means_construction,
    output_col: Optional[str] = "imputed",
    marker_col: Optional[str] = "imputation_marker",
    forward_link_col: Optional[str] = "forward",
    backward_link_col: Optional[str] = "backward",
    construction_link_col: Optional[str] = "construction",
    count_construction_col: Optional[str] = "count_construction",
    count_forward_col: Optional[str] = "count_forward",
    count_backward_col: Optional[str] = "count_backward",
    default_construction_col: Optional[str] = "default_construction",
    default_forward_col: Optional[str] = "default_forward",
    default_backward_col: Optional[str] = "default_backward",
    link_inclusion_current_col: Optional[str] = "link_inclusion_current",
    link_inclusion_previous_col: Optional[str] = "link_inclusion_previous",
    link_inclusion_next_col: Optional[str] = "link_inclusion_next",
    back_data_df: Optional[DataFrame] = None,
    link_filter: Optional[Union[str, Column]] = None,
    periodicity: Optional[int] = 1,
    weight: Optional[Decimal] = None,
    weight_periodicity_multiplier: Optional[int] = None,
    unweighted_forward_link_col: Optional[str] = "forward_unweighted",
    unweighted_backward_link_col: Optional[str] = "backward_unweighted",
    unweighted_construction_link_col: Optional[str] = "construction_unweighted",
    manual_construction_col: Optional[str] = None,
    **ratio_calculator_params,
) -> DataFrame:
    """
    Impute a target variable using imputation links (a.k.a. ratios).

    Args:
        input_df: The input data frame.
        reference_col: The name of the column containing the identifier for a
          contributor. Must be unique within a period and grouping.
        period_col: The name of the column containing the period.
        grouping_col: The name of the column containing the imputation
          grouping for a given record..
        target_col: The name of the column containing the target variable.
        auxiliary_col: The name of the column containing the auxiliary variable.
        forward_backward_ratio_calculator: Used to calculate the ratios for
            forward and backward imputation. See the
          `imputation.ratio_calculators` module for more details.
        construction_ratio_calculator: Used to calculate the ratio for
          construction imputation. See the `imputation.ratio_calculators`
          module for more details.
        output_col: The name of the column containing the imputed variable.
        marker_col: The name of the column containing the imputation marker.
        forward_link_col: The name of the column containing the forward
          imputation link.
        backward_link_col: The name of the column containing the backward
          imputation link.
        construction_link_col: The name of the column containing the
          construction imputation link.
        count_construction_col: The name of the column containing the count of
          matched pairs used in construction link calculations.
        count_forward_col: The name of the column containing the count of
          matched pairs used inforward link calculations.
        count_backward_col: The name of the column containing the count of
          matched pairs used in backward link calculations.
        default_construction_col: The name of the column containing the default
          marker for the construction link.
        default_forward_col: The name of the column containing the default
          marker for the forward link.
        default_backward_col: The name of the column containing the default
          marker for the backward link.
        link_inclusion_current_col: The name of the column marking whether the
          value was included in link calculations based on filtering for the
          current period.
        link_inclusion_previous_col: The name of the column marking whether the
          value was included in link calculations based on filtering for the
          previous period.
        link_inclusion_next_col: The name of the column marking whether the
          value was included in link calculations based on filtering for the
          next period.
        back_data_df: The back data data frame.
        link_filter: An inclusive filter that specifies whether a response
          can be used for ratio calculation. The link inclusion marker
          columns will only be present in the output if this is provided.
          This filter is either a boolean column expression or a spark
          sql string which yields a boolean result.
        periodicity: The periodicity of the data as used by the calculation
          functions in the `utilities.periods` module.
        unweighted_forward_link_col: The name of the column containing the
          forward link prior to weighting.
        unweighted_backward_link_col: The name of the column containing the
          backward link prior to weighting.
        unweighted_construction_link_col: The name of the column containing the
          construction link prior to weighting.
        weight: A decimal value between 0 and 1 inclusive used to weigh links
          in the current period against those in a previous period. The current
          link is multiplied by the weight whereas the corresponding previous
          link is multiplied by (1 - weight). If no corresponding previous link
          can be found the current link is left unchanged. Link weighting is
          only performed and the unweighted link columns above are only present
          in the output if this value is provided.
        weight_periodicity_multiplier: Multiplied by the periodicity of the
          dataset to calculate the previous period when finding the previous
          links for weighting.
        manual_construction_col: The name of the column containing the
          construction value.
        ratio_calculator_params: Any extra keyword arguments to the engine are
          passed to the specified ratio calculators as keyword args and are
          otherwise ignored by this function. Please see the specified ratio
          calculator callables for details.

    Returns:
    A data frame containing the imputed variable and links. The exact columns
    depend on the provided arguments. The data frame contains a row for each
    reference, period and grouping combination in the input data.

    Either both or neither of `weight` and `weight_periodicity_multiplier`
    must be specified.

    Either both or neither of `forward_link_col` and `backward_link_col` must
    be in the input.

    If `forward_link_col` and `backward_link_col` are present in the input
    then they will not be calculated or weighted. The same also applies to
    `construction_link_col`. This also means that the corresponding unweighted
    columns will not be present in the output even if the arguments for
    weighting are specified. In addition the corresponding count columns will
    be set to `0` in this case.

    If `link_filter` is provided then the inclusion marker columns will be
    present in the output otherwise they will not be.

    Ratio calculators may also provide additional output columns.
    Please see the specified ratio calculators and the
    `imputation.ratio_calculators` module for more details.
    """
    # --- Validate params ---
    if not isinstance(input_df, DataFrame):
        raise TypeError("Input is not a DataFrame")

    link_cols = [forward_link_col, backward_link_col]
    if any(link_cols) and not all(link_cols):
        raise TypeError("Either all or no link columns must be specified")
    input_params = {
        "ref": reference_col,
        "period": period_col,
        "grouping": grouping_col,
        "target": target_col,
        "aux": auxiliary_col,
    }

    # Mapping of column aliases to parameters
    output_col_mapping = {
        "output": output_col,
        "marker": marker_col,
        "count_construction": count_construction_col,
        "count_forward": count_forward_col,
        "count_backward": count_backward_col,
        "default_construction": default_construction_col,
        "default_forward": default_forward_col,
        "default_backward": default_backward_col,
        "forward": forward_link_col,
        "backward": backward_link_col,
        "construction": construction_link_col,
        "ref": reference_col,
        "period": period_col,
        "grouping": grouping_col,
    }

    if forward_link_col in input_df.columns or backward_link_col in input_df.columns:
        input_params.update(
            {
                "forward": forward_link_col,
                "backward": backward_link_col,
            }
        )

    if construction_link_col in input_df.columns:
        input_params["construction"] = construction_link_col

    back_input_params = {
        "ref": reference_col,
        "period": period_col,
        "grouping": grouping_col,
        "output": output_col,
        "marker": marker_col,
    }
    # Add manual_construction parm
    # only if manual_construction_col is not None.
    if manual_construction_col:
        input_params["manual_const"] = manual_construction_col

    if back_data_df:
        if not isinstance(back_data_df, DataFrame):
            raise TypeError("Input is not a DataFrame")

    if weight is not None:
        if not isinstance(weight, Decimal):
            raise TypeError("weight must be of type Decimal")

        weight = lit(weight)
        weight_periodicity = weight_periodicity_multiplier * periodicity
        weight_col_mapping = {
            "forward_unweighted": unweighted_forward_link_col,
            "backward_unweighted": unweighted_backward_link_col,
            "construction_unweighted": unweighted_construction_link_col,
        }

        for name in "forward", "backward", "construction":
            if name in input_params:
                del weight_col_mapping[f"{name}_unweighted"]

        back_input_params.update(weight_col_mapping)

        output_col_mapping.update(weight_col_mapping)

    type_mapping = {
        "period": StringType,
        "target": DecimalType,
        "aux": DecimalType,
        "output": DecimalType,
        "marker": StringType,
        "forward": DecimalType,
        "backward": DecimalType,
        "construction": DecimalType,
        "forward_unweighted": DecimalType,
        "backward_unweighted": DecimalType,
        "construction_unweighted": DecimalType,
        "manual_const": DecimalType,
    }

    if link_filter:
        if back_data_df:
            filtered_refs = input_df.unionByName(back_data_df, allowMissingColumns=True)
        else:
            filtered_refs = input_df

        filtered_refs = filtered_refs.select(
            col(reference_col).alias("ref"),
            col(period_col).alias("period"),
            col(grouping_col).alias("grouping"),
            (expr(link_filter) if isinstance(link_filter, str) else link_filter).alias(
                "match"
            ),
        ).localCheckpoint(eager=False)

    prepared_df = (
        validate_dataframe(
            input_df,
            input_params,
            type_mapping,
            ["ref", "period", "grouping"],
            ["target", "manual_const"],
        )
        .withColumnRenamed("target", "output")
        .withColumn("marker", when(~col("output").isNull(), Marker.RESPONSE.value))
        .withColumn(
            "previous_period", calculate_previous_period(col("period"), periodicity)
        )
        .withColumn("next_period", calculate_next_period(col("period"), periodicity))
    )
    prior_period_df = prepared_df.selectExpr(
        "min(previous_period) AS prior_period"
    ).localCheckpoint(eager=False)

    if back_data_df:
        validated_back_data_df = validate_dataframe(
            back_data_df, back_input_params, type_mapping, ["ref", "period", "grouping"]
        ).localCheckpoint(eager=False)
        back_data_period_df = (
            validated_back_data_df.select(
                "ref", "period", "grouping", "output", "marker"
            )
            .join(prior_period_df, [col("period") == col("prior_period")])
            .drop("prior_period")
            .filter(((col(marker_col) != lit(Marker.BACKWARD_IMPUTE.value))))
            .withColumn(
                "previous_period",
                calculate_previous_period(col("period"), periodicity),
            )
            .withColumn(
                "next_period", calculate_next_period(col("period"), periodicity)
            )
            .localCheckpoint(eager=False)
        )

        prepared_df = prepared_df.unionByName(
            back_data_period_df.filter(col("marker") == lit(Marker.RESPONSE.value)),
            allowMissingColumns=True,
        )

    def calculate_ratios():
        # This allows us to return early if we have nothing to do
        nonlocal prepared_df

        ratio_calculators = []
        if "forward" in prepared_df.columns:
            prepared_df = (
                prepared_df.withColumn("default_forward", expr("forward IS NULL"))
                .withColumn("default_backward", expr("backward IS NULL"))
                .withColumn("count_forward", lit(0).cast("long"))
                .withColumn("count_backward", lit(0).cast("long"))
            )

        else:
            ratio_calculators.append(forward_backward_ratio_calculator)

        if "construction" in prepared_df.columns:
            prepared_df = prepared_df.withColumn(
                "default_construction", expr("construction IS NULL")
            ).withColumn("count_construction", lit(0).cast("long"))

        else:
            ratio_calculators.append(construction_ratio_calculator)

        if not ratio_calculators:
            return

        # Since we're going to join on to the main df filtering here
        # won't cause us to lose grouping as they'll just be filled with
        # default ratios.
        if link_filter:
            ratio_filter_df = prepared_df.join(
                filtered_refs, ["ref", "period", "grouping"]
            )
        else:
            ratio_filter_df = prepared_df.withColumn("match", lit(True))

        ratio_filter_df = ratio_filter_df.filter("output IS NOT NULL").select(
            "ref",
            "period",
            "grouping",
            "output",
            "aux",
            "previous_period",
            "next_period",
            "match",
        )

        # Put the values from the current and previous periods for a
        # contributor on the same row.
        ratio_calculation_df = (
            ratio_filter_df.join(
                ratio_filter_df.selectExpr(
                    "ref",
                    "period AS previous_period",
                    "output AS previous_output",
                    "grouping",
                    "match AS link_inclusion_previous",
                ),
                ["ref", "grouping", "previous_period"],
                "leftouter",
            )
            .join(
                ratio_filter_df.selectExpr(
                    "ref",
                    "period AS next_period",
                    "output AS next_output",
                    "grouping",
                    "match AS link_inclusion_next",
                ),
                ["ref", "next_period", "grouping"],
                "leftouter",
            )
            .selectExpr(
                "ref",
                "grouping",
                "period",
                "aux",
                "output",
                "match AS link_inclusion_current",
                "next_output",
                "link_inclusion_next",
                "previous_output",
                "link_inclusion_previous",
            )
        )

        # Join the grouping ratios onto the input such that each contributor has
        # a set of ratios.
        fill_values = {}
        for result in sum(
            (
                calculator(df=ratio_calculation_df, **ratio_calculator_params)
                for calculator in ratio_calculators
            ),
            [],
        ):
            prepared_df = prepared_df.join(result.data, result.join_columns, "left")
            fill_values.update(result.fill_values)
            output_col_mapping.update(result.additional_outputs)

        prepared_df = prepared_df.fillna(fill_values)

        if link_filter:
            prepared_df = prepared_df.join(
                ratio_calculation_df.select(
                    "ref",
                    "period",
                    "grouping",
                    "link_inclusion_previous",
                    "link_inclusion_current",
                    "link_inclusion_next",
                ),
                ["ref", "period", "grouping"],
                "left",
            )
            output_col_mapping.update(
                {
                    "link_inclusion_current": link_inclusion_current_col,
                    "link_inclusion_previous": link_inclusion_previous_col,
                    "link_inclusion_next": link_inclusion_next_col,
                }
            )

        if weight is not None:

            def calculate_weighted_link(link_name):
                prev_link = col(f"prev.{link_name}")
                curr_link = col(f"curr.{link_name}")
                return (
                    when(
                        prev_link.isNotNull(),
                        weight * curr_link + (lit(Decimal(1)) - weight) * prev_link,
                    )
                    .otherwise(curr_link)
                    .alias(link_name)
                )

            weight_col_names = [
                name
                for name in ("forward", "backward", "construction")
                if name not in input_params
            ]

            if not weight_col_names:
                return

            weighting_df = (
                prepared_df.join(prior_period_df, (col("prior_period") < col("period")))
                .select(
                    "period",
                    "grouping",
                    *(
                        col(name).alias(f"{name}_unweighted")
                        for name in weight_col_names
                    ),
                )
                .unionByName(
                    validated_back_data_df.select(
                        "period",
                        "grouping",
                        *(f"{name}_unweighted" for name in weight_col_names),
                    )
                )
                .groupBy("period", "grouping")
                .agg(
                    *(
                        first(f"{name}_unweighted").alias(name)
                        for name in weight_col_names
                    )
                )
            )

            curr_df = weighting_df.alias("curr")
            prev_df = weighting_df.alias("prev")
            prepared_df = (
                curr_df.join(
                    prev_df,
                    (
                        (
                            col("prev.period")
                            == calculate_previous_period(
                                col("curr.period"), weight_periodicity
                            )
                        )
                        & (col("curr.grouping") == col("prev.grouping"))
                    ),
                    "left",
                )
                .select(
                    expr("curr.period AS period"),
                    expr("curr.grouping AS grouping"),
                    *(calculate_weighted_link(name) for name in weight_col_names),
                )
                .join(
                    reduce(
                        lambda d, n: d.withColumnRenamed(n, f"{n}_unweighted"),
                        weight_col_names,
                        prepared_df,
                    ),
                    ["period", "grouping"],
                )
            )

    calculate_ratios()

    # Caching for both imputed and unimputed data.
    imputed_df = None
    null_response_df = None

    # --- Impute helper ---
    def impute_helper(
        df: DataFrame, link_col: str, marker: Marker, direction: bool
    ) -> DataFrame:
        nonlocal imputed_df
        nonlocal null_response_df
        if direction:
            # Forward imputation
            other_period_col = "previous_period"
        else:
            # Backward imputation
            other_period_col = "next_period"

        if imputed_df is None:
            working_df = df.select(
                "ref",
                "period",
                "grouping",
                "output",
                "marker",
                "previous_period",
                "next_period",
                "forward",
                "backward",
            )
            # Anything which isn't null is already imputed or a response and thus
            # can be imputed from. Note that in the case of backward imputation
            # this still holds since it always happens after forward imputation
            # and thus it can never attempt to backward impute from a forward
            # imputation since there will never be a null value directly prior to
            # one.
            imputed_df = working_df.filter(~col("output").isNull()).localCheckpoint(
                eager=True
            )
            # Any ref and grouping combos which have no values at all can't be
            # imputed from so we don't care about them here.
            ref_df = imputed_df.select("ref", "grouping").distinct()
            null_response_df = (
                working_df.filter(col("output").isNull())
                .drop("output", "marker")
                .join(ref_df, ["ref", "grouping"])
                .localCheckpoint(eager=True)
            )

        while True:
            other_df = imputed_df.selectExpr(
                "ref AS other_ref",
                "period AS other_period",
                "output AS other_output",
                "grouping AS other_grouping",
            )
            calculation_df = (
                null_response_df.join(
                    other_df,
                    [
                        col(other_period_col) == col("other_period"),
                        col("ref") == col("other_ref"),
                        col("grouping") == col("other_grouping"),
                    ],
                )
                .select(
                    "ref",
                    "period",
                    "grouping",
                    (col(link_col) * col("other_output")).alias("output"),
                    lit(marker.value).alias("marker"),
                    "previous_period",
                    "next_period",
                    "forward",
                    "backward",
                )
                .localCheckpoint(eager=False)
            )
            # If we've imputed nothing then we've got as far as we can get for
            # this phase.
            if calculation_df.count() == 0:
                break

            # Store this set of imputed values in our main set for the next
            # iteration. Use eager checkpoints to help prevent rdd DAG explosion.
            imputed_df = imputed_df.union(calculation_df).localCheckpoint(eager=True)
            # Remove the newly imputed rows from our filtered set.
            null_response_df = null_response_df.join(
                calculation_df.select("ref", "period", "grouping"),
                ["ref", "period", "grouping"],
                "leftanti",
            ).localCheckpoint(eager=True)
        # We should now have an output column which is as fully populated as
        # this phase of imputation can manage. As such replace the existing
        # output column with our one. Same goes for the marker column.
        return df.drop("output", "marker").join(
            imputed_df.select("ref", "period", "grouping", "output", "marker"),
            ["ref", "period", "grouping"],
            "leftouter",
        )

    # --- Imputation functions ---
    def forward_impute_from_response(df: DataFrame) -> DataFrame:
        if back_data_df:
            # Add the forward imputes from responses from the back data
            df = df.unionByName(
                back_data_period_df.filter(
                    col("marker") == lit(Marker.FORWARD_IMPUTE_FROM_RESPONSE.value)
                ),
                allowMissingColumns=True,
            )
        return impute_helper(df, "forward", Marker.FORWARD_IMPUTE_FROM_RESPONSE, True)

    def backward_impute(df: DataFrame) -> DataFrame:
        return impute_helper(df, "backward", Marker.BACKWARD_IMPUTE, False)

    # --- Forward impute from manual construction ---
    def forward_impute_from_manual_construction(df: DataFrame) -> DataFrame:
        nonlocal imputed_df
        nonlocal null_response_df
        imputed_df = None
        null_response_df = None
        if back_data_df:
            # Add the MC and FIMC from the back data
            df = df.unionByName(
                back_data_period_df.filter(
                    (col("marker") == lit(Marker.MANUAL_CONSTRUCTION.value))
                    | (
                        col("marker")
                        == lit(Marker.FORWARD_IMPUTE_FROM_MANUAL_CONSTRUCTION.value)
                    )
                ),
                allowMissingColumns=True,
            )

        return impute_helper(
            df, "forward", Marker.FORWARD_IMPUTE_FROM_MANUAL_CONSTRUCTION, True
        )

    # --- Construction functions ---
    def construct_values(df: DataFrame) -> DataFrame:
        if back_data_df:
            df = df.unionByName(
                back_data_period_df.filter(
                    (
                        (col("marker") == lit(Marker.CONSTRUCTED.value))
                        | (
                            col("marker")
                            == lit(Marker.FORWARD_IMPUTE_FROM_CONSTRUCTION.value)
                        )
                    )
                ),
                allowMissingColumns=True,
            )

        construction_df = df.filter(df.output.isNull()).select(
            "ref", "period", "grouping", "aux", "construction", "previous_period"
        )
        other_df = df.select("ref", "period", "grouping").alias("other")
        construction_df = construction_df.alias("construction")
        construction_df = construction_df.join(
            other_df,
            [
                col("construction.ref") == col("other.ref"),
                col("construction.previous_period") == col("other.period"),
                col("construction.grouping") == col("other.grouping"),
            ],
            "leftanti",
        ).select(
            col("construction.ref").alias("ref"),
            col("construction.period").alias("period"),
            col("construction.grouping").alias("grouping"),
            (col("aux") * col("construction")).alias("constructed_output"),
            lit(Marker.CONSTRUCTED.value).alias("constructed_marker"),
        )

        return (
            df.withColumnRenamed("output", "existing_output")
            .withColumnRenamed("marker", "existing_marker")
            .join(
                construction_df,
                ["ref", "period", "grouping"],
                "leftouter",
            )
            .select(
                "*",
                when(col("existing_output").isNull(), col("constructed_output"))
                .otherwise(col("existing_output"))
                .alias("output"),
                when(col("existing_marker").isNull(), col("constructed_marker"))
                .otherwise(col("existing_marker"))
                .alias("marker"),
            )
            .drop("existing_output", "constructed_output", "constructed_marker")
        )

    def forward_impute_from_construction(df: DataFrame) -> DataFrame:
        # We need to recalculate our imputed and null response data frames to
        # account for construction.
        nonlocal imputed_df
        nonlocal null_response_df
        imputed_df = None
        null_response_df = None
        return impute_helper(
            df, "forward", Marker.FORWARD_IMPUTE_FROM_CONSTRUCTION, True
        )

    if manual_construction_col:
        # Set manual construction value as output
        # and marker as MC
        mc_df = prepared_df.withColumn(
            "marker",
            when(
                (col("manual_const").isNotNull()) & (col("output").isNull()),
                lit(Marker.MANUAL_CONSTRUCTION.value),
            ).otherwise(col("marker")),
        ).withColumn(
            "output",
            when(
                (col("manual_const").isNotNull()) & (col("output").isNull()),
                col("manual_const"),
            ).otherwise(col("output")),
        )

        # Filter out MC data, which leaves a gap in the imputation pattern for
        # identifiers with MC values. As a result, it prevents the FIR from
        # being issued against the targeted FIMC. This MC data will be merged with
        # the main df prior to the forward_impute_from_manual_construction stage.
        manual_construction_df = mc_df.filter(
            (col("marker") == Marker.MANUAL_CONSTRUCTION.value)
        )
        prepared_df = mc_df.filter(
            col("marker").isNull()
            | (~(col("marker") == Marker.MANUAL_CONSTRUCTION.value))
        )

    df = prepared_df
    for stage in (
        forward_impute_from_response,
        backward_impute,
        forward_impute_from_manual_construction,
        construct_values,
        forward_impute_from_construction,
    ):
        if manual_construction_col and stage == forward_impute_from_manual_construction:
            # Add the mc data
            df = df.unionByName(manual_construction_df, allowMissingColumns=True)

        df = stage(df).localCheckpoint(eager=False)

        if df.filter(col("output").isNull()).count() == 0:
            if (not manual_construction_col) or (
                manual_construction_col and stage == construct_values
            ):
                break
    return df.join(prior_period_df, [col("prior_period") < col("period")]).select(
        [
            col(k).alias(output_col_mapping[k])
            for k in sorted(output_col_mapping.keys() & set(df.columns))
        ]
    )
