# For Copyright information, please see LICENCE.

from pyspark.sql import DataFrame
from pyspark.sql.functions import col, count, sum, when

from . import engine


def impute(**kwargs) -> DataFrame:
    """
    Perform Ratio of means (also known as Ratio of Sums) imputation on a
    dataframe.

    ###Arguments
    * `input_df`: The input dataframe
    * `reference_col`: The name of the column to reference a unique
      contributor.
    * `period_col`: The name of the column containing the period
      information for the contributor.
    * `strata_col`: The Name of the column containing the strata information
      for the contributor.
    * `target_col`: The name of the column containing the target
      variable.
    * `auxiliary_col`: The name of the column containing the auxiliary
      variable.
    * `output_col`: The name of the column which will contain the
      output. Defaults to `imputed`.
    * `marker_col`: The name of the column which will contain the marker
      information for a given value. Defaults to `imputation_marker`.
    * `forward_link_col`: If specified, the name of an existing column
      containing forward ratio (or link) information. Defaults to None which
      means that a default column name of `forward` will be created and the
      forward ratios will be calculated.
    * `backward_link_col`: If specified, the name of an existing column
      containing backward ratio (or link) information. Defaults to None which
      means that a default column name of `backward` will be created and the
      backward ratios will be calculated.
    * `construction_link_col`: If specified, the name of an existing column
      containing construction ratio (or link) information. Defaults to None
      which means that a default column name of `construction` will be created
      and the construction ratios will be calculated.
    * `count_construction_col`: If specified, the name of the column that will
      hold the count of matched pairs for construction.
      Defaults to 'count_construction'
    * `count_forward_col`: If specified, the name of the column that will
      hold the count of matched pairs for forward imputation.
      Defaults to 'count_forward'
    * `count_backward_col`: If specified, the name of the column that will
      hold the count of matched pairs for forward imputation.
      Defaults to 'count_forward'
    * `back_data_df`: If specified, will use this to base the initial imputation
      calculations on.
    * `link_filter`: A filter compatible with the pyspark DataFrame.filter
      method. Only responses that match the filter conditions will be included
      in link calculations.
      This will not prevent non responses from being imputed for.

    ###Returns
    A new dataframe containing:

    * `reference_col`
    * `period_col`
    * `output_col`
    * `marker_col`
    * `forward_col`
    * `backward_col`
    * `construction_col`
    * `construction_count_col`
    * `forward_count_col`
    * `backward_count_col`

    No other columns are created. In particular, no other columns
    will be passed through from the input since it is expected that the
    information in the output dataframe will be sufficient to join on any
    other required input data.

    ###Notes
    If no Imputation needs to take place, the forward, backward and
    construction columns returned will still be calculated if they are not
    passed in.

    The existence of `output_col` and `marker_col` in the input data is
    an error.

    All or none of `forward_link_col`, `backward_link_col` and
    `construction_link_col` must be specified.

    `marker_col` will contain one of the marker constants defined in the
    `Marker` enum.

    This method implements rolling imputation, that is imputed values chain
    together until either a return is present or the contributor is not present
    in the sample. Values either side of such a gap do not interact (i.e.
    ratios and imputes will not take into account values for a contributor
    from before or after periods where they were dropped from the sample). In
    the case of rolling imputation, the markers will be the same for chains of
    imputed values.

    If `back_data_df` is provided it must contain the following columns:
        *`reference_col`
        *`period_col`
        *`strata_col`
        *`auxiliary_col`
        *`output_col`
        *`marker_col`
    """

    def ratio_of_means(df: DataFrame) -> list[engine.RatioCalculationResult]:
        working_df = df.groupBy("period", "strata").agg(
            sum(col("output")),
            sum(col("other_output")),
            sum(col("aux")),
            sum(col("output_for_construction")),
            count(col("output_for_construction")),
            count(col("other_output")),
            count(col("output")),
        )

        # Calculate the forward ratio for every period using 1 as the link in
        # the case of a 0 denominator. We also calculate construction
        # links at the same time for efficiency reasons. This shares
        # the same behaviour of defaulting to a 1 in the case of a 0
        # denominator.

        forward_df = (
            working_df.withColumn(
                "forward", col("sum(output)") / col("sum(other_output)")
            )
            .withColumn(
                "count_forward",
                when(col("sum(other_output)") == 0, 0)
                .when(col("sum(other_output)").isNotNull(), col("count(other_output)"))
                .cast("long"),
            )
            .withColumn(
                "construction", col("sum(output_for_construction)") / col("sum(aux)")
            )
            .withColumn(
                "count_construction",
                when(col("sum(aux)") == 0, 0)
                .when(
                    col("sum(aux)").isNotNull(), col("count(output_for_construction)")
                )
                .cast("long"),
            )
            .join(
                df.select("period", "strata", "next_period").distinct(),
                ["period", "strata"],
            )
        )

        # Calculate backward ratio for each strata; reuse calculations from
        # above where applicable.
        returned = forward_df.join(
            forward_df.select(
                col("period").alias("other_period"),
                col("strata").alias("other_strata"),
                col("sum(other_output)").alias("sum_output"),
                col("sum(output)").alias("sum_other_output"),
                col("count(output)").alias("count_output"),
            ),
            [
                col("next_period") == col("other_period"),
                col("strata") == col("other_strata"),
            ],
            "leftouter",
        ).select(
            col("period"),
            col("strata"),
            col("forward"),
            (col("sum_output") / col("sum_other_output")).alias("backward"),
            when(col("sum_other_output") == 0, 0)
            .when(col("sum_other_output").isNotNull(), col("count_output"))
            .cast("long")
            .alias("count_backward"),
            col("construction"),
            col("count_construction"),
            col("count_forward"),
        )
        return [engine.RatioCalculationResult(data=returned_df, columns=["period", "strata"], fill_columns=["fforward", "backward", "construction"])]

    kwargs["ratio_calculation_function"] = ratio_of_means

    return engine.impute(**kwargs)
