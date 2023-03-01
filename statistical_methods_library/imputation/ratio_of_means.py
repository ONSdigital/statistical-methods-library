# For Copyright information, please see LICENCE.

from typing import List

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
    * `grouping_col`: The Name of the column containing the grouping information
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
        *`grouping_col`
        *`auxiliary_col`
        *`output_col`
        *`marker_col`
    """

    def ratio_of_means(df: DataFrame) -> List[engine.RatioCalculationResult]:
        returned_df = df.groupBy("period", "grouping").selectExpr(
            "period",
            "grouping",
            "sum(CASE WHEN previous.output IS NOT NULL THEN current.output)/sum(previous.output END) AS forward",
            "sum(CASE WHEN next.output IS NOT NULL THEN current.output)/sum(next.output END) AS backward",
            "sum(current.output)/sum(aux) AS construction",
            "sum(CASE WHEN previous.output IS NOT NULL THEN 1 END) AS count_forward",
            "sum(CASE WHEN next.output IS NOT NULL THEN 1 END) AS count_backward",
            "count(current.output) AS count_construction",
            )

        return [
            engine.RatioCalculationResult(
                data=returned_df,
                join_columns=["period", "grouping"],
                fill_columns=["forward", "backward", "construction"],
            )
        ]

    kwargs["ratio_calculation_function"] = ratio_of_means

    return engine.impute(**kwargs)
