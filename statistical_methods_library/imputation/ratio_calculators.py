"""
Ratio calculator functions written for use with the imputation engine.

At a minimum ratio calculators are passed a data frame containing the
following columns:
    ref: Aliased from the `reference_col` engine argument.
    grouping: Aliased from the `grouping_col` engine argument.
    period: Aliased from the `period_col` engine argument.
    aux: Aliased from the `auxiliary_col` engine argument.
    output: Aliased version of the `target_col` engine argument for
      the data in the current period.
    link_inclusion_current: See the `link_inclusion_current_col` argument for
      the semantics of this column.
    next_output: Aliased version of the `target_col` engine argument for
      the data in the next period.
    link_inclusion_next: See the `link_inclusion_next_col` argument for the
      semantics of this column.
    previous_output: Aliased version of the `target_col` engine argument for
      the data in the previous period.
    link_inclusion_previous: See the `link_inclusion_previous_col` argument for the
      semantics of this column.

Ratio calculators can also accept arbitrary keyword arguments from the engine
(see the `ratio_calculator_params` engine argument).

For Copyright information, please see LICENCE.
"""

from dataclasses import dataclass, field
from decimal import Decimal
from numbers import Number
from typing import Any, Callable, Dict, Iterable, List, Optional

from pyspark.sql import DataFrame
from pyspark.sql.functions import ceil as sql_ceil
from pyspark.sql.functions import col, expr
from pyspark.sql.functions import floor as sql_floor
from pyspark.sql.functions import lit, when


@dataclass
class RatioCalculationResult:
    "Type for returns from ratio calculators"

    data: DataFrame
    "The data being returned"

    join_columns: List[str]
    """
    The names of columns used to join the `data` attribute to the data to be
    imputed.
    """

    fill_values: Optional[Dict[str, Any]] = field(default_factory=dict)
    """
    A mapping of column names in the `data` attribute to their fill values for
    columns requiring filling.
    """
    additional_outputs: Optional[Dict[str, str]] = field(default_factory=dict)
    """
    A mapping from the column name in the `data` attribute to the column's
    user-provided name for output in cases where the ratio calculator provides
    additional output columns.
    """


RatioCalculator = Callable[[DataFrame, Any], Iterable[RatioCalculationResult]]
"The overall type for a ratio calculator to be provided to the engine."


def mean_of_ratios(
    *,
    df: DataFrame,
    trim_threshold: Optional[Number] = None,
    lower_trim: Optional[Number] = None,
    upper_trim: Optional[Number] = None,
    include_zeros: Optional[bool] = False,
    growth_forward_col: Optional[str] = "growth_forward",
    growth_backward_col: Optional[str] = "growth_backward",
    trim_inclusion_forward_col: Optional[str] = "trim_inclusion_forward",
    trim_inclusion_backward_col: Optional[str] = "trim_inclusion_backward",
    **_kwargs,
) -> List[RatioCalculationResult]:
    """
    Perform Mean of Ratios forward and backward ratio calculation.

    Args:
        df: The input data frame.
        trim_threshold: The minimum number of matched pairs needed for trimming
          to occur.
        lower_trim: The percentage to trim off the bottom.
        upper_trim: The percentage to trim off the top.
        include_zeros: Set to True to include zeros in ratio calculations,
            otherwise zeros are removed from the data prior to processing.
        growth_forward_col: The name of the column containing the forward
          growth ratio.
        growth_backward_col: The name of the column containing the backward
          growth ratio.
        trim_inclusion_forward_col: The name of the column marking whether the
          growth ratio was included in forward ratio calculations post trimming.
        trim_inclusion_backward_col: The name of the column marking whether the
          growth ratio was included in backward ratio calculations post
          trimming.

    Returns:
        A result containing forward and backward links, count of matched pairs
          and if the links were defaulted. The data frame contains a row for
          each period and grouping combination in the input data.
        A result containing forward and backward growth ratios. The exact
          columns depend on if trimming is performed as specified by the
          provided arguments. The data frame contains a row for each
          reference, period and grouping combination in the input data.

    For trimming to occur, `trim_threshold`, `lower_trim` and `upper_trim` need
    to be present. When trimming occurs, `trim_inclusion_forward_col` and
    `trim_inclusion_backward_col` will be output.

    The `lower_trim` and `upper_trim` are approximate percentages as trimming
    uses exclusive bounds when calculating which rows to remove.
    """
    print("before the growth calculation")
    df.printSchema()
    df.show(5)
    if lower_trim is not None:
        lower_trim = Decimal(lower_trim)
        upper_trim = Decimal(upper_trim)
        trim_threshold = Decimal(trim_threshold)

    df = df.select(
        "period",
        "grouping",
        "ref",
        when(col("link_inclusion_current"), col("aux")).alias("aux"),
        "link_inclusion_previous",
        "link_inclusion_next",
        "link_inclusion_current",
        when(
            col("link_inclusion_previous")
            & (lit(include_zeros) | (col("previous_output") != lit(0))),
            col("previous_output"),
        ).alias("previous_output"),
        when(
            col("link_inclusion_current")
            & (lit(include_zeros) | (col("output") != lit(0))),
            col("output"),
        ).alias("current_output"),
        when(
            col("link_inclusion_next")
            & (lit(include_zeros) | (col("next_output") != lit(0))),
            col("next_output"),
        ).alias("next_output"),
    ).selectExpr(
        "period",
        "grouping",
        "ref",
        "aux",
        "current_output",
        "link_inclusion_previous",
        "link_inclusion_next",
        "link_inclusion_current",
        """CASE
                WHEN link_inclusion_current THEN CASE
                    WHEN link_inclusion_previous AND link_inclusion_previous IS NOT NULL
                    THEN CASE
                        WHEN previous_output = 0 OR current_output = 0
                        THEN 1
                        ELSE current_output/previous_output
                    END
                END
            END AS growth_forward""",
        """CASE
                WHEN link_inclusion_current THEN CASE
                    WHEN link_inclusion_next AND link_inclusion_next IS NOT NULL
                    THEN CASE
                        WHEN next_output = 0 OR current_output = 0
                        THEN 1
                        ELSE current_output/next_output
                    END
                END
            END AS growth_backward""",
    )
    print("after the growth calculation")
    df.printSchema()
    df.show(5)
    if lower_trim is not None:

        def lower_bound(c):
            return sql_ceil(c * lower_trim / 100)

        def upper_bound(c):
            return 1 + sql_floor(c * (100 - upper_trim) / 100)

        df_lwr_upr_bound = (
            df.groupBy("period", "grouping")
            .agg(
                expr(
                    """
                            sum(
                                cast(growth_forward IS NOT NULL AS integer)
                            ) AS count_forward
                            """
                ),
                expr(
                    """
                            sum(
                                cast(growth_backward IS NOT NULL AS integer)
                            ) AS count_backward
                            """
                ),
                expr(
                    """
                            sum(
                                cast(
                                    not (
                                        (link_inclusion_previous OR
                                        link_inclusion_previous IS NULL)
                                        AND link_inclusion_current
                                    )
                                AS integer)
                            )
                            AS count_exclusion_forward
                            """
                ),
                expr(
                    """
                            sum(
                                cast(
                                    not (
                                        (link_inclusion_next OR
                                        link_inclusion_next IS NULL)
                                        AND link_inclusion_current
                                    )
                                    AS integer
                                )
                            )
                            AS count_exclusion_backward
                            """
                ),
            )
            .select(
                col("period"),
                col("grouping"),
                col("count_exclusion_forward"),
                col("count_exclusion_backward"),
                col("count_forward"),
                col("count_backward"),
                lower_bound(
                    col("count_forward"),
                ).alias("lower_forward"),
                upper_bound(
                    col("count_forward"),
                ).alias("upper_forward"),
                lower_bound(
                    col("count_backward"),
                ).alias("lower_backward"),
                upper_bound(
                    col("count_backward"),
                ).alias("upper_backward"),
            )
            .localCheckpoint(eager=True)
        )
        print("after the lower & upper bound calculation:: df_lwr_upr_bound")
        df_lwr_upr_bound.printSchema()
        df_lwr_upr_bound.show(5)
        df = df.join(df_lwr_upr_bound, ["period", "grouping"])
        # When calculating row numbers we put the null values last to avoid
        # them impacting the trimmed mean. This works because the upper
        # bound is calculated based on the count of non-null growth ratios.
        # Secondary ordering by reference. This does not impact the calculated
        # forward/backward links, since it will only apply to contributors with equal
        # growth ratios, but keeps the selection of rows for trimming deterministic.

        df = (
            df.withColumn(
                "num_forward",
                expr(
                    """
                    row_number() OVER (
                        PARTITION BY period, grouping
                        ORDER BY growth_forward ASC NULLS LAST, ref ASC
                    )
                """
                ),
            )
            .withColumn(
                "num_backward",
                expr(
                    """
                    row_number() OVER (
                        PARTITION BY period, grouping
                        ORDER BY growth_backward ASC NULLS LAST, ref ASC
                    )
                """
                ),
            )
            .withColumn(
                "trim_inclusion_forward",
                (
                    when(col("growth_forward").isNull(), None).otherwise(
                        col("num_forward").between(
                            col("lower_forward"), col("upper_forward")
                        )
                        | (
                            (trim_threshold - col("count_exclusion_forward"))
                            >= col("count_forward")
                        )
                    )
                ),
            )
            .withColumn(
                "trim_inclusion_backward",
                (
                    when(col("growth_backward").isNull(), None).otherwise(
                        col("num_backward").between(
                            col("lower_backward"), col("upper_backward")
                        )
                        | (
                            (trim_threshold - col("count_exclusion_backward"))
                            >= col("count_backward")
                        )
                    )
                ),
            )
        )

    else:
        df = df.withColumn("trim_inclusion_forward", lit(True)).withColumn(
            "trim_inclusion_backward", lit(True)
        )
    print("**************")
    print("before mean agg")
    df.printSchema
    df.show(50)
    ratio_df = (
        df.groupBy("period", "grouping")
        .agg(
            expr(
                """mean(
                CASE WHEN trim_inclusion_forward THEN growth_forward END
            ) AS forward"""
            ),
            expr(
                """mean(
                CASE WHEN trim_inclusion_backward THEN growth_backward END
            ) AS backward"""
            ),
            expr(
                """sum(cast(
                trim_inclusion_forward AND growth_forward IS NOT NULL AS integer
            )) AS count_forward"""
            ),
            expr(
                """sum(cast(
                trim_inclusion_backward AND growth_backward IS NOT NULL AS integer
            )) AS count_backward"""
            ),
        )
        .withColumn("default_forward", expr("forward IS NULL"))
        .withColumn("default_backward", expr("backward IS NULL"))
    )
    print("after mean agg")
    ratio_df.show(50)
    print("----------------")
    growth_additional_outputs = {
        "growth_forward": growth_forward_col,
        "growth_backward": growth_backward_col,
    }

    if lower_trim is not None:
        growth_additional_outputs.update(
            {
                "trim_inclusion_forward": trim_inclusion_forward_col,
                "trim_inclusion_backward": trim_inclusion_backward_col,
            }
        )

    growth_df = df.select(
        "ref", "period", "grouping", *growth_additional_outputs.keys()
    )

    return [
        RatioCalculationResult(
            data=growth_df,
            join_columns=["period", "grouping", "ref"],
            additional_outputs=growth_additional_outputs,
        ),
        RatioCalculationResult(
            data=ratio_df,
            join_columns=["period", "grouping"],
            fill_values={
                "forward": 1,
                "backward": 1,
                "count_forward": 0,
                "count_backward": 0,
                "default_forward": True,
                "default_backward": True,
            },
        ),
    ]


def ratio_of_means(*, df: DataFrame, **_kw) -> List[RatioCalculationResult]:
    """
    Perform Ratio of Means forward and backward ratio calculation.

    Args:
        df: The input data frame.

    Returns:
    A result containing forward and backward links, count of matched pairs and
        if the links were defaulted. The data frame contains a row for each
        period and grouping combination in the input data.
    """
    df = (
        df.filter(col("link_inclusion_current"))
        .withColumn(
            "previous_output",
            expr("CASE WHEN link_inclusion_previous <=> TRUE then previous_output END"),
        )
        .withColumn(
            "next_output",
            expr("CASE WHEN link_inclusion_next <=> TRUE then next_output END"),
        )
        .groupBy("period", "grouping")
        .agg(
            expr(
                """
                    sum(
                        CASE
                            WHEN previous_output IS NOT NULL
                            THEN output
                        END
                    )/sum(previous_output) AS forward
                """
            ),
            expr(
                """
                    sum(
                        CASE
                            WHEN next_output IS NOT NULL
                            THEN output
                        END
                    )/sum(next_output) AS backward
                """
            ),
            expr("count(previous_output) AS count_forward"),
            expr("count(next_output) AS count_backward"),
            expr("coalesce(sum(previous_output), 0) = 0 AS default_forward"),
            expr("coalesce(sum(next_output), 0) = 0 AS default_backward"),
        )
    )
    return [
        RatioCalculationResult(
            data=df,
            join_columns=["period", "grouping"],
            fill_values={
                "forward": 1,
                "backward": 1,
                "count_forward": 0,
                "count_backward": 0,
                "default_forward": True,
                "default_backward": True,
            },
        )
    ]


def ratio_of_means_construction(
    *, df: DataFrame, **_kw
) -> List[RatioCalculationResult]:
    """
    Perform Ratio of Means construction ratio calculation.

    Args:
        df: The input data frame.

    Returns:
    A result containing construction links, the count of matched pairs and if the
    links were defaulted. The data frame contains a row for each period and
    grouping combination in the input data.
    """
    return [
        RatioCalculationResult(
            data=(
                df.filter(col("link_inclusion_current"))
                .groupBy("period", "grouping")
                .agg(
                    expr("sum(output)/sum(aux) AS construction"),
                    expr("count(aux) AS count_construction"),
                    expr("coalesce(sum(aux), 0) = 0 AS default_construction"),
                )
            ),
            join_columns=["period", "grouping"],
            fill_values={
                "construction": 1,
                "count_construction": 0,
                "default_construction": True,
            },
        )
    ]
