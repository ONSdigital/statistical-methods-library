# For Copyright information, please see LICENCE.
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
    data: DataFrame
    join_columns: List[str]
    fill_values: Optional[Dict[str, str]] = field(default_factory=dict)
    additional_outputs: Optional[Dict[str, str]] = field(default_factory=dict)


RatioCalculator = Callable[[DataFrame, Any], Iterable[RatioCalculationResult]]


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
            & (lit(include_zeros) | (col("previous.output") != lit(0))),
            col("previous.output"),
        ).alias("previous_output"),
        when(
            col("link_inclusion_current")
            & (lit(include_zeros) | (col("current.output") != lit(0))),
            col("current.output"),
        ).alias("current_output"),
        when(
            col("link_inclusion_next")
            & (lit(include_zeros) | (col("next.output") != lit(0))),
            col("next.output"),
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

    if lower_trim is not None:

        def lower_bound(c):
            return sql_ceil(c * lower_trim / 100)

        def upper_bound(c):
            return 1 + sql_floor(c * (100 - upper_trim) / 100)

        df = (
            df.join(
                (
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
                ),
                ["period", "grouping"],
            )
            # When calculating row numbers we put the null values last to avoid
            # them impacting the trimmed mean. This works because the upper
            # bound is calculated based on the count of non-null growth ratios.
            .withColumn(
                "num_forward",
                expr(
                    """
                    row_number() OVER (
                        PARTITION BY period, grouping
                        ORDER BY growth_forward ASC NULLS LAST
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
                        ORDER BY growth_backward ASC NULLS LAST
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
        "ref", "period", "grouping", "link_inclusion_previous", "link_inclusion_next",
        "link_inclusion_current", *growth_additional_outputs.keys()
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
    df = (
        df.filter(col("link_inclusion_current"))
        .groupBy("period", "grouping")
        .agg(
            expr(
                """
                    sum(
                        CASE
                            WHEN previous.output IS NOT NULL
                            THEN current.output
                        END
                    )/sum(previous.output) AS forward
                """
            ),
            expr(
                """
                    sum(
                        CASE
                            WHEN next.output IS NOT NULL
                            THEN current.output
                        END
                    )/sum(next.output) AS backward
                """
            ),
            expr("count(previous.output) AS count_forward"),
            expr("count(next.output) AS count_backward"),
            expr("coalesce(sum(previous.output), 0) = 0 AS default_forward"),
            expr("coalesce(sum(next.output), 0) = 0 AS default_backward"),
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


def construction(*, df: DataFrame, **_kw) -> List[RatioCalculationResult]:
    return [
        RatioCalculationResult(
            data=(
                df.filter(col("link_inclusion_current"))
                .groupBy("period", "grouping")
                .agg(
                    expr("sum(current.output)/sum(aux) AS construction"),
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
