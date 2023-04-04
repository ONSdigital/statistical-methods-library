# For Copyright information, please see LICENCE.
from dataclasses import dataclass, field
from decimal import Decimal
from numbers import Number
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

from pyspark.sql import Column, DataFrame
# Avoid shadowing builtin floor and ceil functions
from pyspark.sql.functions import ceil as sql_ceil
from pyspark.sql.functions import col, expr
from pyspark.sql.functions import floor as sql_floor
from pyspark.sql.functions import lit, when


@dataclass
class RatioCalculationResult:
    data: DataFrame
    join_columns: List[Union[str, Column]]
    fill_columns: List[Union[str, Column]] = field(default_factory=list)
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
    filtered_forward_col: Optional[str] = "filtered_forward",
    filtered_backward_col: Optional[str] = "filtered_backward",
    trimmed_forward_col: Optional[str] = "trimmed_forward",
    trimmed_backward_col: Optional[str] = "trimmed_backward",
    **_kwargs,
) -> List[RatioCalculationResult]:
    df = df.select(
        "period",
        "grouping",
        "ref",
        when(col("current.match"), col("aux")).alias("aux"),
        when(
            col("previous.match")
            & (lit(include_zeros) | (col("previous.output") != lit(0))),
            col("previous.output"),
        ).alias("previous_output"),
        when(
            col("current.match")
            & (lit(include_zeros) | (col("current.output") != lit(0))),
            col("current.output"),
        ).alias("current_output"),
        when(
            col("next.match") & (lit(include_zeros) | (col("next.output") != lit(0))),
            col("next.output"),
        ).alias("next_output"),
        expr(
            """
            NOT (previous.match AND current.match) AND previous.match IS NOT NULL
            AS filtered_forward"""
        ),
        expr(
            """NOT (next.match AND current.match) AND next.match IS NOT NULL
            AS filtered_backward"""
        ),
    ).selectExpr(
        "period",
        "grouping",
        "ref",
        "aux",
        "current_output",
        "filtered_forward",
        "filtered_backward",
        """CASE
                WHEN NOT filtered_forward THEN CASE
                    WHEN previous_output = 0 OR
                    (current_output = 0 AND previous_output IS NOT NULL)
                    THEN 1
                    ELSE current_output/previous_output
                END
            END AS growth_forward""",
        """CASE
                WHEN NOT filtered_backward
                THEN CASE
                    WHEN
                        next_output = 0
                        OR (current_output = 0 AND next_output IS NOT NULL)
                    THEN 1
                    ELSE current_output/next_output
                END
            END AS growth_backward""",
    )

    if lower_trim is not None:

        def lower_bound(c):
            return sql_ceil(c * lower_trim / 100)

        def upper_bound(c, f):
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
                            sum(cast(filtered_forward AS integer))
                            AS count_filtered_forward
                        """
                        ),
                        expr(
                            """
                            sum(cast(filtered_backward AS integer))
                            AS count_filtered_forward
                        """
                        ),
                    )
                    .select(
                        col("period"),
                        col("grouping"),
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
            .withColumn(
                "num_forward",
                expr(
                    """
                    row_number() OVER (
                        PARTITION BY period, grouping
                        ORDER BY growth_forward ASC
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
                        ORDER BY growth_backward ASC
                    )
                """
                ),
            )
            .withColumn(
                "trimmed_forward",
                ~(
                    col("num_forward").between(
                        col("lower_forward"), col("upper_forward")
                    )
                    | trim_threshold - col("count_filtered_forward")
                    >= col("count_forward")
                ),
            )
            .withColumn(
                "trimmed_backward",
                ~(
                    col("num_backward").between(
                        col("lower_backward"), col("upper_backward")
                    )
                    | trim_threshold - col("count_filtered_backward")
                    >= col("count_backward")
                ),
            )
        )

    else:
        df = df.withColumn("trimmed_forward", lit(False)).withColumn(
            "trimmed_backward", lit(False)
        )

    ratio_df = df.groupBy("period", "grouping").agg(
        expr(
            """mean(
                CASE WHEN NOT trimmed_forward THEN growth_forward END
            ) AS forward"""
        ),
        expr(
            """mean(
                CASE WHEN NOT trimmed_backward THEN growth_backward END
            ) AS backward"""
        ),
        expr("sum(current_output)/sum(aux) AS construction"),
        expr(
            "sum(cast(NOT trimmed_forward AND growth_forward IS NOT NULL AS integer)) AS count_forward"
        ),
        expr(
            "sum(cast(NOT trimmed_backward AND growth_backward IS NOT NULL AS integer)) AS count_backward"
        ),
        expr("count(aux) AS count_construction"),
    )

    growth_df = df.select(
        "ref",
        "period",
        "grouping",
        "growth_forward",
        "growth_backward",
        "filtered_forward",
        "filtered_backward",
        "trimmed_forward",
        "trimmed_backward",
    )

    return [
        RatioCalculationResult(
            data=growth_df,
            join_columns=["period", "grouping", "ref"],
            additional_outputs={
                "growth_forward": growth_forward_col,
                "growth_backward": growth_backward_col,
                "filtered_forward": filtered_forward_col,
                "filtered_backward": filtered_backward_col,
                "trimmed_forward": trimmed_forward_col,
                "trimmed_backward": trimmed_backward_col,
            },
        ),
        RatioCalculationResult(
            data=ratio_df,
            join_columns=["period", "grouping"],
            fill_columns=["forward", "backward", "construction"],
        ),
    ]


def ratio_of_means(*, df: DataFrame, **_kw) -> List[RatioCalculationResult]:
    df = (
        df.filter(col("current.match"))
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
            expr("sum(current.output)/sum(aux) AS construction"),
            expr(
                """
                    CASE
                        WHEN sum(previous.output) = 0
                        THEN 0
                        WHEN sum(previous.output) IS NOT NULL
                        THEN sum(cast(previous.output IS NOT NULL AS integer))
                    END AS count_forward
                """
            ),
            expr(
                """
                    CASE
                        WHEN sum(next.output) = 0
                        THEN 0
                        WHEN sum(next.output) IS NOT NULL
                        THEN sum(cast(next.output IS NOT NULL AS integer))
                    END AS count_backward
                """
            ),
            expr(
                """
                    CASE
                        WHEN sum(aux) = 0
                        THEN 0
                        ELSE count(current.output)
                    END AS count_construction
                """
            ),
        )
    )
    return [
        RatioCalculationResult(
            data=df,
            join_columns=["period", "grouping"],
            fill_columns=["forward", "backward", "construction"],
        )
    ]
