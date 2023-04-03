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
from pyspark.sql.functions import when


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
    **_kwargs,
) -> List[RatioCalculationResult]:
    ratio_cols = [
        "period",
        "grouping",
        "ref",
        "aux",
    ]
    match_cols = ["previous.match", "current.match", "next.match"]
    if not include_zeros:
        df = df.selectExpr(
            *ratio_cols + match_cols,
            """
                CASE
                    WHEN previous.output != 0 THEN previous.output
                END AS previous_output
            """,
            """
                CASE
                    WHEN current.output != 0 THEN current.output
                END AS current_output
            """,
            "CASE WHEN next.output != 0 THEN next.output END AS next_output",
        )

    else:
        df = df.selectExpr(
            *ratio_cols + match_cols,
            "previous.output AS previous_output",
            "current.output AS current_output",
            "next.output AS next_output",
        )

    df = df.selectExpr(
        *ratio_cols,
        "CASE WHEN previous.match THEN previous_output END AS previous_output",
        "CASE WHEN current.match THEN current_output END AS current_output",
        "CASE WHEN next.match THEN next_output END AS next_output",
        """NOT (previous.match AND current.match) AND previous.match IS NOT NULL
        AS filtered_forward""",
        """NOT (next.match AND current.match) AND next.match IS NOT NULL
        AS filtered_backward""",
    )
    df = df.selectExpr(
        *ratio_cols,
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

        trimmed_df = (
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
            .select(
                col("period"),
                col("grouping"),
                col("aux"),
                col("current_output"),
                when(
                    (
                        col("num_forward").between(
                            col("lower_forward"), col("upper_forward")
                        )
                        | trim_threshold - col("count_filtered_forward")
                        >= col("count_forward")
                    ),
                    col("growth_forward"),
                ).alias("trimmed_forward"),
                when(
                    (
                        col("num_backward").between(
                            col("lower_backward"), col("upper_backward")
                        )
                        | trim_threshold - col("count_filtered_backward")
                        >= col("count_backward")
                    ),
                    col("growth_backward"),
                ).alias("trimmed_backward"),
            ),
        )

    else:
        trimmed_df = df.selectExpr(
            "period",
            "grouping",
            "aux",
            "current_output",
            "growth_forward AS trimmed_forward",
            "growth_backward AS trimmed_backward",
        )

    ratio_df = trimmed_df.groupBy("period", "grouping").agg(
        expr("mean(trimmed_forward) AS forward"),
        expr("mean(trimmed_backward) AS backward"),
        expr("sum(current_output)/sum(aux) AS construction"),
        expr("count(trimmed_forward) AS count_forward"),
        expr("count(trimmed_backward) AS count_backward"),
        expr("count(current_output) AS count_construction"),
    )

    growth_df = df.select(
        "ref",
        "period",
        "grouping",
        "growth_forward",
        "growth_backward",
        "filtered_forward",
        "filtered_backward",
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
