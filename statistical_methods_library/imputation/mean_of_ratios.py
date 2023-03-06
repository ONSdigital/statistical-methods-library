# For Copyright information, please see LICENCE.
from decimal import Decimal
from numbers import Number
from typing import List, Optional

from pyspark.sql import DataFrame
from pyspark.sql.functions import col, expr, when

from . import engine


def impute(
    *,
    trim_threshold: Optional[Number] = None,
    lower_trim: Optional[Number] = None,
    upper_trim: Optional[Number] = None,
    include_zeros: Optional[Boolean] = False,
    **kwargs
) -> DataFrame:
    def mean_of_ratios(df: DataFrame) -> List[engine.RatioCalculationResult]:
        if not include_zeros:
            df = df.filter(
                "0 NOT IN (previous.output, current.output, next.output)"
            )

        df = df.selectExpr(
            "period",
            "grouping",
            "ref",
            "aux",
            "current.output",
            """CASE
                WHEN previous.output = 0
                THEN 1
                ELSE current.output/previous.output
            END AS growth_forward""",
            """CASE
                WHEN next.output = 0
                THEN 1
                ELSE current.output/next.output
            END AS growth_backward""",
        )

        if lower_trim is not None:
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
                        )
                        .select(
                            col("period"),
                            col("grouping"),
                            (col("count_forward") * Decimal(lower_trim) / 100).alias(
                                "lower_forward"
                            ),
                            (
                                col("count_forward") * (100 - Decimal(upper_trim)) / 100
                            ).alias("upper_forward"),
                            (col("count_backward") * Decimal(lower_trim) / 100).alias(
                                "lower_backward"
                            ),
                            (
                                col("count_backward")
                                * (100 - Decimal(upper_trim))
                                / 100
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
                    when(
                        (
                            col("num_forward").between(
                                col("lower_forward"), col("upper_forward")
                            )
                            | trim_threshold
                            >= col("count_forward")
                        ),
                        col("growth_forward"),
                    ).alias("trimmed_forward"),
                    when(
                        (
                            col("num_backward").between(
                                col("lower_backward"), col("upper_backward")
                            )
                            | trim_threshold
                            >= col("count_backward")
                        ),
                        col("growth_backward"),
                    ).alias("trimmed_backward"),
                )
            )

        else:
            trimmed_df = df.withColumn(
                "trimmed_forward", col("growth_forward")
            ).withColumn("trimmed_backward", col("growth_backward"))

        ratio_df = trimmed_df.groupBy("period", "grouping").agg(
            expr("mean(trimmed_forward) AS forward"),
            expr("mean(trimmed_backward) AS backward"),
            expr("sum(current.output)/sum(aux) AS construction"),
            expr("count(trimmed_forward) AS count_forward"),
            expr("count(trimmed_backward) AS count_backward"),
            expr("count(current.output) AS count_construction"),
        )

        growth_df = df.select(
            "ref", "period", "grouping", "growth_forward", "growth_backward"
        )
        return [
            engine.RatioCalculationResult(
                data=growth_df,
                join_columns=["period", "grouping", "ref"],
            ),
            engine.RatioCalculationResult(
                data=ratio_df,
                join_columns=["period", "grouping"],
                fill_columns=["forward", "backward", "construction"],
            ),
        ]

    kwargs["ratio_calculation_function"] = mean_of_ratios

    return engine.impute(**kwargs)
