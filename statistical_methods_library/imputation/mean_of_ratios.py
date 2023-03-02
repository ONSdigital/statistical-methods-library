lower_# For Copyright information, please see LICENCE.

from typing import List, Number, Optional

from pyspark.sql import DataFrame
from pyspark.sql.functions import col, expr, when

from . import engine


def impute(*, lower_trim: Optional[Number]=None, upper_trim: Optional[Number]=None, **kwargs) -> DataFrame:
    def mean_of_ratios(df: DataFrame) -> List[engine.RatioCalculationResult]:
        growth_df = df.selectExpr(
            "period",
            "grouping",
            "ref",
            "aux",
            """CASE
                WHEN previous.output = 0
                THEN 1
                ELSE current.output/previous.output AS growth_forward
            END""",
            """CASE
                WHEN next.output = 0
                THEN 1
                ELSE current.output/next.output AS growth_backward
            END"""
        )

        ratio_df = (
            growth_df.join(
                (
                    growth_df.groupBy("period", "grouping")
                    .agg(
                        expr("""
                            sum(
                                cast(growth_forward IS NOT NULL AS integer)
                            ) AS count_forward
                        """),
                        expr("""
                            sum(
                                cast(growth_backward IS NOT NULL AS integer)
                            ) AS count_backward
                        """),
                    )
                    .select(
                        col("period"),
                        col("grouping"),
                        (
                            1 + (col("count_forward") * int(lower_trim) / 100)
                        ).alias("lower_forward"),
                        (
                            col("count_forward") * (100 - int(upper_trim)) / 100
                        ).alias("upper_forward"),
                        (
                            1 + (col("count_backward") * int(lower_trim) / 100)
                        ).alias("lower_backward"),
                        (
                            col("count_backward") * (100 - int(upper_trim)) / 100
                        ).alias("upper_backward"),
                    )
                ),
                ["period", "grouping"]
            )
            .withColumn(
                "num_forward",
                expr("""
                    row_number() OVER (
                        PARTITION BY period, grouping
                        ORDER BY growth_forward ASC
                    )
                """)
            )
            .withColumn(
                "num_backward",
                expr("""
                    row_number() OVER (
                        PARTITION BY period, grouping
                        ORDER BY growth_backward ASC
                    )
                """)
            )
            .select(
                col("period"),
                col("grouping"),
                when(
                    (
                        col("num_forward").between(
                            col("lower_forward""),
                            col("upper_forward")
                        )
                        | trim_threshold < col("count_forward")
                    ),
                    col("growth_forward")
                ).alias("trimmed_forward"),
                when(
                    (
                        col("num_backward").between(
                            col("lower_backward""),
                            col("upper_backward")
                        )
                        | trim_threshold < col("count_backward")
                    ),
                    col("growth_backward")
                ).alias("trimmed_backward")
            )
            .groupBy("period", "grouping")
            .agg(
                expr("mean(trimmed_forward) AS forward"),
                expr("mean(trimmed_backward) AS backward"),
            expr("sum(current.output)/sum(aux) AS construction"),
        )

        return [
            engine.RatioCalculationResult(
                data=growth_df,
                join_columns=["period", "grouping", "ref"],
            )
        ]

    kwargs["ratio_calculation_function"] = mean_of_ratios

    return engine.impute(**kwargs)
