# For Copyright information, please see LICENCE.

from typing import List, Number, Optional

from pyspark.sql import DataFrame

from . import engine


def impute(*, lower_trim: Optional[Number]=None, upper_trim: Optional[Number]=None, **kwargs) -> DataFrame:
    def mean_of_ratios(df: DataFrame) -> List[engine.RatioCalculationResult]:
        growth_df = df.selectExpr(
            "period",
            "grouping",
            "ref",
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

        trimmed_df = (
            growth_df.groupBy("period", "grouping")
            .agg(
                expr("sum(growth_forward IS NOT NULL) AS count_forward"),
                expr("sum(growth_backward IS NOT NULL) AS count_backward"),
            )
            .select(
                col("period"),
                col("grouping"),
                (1 + (col("count_forward") * int(lower_trim) / 100)).alias("lower_forward"),
                (col("count_forward") * (100 - int(upper_trim)) / 100).alias("lower_forward"),
                (1 + (col("count_backward") * int(lower_trim) / 100)).alias("lower_backward"),
                (col("count_backward") * (100 - int(upper_trim)) / 100).alias("lower_backward"),
            )
        )

        return [
            engine.RatioCalculationResult(
                data=growth_df,
                join_columns=["period", "grouping", "ref"],
            )
        ]

    kwargs["ratio_calculation_function"] = mean_of_ratios

    return engine.impute(**kwargs)
