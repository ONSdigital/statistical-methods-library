# For Copyright information, please see LICENCE.

from typing import List

from pyspark.sql import DataFrame
from pyspark.sql.functions import col, count, sum, when

from . import engine


def impute(**kwargs) -> DataFrame:
    def mean_of_ratios(df: DataFrame) -> List[engine.RatioCalculationResult]:
        growth_df = df.selectExpr(
            "period",
            "grouping",
            "ref",
            "current.output/previous.output AS growth_forward",
            "current.output/next.output AS growth_backward"

        )

        return [
            engine.RatioCalculationResult(
                data=growth_df,
                join_columns=["period", "ref", "grouping"],
                fill_columns=["growth_forward", "growth_backward"]
            )
        ]

    kwargs["ratio_calculation_function"] = mean_of_ratios

    return engine.impute(**kwargs)
