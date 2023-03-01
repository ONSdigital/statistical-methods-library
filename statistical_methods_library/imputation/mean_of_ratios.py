# For Copyright information, please see LICENCE.

from typing import List

from pyspark.sql import DataFrame

from . import engine


def impute(**kwargs) -> DataFrame:
    def mean_of_ratios(df: DataFrame) -> List[engine.RatioCalculationResult]:
        growth_df = df.selectExpr(
            "period",
            "grouping",
            "ref",
            "current.output/previous.output AS growth_forward",
            "current.output/next.output AS growth_backward",
        ).fillna(1.0, "growth_forward", "growth_backward")

        return [
            engine.RatioCalculationResult(
                data=growth_df,
                join_columns=["period", "grouping", "ref"],
            )
        ]

    kwargs["ratio_calculation_function"] = mean_of_ratios

    return engine.impute(**kwargs)
