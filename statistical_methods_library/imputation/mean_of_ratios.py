# For Copyright information, please see LICENCE.

from typing import List

from pyspark.sql import DataFrame
from pyspark.sql.functions import col, count, sum, when

from . import engine


def impute(**kwargs) -> DataFrame:

    def mean_of_ratios(df: DataFrame) -> List[engine.RatioCalculationResult]:
        working_df = df.withColumn(
            "growth_ratio_forward",
            col("output")/col("other_output")
            ).join(
            df.select(
                col("period").alias("other_period"),
                col("grouping").alias("other_grouping"),
                col("other_output").alias("reverse_output"),
                col("output").alias("reverse_other_output"),
            ),
            [
                col("next_period") == col("other_period"),
                col("grouping") == col("other_grouping"),
            ],
            "leftouter",
        ).select(
            col("period"),
            col("grouping"),
            col("growth_ratio_forward"),
            (col("reverse_output") / col("reverse_other_output")).alias("growth_ratio_backward"),
        )

        # Calculate the construction links as a ratio of means between the returned values and the auxiliary value
        construction_df = (
            df.groupBy("period", "grouping").agg(
            sum(col("aux")),
            sum(col("output_for_construction")),
            count(col("output_for_construction")),
            ).withColumn(
                "construction", col("sum(output_for_construction)") / col("sum(aux)")
            )
            .withColumn(
                "count_construction",
                when(col("sum(aux)") == 0, 0)
                .when(
                    col("sum(aux)").isNotNull(), col("count(output_for_construction)")
                )
                .cast("long"),
            )
            .join(
                df.select("period", "grouping").distinct(),
                ["period", "grouping"],
            )
        )

        return [
            engine.RatioCalculationResult(
                data=returned_df,
                join_columns=["period", "grouping"],
                fill_columns=["forward", "backward", "construction"],
            )
        ]

    kwargs["ratio_calculation_function"] = mean_of_ratios

    return engine.impute(**kwargs)
