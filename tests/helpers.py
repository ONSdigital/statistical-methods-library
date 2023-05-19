from functools import reduce

from pyspark.sql.functions import (
    col,
    count,
    create_map,
    lit,
    monotonically_increasing_id,
    when,
)
from pytest import fail


def check_df_equality(expected, actual, keep_cols=None):
    if keep_cols is None:
        keep_cols = []
    else:
        keep_cols = list(keep_cols)
    msg = []
    expected_cols = set(expected.columns)
    actual_cols = set(actual.columns)
    expected_col_check = expected_cols - actual_cols
    actual_col_check = actual_cols - expected_cols
    if expected_col_check:
        msg.append("extra columns in expected: {', '.join(expected_col_check)}")

    if actual_col_check:
        msg.append("extra columns in actual: {', '.join(actual_col_check)}")

    if msg:
        fail("\n".join(msg))

    col_list = sorted(expected.columns)
    expected = (
        expected.select(col_list)
        .alias("expected")
        .withColumn("id", monotonically_increasing_id())
    )
    actual = (
        actual.select(col_list)
        .alias("actual")
        .withColumn("id", monotonically_increasing_id())
    )
    filter_list = [
        ~col(f"expected.{name}").eqNullSafe(col(f"actual.{name}")) for name in col_list
    ]
    diff_df = (
        expected.join(actual, ["id"], "full")
        .filter(reduce(lambda x, y: x | y, filter_list))
        .select("id",
            *(col(f"expected.{name}").alias(f"expected_{name}") for name in col_list),
            *(col(f"actual.{name}").alias(f"actual_{name}") for name in col_list),
        )
    )
    diff_count = diff_df.count()
    if diff_count > 0:
        # Drop any columns where all values are equal as we don't need these
        # in our output.
        # This expression should only return 1 row so will scale.
        equal_counts = (
            diff_df.select(
                [
                    count(
                        when(col(f"expected_{c}").eqNullSafe(col(f"actual_{c}")), c)
                    ).alias(c)
                    for c in col_list
                ]
            )
            .collect()[0]
            .asDict()
        )

        diff_df = diff_df.sort("id").select(
            [
                create_map(
                    lit("expected"),
                    col(f"expected_{name}"),
                    lit("actual"),
                    col(f"actual_{name}"),
                ).alias(name)
                for name in sorted(col_list)
                if equal_counts[name] != diff_count or name in keep_cols
            ]
        )

        diff_str = "\n".join(str(row) for row in diff_df.take(100))
        fail(
            f"Mismatching rows in provided data frames (showing up to 100):\n\n{diff_str}"
        )