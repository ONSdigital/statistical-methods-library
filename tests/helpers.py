from pytest import fail
from pyspark.sql.functions import col, count, create_map, monotonically_increasing_id, lit, when
from functools import reduce
def check_df_equality(df1, df2, keep_cols=None):
    if keep_cols is None:
        keep_cols = []
    else:
        keep_cols = list(keep_cols)
    msg = []
    df1_cols = set(df1.columns)
    df2_cols = set(df2.columns)
    df1_col_check = df1_cols - df2_cols
    df2_col_check = df2_cols - df1_cols
    if df1_col_check:
        msg.append("extra columns in df1: {', '.join(df1_col_check)}")

    if df2_col_check:
        msg.append("extra columns in df2: {', '.join(df2_col_check)}")

    if msg:
        fail('\n'.join(msg))

    col_list = sorted(df1.columns)
    df1 = df1.select(col_list).alias("df1").withColumn("id", monotonically_increasing_id())
    df2 = df2.select(col_list).alias("df2").withColumn("id", monotonically_increasing_id())
    filter_list = [
        ~col(f"df1.{name}").eqNullSafe(col(f"df2.{name}"))
        for name in col_list
    ]
    diff_df = (
        df1.join(df2, ["id"], "full")
        .filter(reduce(lambda x, y: x | y, filter_list))
        .select(
            *(col(f"df1.{name}").alias(f"df1_{name}") for name in col_list),
            *(col(f"df2.{name}").alias(f"df2_{name}") for name in col_list)
        )
    )
    diff_count = diff_df.count()
    if diff_count > 0:
        # Drop any columns where all values are equal as we don't need these
        # in our output.
        # This expression should only return 1 row so will scale.
        equal_counts = diff_df.select(
            [
                count(
                    when(col(f"df1_{c}").eqNullSafe(col(f"df2_{c}")), c)
                ).alias(c)
                for c in col_list
            ]
        ).collect()[0].asDict()

        diff_df = diff_df.select(
            [
                create_map(
                    lit("df1"),
                    col(f"df1_{name}"),
                    lit("df2"),
                    col(f"df2_{name}")
                ).alias(name)
                for name in sorted(col_list)
                if equal_counts[name] != diff_count or name in keep_cols
            ]
        )

        diff_str = diff_df._jdf.showString(100, 100, False)
        fail(
            f"Mismatching rows in provided data frames:\n\n{diff_str}"
        )
