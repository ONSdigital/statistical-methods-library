from pytest import fail
from pyspark.sql.functions import col, monotonically_increasing_id
from functools import reduce
def check_df_equality(df1, df2):
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

    df1 = df1.select(sorted(df1.columns)).alias("df1").withColumn("id", monotonically_increasing_id())
    df2 = df2.select(sorted(df2.columns)).alias("df2").withColumn("id", monotonically_increasing_id())
    filter_list = [
        ~col(f"df1.{name}").eqNullSafe(col(f"df2.{name}"))
        for name in df1_cols
    ]
    diff_df = (
        df1.join(df2, ["id"], "full")
        .filter(reduce(lambda x, y: x & y, filter_list))
    )
    if diff_df.count() > 0:
        diff_str = diff_df._jdf.showString(100, 100, False)
        fail(
            "Mismatching rows in provided data frames:\n\n{diff_str}"
        )
