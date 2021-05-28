from pyspark.sql import DataFrame
from pyspark.sql.functions import col, lead, lit, when

# --- Imputation errors ---


# Base type for imputation errors
class ImputationError(Exception):
    pass


# Error raised by dataframe validation
class ValidationError(ImputationError):
    pass


# Error raised when imputation has failed to impute for data integrity reasons
class DataIntegrityError(ImputationError):
    pass


def imputation(
    input_df,
    reference_col,
    period_col,
    strata_col,
    target_col,
    auxiliary_col,
    output_col,
    marker_col,
    forward_link_col=None,
    backward_link_col=None
):

    # --- Validate params ---
    if not isinstance(input_df, DataFrame):
        raise TypeError("input_df must be an instance of pyspark.sql.DataFrame")

    def run(df):
        validate_df(df)
        stages = (
            prepare_df,
            calculate_ratios,
            forward_impute_from_response,
            backward_impute,
            construct_values,
            forward_impute_from_construction,
        )

        for stage in stages:
            df = stage(df)
            if df.filter(col("output").isNull()).count() == 0:
                return create_output(df)

        if df.filter(col("output").isNull()).count() > 0:
            raise DataIntegrityError("Found null in output after imputation")

        return create_output(df)

    def validate_df(df):
        input_cols = set(df.columns)
        expected_cols = {
            reference_col,
            period_col,
            strata_col,
            target_col,
            auxiliary_col
        }
        if forward_link_col is not None:
            expected_cols.add(forward_link_col)

        if backward_link_col is not None:
            expected_cols.add(backward_link_col)

        # Check to see if the col names are not blank.
        for col_name in expected_cols:
            if not isinstance(col_name, str):
                msg = "All column names provided in params must be strings."
                raise TypeError(msg)

            if col_name == "":
                msg = "Column name strings provided in params must not be empty."
                raise ValueError(msg)

        # Check to see if any required columns are missing from dataframe.
        missing_cols = expected_cols - input_cols
        if missing_cols:
            msg = f"Missing columns: {', '.join(c for c in missing_cols)}"
            raise ValidationError(msg)

    def prepare_df(df):
        col_list = [
            col(period_col).alias("period"),
            col(strata_col).alias("strata"),
            col(target_col).alias("output"),
            col(auxiliary_col).alias("aux"),
            col(reference_col).alias("ref"),
        ]

        if forward_link_col is not None:
            col_list.append(col(forward_link_col).alias("forward"))

        if backward_link_col is not None:
            col_list.append(col(backward_link_col).alias("backward"))

        if marker_col in df.columns:
            col_list.append(col(marker_col).alias("marker"))

        return df.select(col_list)

    def reorder_df(df, order):
        # TODO: implement
        return df

    def calculate_previous_period(period):
        numeric_period = int(period)
        if period.endswith("01"):
            return str(numeric_period - 89)

        else:
            return str(numeric_period - 1)

    def calculate_ratios(df):
        ratio_df_list = []

        for strata_val in df.select("strata").distinct().toLocalIterator():
            strata_df = df.filter(df.strata == strata_val["strata"])
            strata_df_list = []
            for period_val in strata_df.select("period").distinct().toLocalIterator():
                period = period_val["period"]
                df_current_period = strata_df.filter(df.period == period)
                df_previous_period = strata_df.filter(
                    df.period == calculate_previous_period(period))
                if df_previous_period.count() == 0:
                    # No previous period so nothing to do.
                    continue

                working_df = df_current_period.join(
                    df_previous_period,
                    (df_current_period.ref == df_previous_period.ref,
                        df_current_period.strata == df_previous_period.strata),
                    'inner'
                ).select(
                    df_current_period.strata,
                    df_current_period.output,
                    df_previous_period.output.alias("other_output"))
                working_df = working_df.groupBy(working_df.strata).agg(
                    {'output': 'sum', 'other_output': 'sum'})
                working_df = working_df.withColumn(
                    "forward",
                    col("sum(output)")/when(
                        col("sum(other_output)") == 0,
                        1).otherwise(col("sum(other_output)")
                    )
                ).withColumn("period", lit(period))
                strata_df_list.append(working_df)

            strata_union_df = strata_df_list[0]
            for strata_part_df in strata_df_list[1:]:
                strata_union_df.unionAll(strata_part_df)

            strata_ratio_df = strata_union_df.sort(col("period").asc()
                ).withColumn("backward", when(
                    lead(col("forward")).isNull(), lit(None)).otherwise(
                    col(1/lead(col("forward"))))).alias("strata_ratio")

            ratio_df_list.append(strata_ratio_df)

        union_df = ratio_df_list[0]
        for part_df in ratio_df_list[1:]:
            union_df.unionAll(part_df)

        ret_df = df.join(
            union_df,
            (df.period == union_df.strata_ratio_period,
                df.strata == union_df.strata_ratio_strata),
            "inner"
        ).drop("strata_ratio_period", "strata_ratio_strata").withColumnRenamed(
            "strata_ratio_forward", "forward").withColumnRenamed(
            "strata_ratio_backward", "backward").fillna(1, ["forward", "backward"])
        return ret_df

    def remove_constructions(df):
        return df.select(
            df["period"],
            df["strata"],
            when(df["marker"].endswith("C"), lit(None))
            .otherwise(df["output"])
            .alias("output"),
            df["aux"],
            df["ref"],
            df["forward"],
            df["backward"],
            df["marker"],
        )

    def impute(df, link_col, marker):
        # TODO: imputation calculation
        return df

    def forward_impute_from_response(df):
        return impute(reorder_df(df, "asc"), "forward", "fir")

    def backward_impute(df):
        return impute(reorder_df(remove_constructions(df), "desc"), "backward", "bi")

    def construct_values(df):
        # TODO: construction calculation
        return df

    def forward_impute_from_construction(df):
        return impute(reorder_df(df, "asc"), "forward", "fic")

    def create_output(df):
        input_df.createOrReplaceTempView("input")
        df.createOrReplaceTempView("output")
        # TODO: select columns and join on input df
        return df

    # ----------

    return run(input_df)
