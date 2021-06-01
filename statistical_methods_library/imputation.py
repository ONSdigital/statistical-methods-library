from pyspark.sql import DataFrame
from pyspark.sql.functions import col, lit, when

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

    def calculate_next_period(period):
        numeric_period = int(period)
        if period.endswith("12"):
            return str(numeric_period + 89)
        else:
            return str(numeric_period + 1)

    def calculate_ratios(df):
        ratio_df_list = []
        # Since we're going to join on to the main df at the end filtering for
        # nulls won't cause us to lose strata as they'll just be filled with
        # default ratios.
        filtered_df = df.filter(~df.output.isNull())
        for strata_val in filtered_df.select("strata").distinct().toLocalIterator():
            strata_df = filtered_df.filter(df.strata == strata_val["strata"]
                ).select(
                    "ref",
                    "period",
                    "output"
                )
            period_df = strata_df.select('period').distinct()
            strata_forward_union_df = None
            for period_val in period_df.toLocalIterator():
                period = period_val["period"]
                df_current_period = strata_df.filter(strata_df.period == period).alias(
                    "current")
                df_previous_period = strata_df.filter(
                    strata_df.period == calculate_previous_period(period)
                ).alias("prev")
                if df_previous_period.count() == 0:
                    # No previous period so nothing to do.
                    continue

                # Put the values from the current and previous periods for a
                # contributor on the same row. Then calculate the sum for both
                # for all contributors in a period as the values now line up.
                working_df = df_current_period.join(
                    df_previous_period,
                    (col("current.ref") == col("prev.ref")),
                    'inner'
                ).select(
                    col("current.period").alias("period"),
                    col("current.output").alias("output"),
                    col("prev.output").alias("other_output"))
                working_df = working_df.groupBy(working_df.period).agg(
                    {'output': 'sum', 'other_output': 'sum'})

                # Calculate the forward ratio for every period using 1 in the
                # case of a 0 denominator.
                working_df = working_df.withColumn(
                    "forward",
                    col("sum(output)")/when(
                        col("sum(other_output)") == 0,
                        1).otherwise(col("sum(other_output)"))
                ).withColumn("period", lit(period))

                # Store the completed period.
                working_df = working_df.select("period", "forward")

                if strata_forward_union_df is None:
                    strata_forward_union_df = working_df

                else:
                    strata_forward_union_df = strata_forward_union_df.union(working_df)

            # Calculate backward ratio as 1/forward for the next period.
            strata_backward_union_df = None
            for period_val in period_df.toLocalIterator():
                period = period_val["period"]
                df_current_period = strata_forward_union_df.filter(
                    strata_forward_union_df.period == period)
                df_next_period = strata_forward_union_df.filter(
                    strata_forward_union_df.period == calculate_next_period(
                        period))
                if df_next_period.count() == 0:
                    # No next period so just add the default backward ratio.
                    working_df = df_current_period.withColumn("backward", lit(1))

                else:
                    working_df = df_current_period.withColumn(
                        "backward", 1/df_next_period.forward)

                if strata_backward_union_df is None:
                    strata_backward_union_df = working_df

                else:
                    strata_backward_union_df = strata_backward_union_df.union(
                        working_df)

            strata_joined_df = period_df.join(
                strata_backward_union_df,
                "period",
                "leftouter"
            ).select(
                period_df.period,
                strata_backward_union_df.forward,
                strata_backward_union_df.backward
            ).fillna(1, ["forward", "backward"])
            strata_ratio_df = strata_joined_df.withColumn(
                "strata",
                lit(strata_val["strata"]))
            # Store the completed ratios for this strata.
            ratio_df_list.append(strata_ratio_df)

        # Reassemble all the strata now we have ratios for them.
        ratio_df = ratio_df_list[0]
        for part_df in ratio_df_list[1:]:
            ratio_df.union(part_df)

        # Join the strata ratios onto the input such that each contributor has
        # a forward ratio. Also fill in any nulls with 1 so that imputation
        # behaves correctly without having to special-case for null values.
        ret_df = df.join(ratio_df, ["period", "strata"]).select(
            df.ref,
            df.period,
            df.strata,
            df.output,
            df.aux,
            ratio_df.forward,
            ratio_df.backward
        )
        return ret_df

    def remove_constructions(df):
        return df

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
