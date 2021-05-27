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
            forward_impute_from_response,
            backward_impute,
            construct_values,
            forward_impute_from_construction,
        )

        for stage in stages:
            df = stage(df)
            print('---------------')
            print(stage)
            print('---------------')
            df.show
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

        missing_cols = expected_cols - input_cols

        if missing_cols:
            raise ValidationError(
                f"Missing columns: {', '.join(c for c in missing_cols)}"
            )


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

    def build_links(df, link_col):
        if link_col in df.columns:
            return df

        # TODO: link calculation
        return df

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
        df = build_links(df, link_col)

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
