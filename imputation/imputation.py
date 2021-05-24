from pyspark.sql.functions import col, when

def impute_df(
    df,
    reference_column,
    period_column,
    strata_column,
    target_column,
    auxiliary_column,
    output_column,
    marker_column,
    forward_link_column=None,
    backward_link_column=None,
    construction_filter="true"
):
    input_df = df

    def validate_df(df):
        if df.filter(col(auxiliary_column) == lit(None)).count() > 0:
            raise ValueError(
                f"Auxiliary column {auxiliary_column} contains null values")

        return df

    def prepare_df(df):
        col_list = [
            col(period_column).alias("period"),
            col(strata_column).alias("strata"),
            col(target_column).alias("output"),
            col(auxiliary_column).alias("aux"),
            col(reference_column).alias("ref")
        ]

        if forward_column is not None:
            col_list.append(col(forward_column).alias("forward"))

        if backward_column is not None:
            col_list.append(col(backward_column).alias("backward"))

        if marker_column in df:
            col_list.append(col(marker_column).alias("marker"))

        return df.select(col_list)

    def reorder_df(df, order):
        # TODO: implement
        return df

    def build_links(df, link_column):
        if link_column in df:
            return df

        # TODO: link calculation
        return df

    def remove_constructions(df):
        return df.select(
            df["period"],
            df["strata"],
            when(df["marker"].endswith("C"), df["output"]).alias("output"),
            df["aux"],
            df["ref"],
            df["forward"],
            df["backward"],
            df["marker"]
        )


    def impute(df, link_column, marker):
        df = build_links(df, link_column)

        # TODO: imputation calculation
        return df

    def forward_impute_from_response(df):
        return impute(reorder_df(df, "asc"), "forward", "fir")

    def backward_impute(df):
        return impute(
            reorder_df(remove_constructions(df),
            "desc"),
            "backward",
            "bi")

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

    stages = (
        validate_df,
        prepare_df,
        forward_impute_from_response,
        backward_impute,
        construct_values,
        forward_impute_from_construction
    )

    for stage in stages:
        if col_not_null(df, "output"):
            return create_output(df)

        df = stage(df)

    if not col_not_null(df, "output"):
        raise RuntimeError("Found null in output after imputation")

    return create_output(df)
