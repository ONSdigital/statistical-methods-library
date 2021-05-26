from pyspark.sql.functions import col, lit, when


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
    backward_link_col=None,
    construction_filter="true"
):

    def run(df):
        stages = (
            validate_df,
            prepare_df,
            forward_impute_from_response,
            backward_impute,
            construct_values,
            forward_impute_from_construction,
        )

        for stage in stages:
            if df.filter("output IS NULL").count() == 0:
                return create_output(df)

            df = stage(df)

        if df.filter("output IS NULL").count() > 0:
            raise RuntimeError("Found null in output after imputation")

        return create_output(df)

    def validate_df(df):
        if df.filter(col(auxiliary_col).isNull()).count() > 0:
            raise ValueError(
                f"Auxiliary column {auxiliary_col} contains null values"
            )

        return df

    def prepare_df(df):
        forward_col = None  # TODO - implement
        backward_col = None  # TODO - implement

        col_list = [
            col(period_col).alias("period"),
            col(strata_col).alias("strata"),
            col(target_col).alias("output"),
            col(auxiliary_col).alias("aux"),
            col(reference_col).alias("ref"),
        ]

        if forward_col is not None:
            col_list.append(col(forward_col).alias("forward"))

        if backward_col is not None:
            col_list.append(col(backward_col).alias("backward"))

        if marker_col in df:
            col_list.append(col(marker_col).alias("marker"))

        return df.select(col_list)

    def reorder_df(df, order):
        # TODO: implement
        return df

    def build_links(df, link_col):
        if link_col in df:
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

    return run(df)
