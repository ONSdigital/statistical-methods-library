"""
Perform imputation on a data frame.

Currently only Ratio of Means (or Ratio of Sums) imputation is implemented.
"""

import typing
from enum import Enum

from pyspark.sql import Column, DataFrame
from pyspark.sql.functions import col, lit, when

# --- Marker constants ---
# Documented after the variable as per Pdoc syntax for documenting variables.


class Marker(Enum):
    """Values for the marker column created during imputation."""

    RESPONSE = "R"
    """The value is a response."""

    FORWARD_IMPUTE_FROM_RESPONSE = "FIR"
    """The value has been forward imputed from a response."""

    BACKWARD_IMPUTE = "BI"
    """The value has been backward imputed from a response,
    backward imputation from construction is not permitted."""

    CONSTRUCTED = "C"
    """The value is constructed."""

    FORWARD_IMPUTE_FROM_CONSTRUCTION = "FIC"
    """The value has been forward imputed from a constructed value."""


# --- Imputation errors ---


class ImputationError(Exception):
    """Base type for imputation errors"""

    pass


class ValidationError(ImputationError):
    """Error raised by dataframe validation"""

    pass


class DataIntegrityError(ImputationError):
    """Error raised when imputation has failed to impute for data integrity
    reasons (currently when the auxiliary column contains nulls)"""

    pass


def impute(
    input_df: DataFrame,
    reference_col: str,
    period_col: str,
    strata_col: str,
    target_col: str,
    auxiliary_col: str,
    output_col: str,
    marker_col: str,
    forward_link_col: typing.Optional[str] = None,
    backward_link_col: typing.Optional[str] = None,
    construction_link_col: typing.Optional[str] = None,
    back_data_df: typing.Optional[DataFrame] = None,
) -> DataFrame:
    """
    Perform Ratio of means (also known as Ratio of Sums) imputation on a
    dataframe.

    ###Arguments
    * `input_df`: The input dataframe
    * `reference_col`: The name of the column to reference a unique
      contributor
    * `period_col`: The name of the column containing the period
      information for the contributor
    * `strata_col`: The Name of the column containing the strata information
      for the contributor
    * `target_col`: The name of the column containing the target
      variable
    * `auxiliary_col`: The name of the column containing the auxiliary
      variable
    * `output_col`: The name of the column which will contain the
      output
    * `marker_col`: The name of the column which will contain the marker
      information for a given value
    * `forward_link_col`: If specified, the name of an existing column
      containing forward ratio (or link) information
      Defaults to None which means that a default column name of "forward" will
      be created and the forward ratios will be calculated
    * `backward_link_col`: If specified, the name of an existing column
      containing backward ratio (or link) information
      Defaults to None which means that a default column name of "backward"
      will be created and the backward ratios will be calculated
    * `construction_link_col`: If specified, the name of an existing column
      containing construction ratio (or link) information
      Defaults to None which means that a default column name of "construction"
      will be created and the construction ratios will be calculated.
    * `back_data_df`: If specified, will use this to base the initial imputation
      calculations on.

    ###Returns
    A new dataframe containing:

    * `reference_col`
    * `period_col`
    * `output_col`
    * `marker_col`
    * `forward_col`
    * `backward_col`
    * `construction_col`

    No other columns are created. In particular, no other columns
    will be passed through from the input since it is expected that the
    information in the output dataframe will be sufficient to join on any
    other required input data.

    ###Notes
    If no Imputation needs to take place, the forward, backward and
    construction columns returned will still be calculated if they are not
    passed in.

    The existence of `output_col` and `marker_col` in the input data is
    an error.

    All or none of `forward_link_col`, `backward_link_col` and
    `construction_link_col` must be specified.

    `marker_col` will contain one of the marker constants defined in the
    `Marker` enum.

    This method implements rolling imputation, that is imputed values chain
    together until either a return is present or the contributor is not present
    in the sample. Values either side of such a gap do not interact (i.e.
    ratios and imputes will not take into account values for a contributor
    from before or after periods where they were dropped from the sample). In
    the case of rolling imputation, the markers will be the same for chains of
    imputed values.

    If `back_data_df` is provided it must contain the same columns as the output
    from this function.
    """
    # --- Validate params ---
    if not isinstance(input_df, DataFrame):
        raise TypeError("input_df must be an instance of pyspark.sql.DataFrame")

    # Mapping of column aliases to parameters
    full_col_mapping = {
        "ref": reference_col,
        "period": period_col,
        "output": output_col,
        "marker": marker_col,
        "target": target_col,
        "strata": strata_col,
        "aux": auxiliary_col,
    }

    if forward_link_col is None:
        full_col_mapping.update(
            {
                "forward": "forward",
                "backward": "backward",
                "construction": "construction",
            }
        )

    else:
        full_col_mapping.update(
            {
                "forward": forward_link_col,
                "backward": backward_link_col,
                "construction": construction_link_col,
            }
        )

    # --- Run ---
    def run() -> DataFrame:
        validate_df(input_df)
        if back_data_df:
            validate_df(back_data_df, allow_nulls=False, back_data=True)

        stages = (
            prepare_df(back_data_df),
            forward_impute_from_response,
            backward_impute,
            construct_values,
            forward_impute_from_construction,
        )
        df = input_df
        for stage in stages:
            df = stage(df).localCheckpoint(eager=False)
            if df.filter(col("output").isNull()).count() == 0:
                break

        return create_output(df)

    # --- Validate DF ---
    def validate_df(
        df: DataFrame, allow_nulls: bool = True, back_data: bool = False
    ) -> None:
        input_cols = set(df.columns)
        expected_cols = {
            reference_col,
            period_col,
            strata_col,
            auxiliary_col,
        }

        if back_data:
            expected_cols.add(marker_col)
            expected_cols.add(output_col)
        else:
            expected_cols.add(target_col)

        link_cols = [
            link_col is not None
            for link_col in [forward_link_col, backward_link_col, construction_link_col]
        ]

        if any(link_cols) and not all(link_cols):
            raise TypeError("Either all or no link columns must be specified")

        if forward_link_col is not None:
            expected_cols.add(forward_link_col)
            expected_cols.add(backward_link_col)
            expected_cols.add(construction_link_col)

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

        # Check columns types of the input dataframes.
        expected_types = {
            reference_col: "string",
            period_col: "string",
            strata_col: "string",
            target_col: "double",
            auxiliary_col: "double",
            output_col: "double",
            marker_col: "string",
            forward_link_col: "double",
            backward_link_col: "double",
            construction_link_col: "double",
        }

        set_types = {a for a in expected_types.items()}
        incorrect_types = [
            t for t in set(df.dtypes) - set_types if t[0] in expected_types
        ]

        if incorrect_types:
            msg = '\n'.join(
                "Column %s of type %s must be of type %s."
                % (t[0], t[1], expected_types[t[0]])
                for t in incorrect_types
            )
            raise ValidationError(msg)

        if not allow_nulls:
            for col_name in expected_cols:
                if df.filter(col(col_name).isNull()).count() > 0:
                    msg = f"Column {col_name} must not contain nulls"
                    raise ValidationError(msg)

    # Cache the prepared back data df since we'll need a few differently
    # filtered versions
    prepared_back_data_df = None

    # --- Prepare DF ---

    def prepare_df(
        back_data_df: typing.Optional[DataFrame],
    ) -> typing.Callable[[DataFrame], DataFrame]:
        def prepare(df: DataFrame) -> DataFrame:
            prepared_df = (
                select_cols(df).withColumn("output", col("target")).drop("target")
            )
            prepared_df = (
                prepared_df.withColumn(
                    "marker", when(~col("output").isNull(), Marker.RESPONSE.value)
                )
                .withColumn("previous_period", calculate_previous_period(col("period")))
                .withColumn("next_period", calculate_next_period(col("period")))
            )

            nonlocal prepared_back_data_df
            if back_data_df:
                prepared_back_data_df = (
                    select_cols(
                        back_data_df.join(
                            prepared_df.selectExpr("min(previous_period)"),
                            [col(period_col) == col("min(previous_period)")],
                            "inner",
                        ).filter(col(marker_col) != lit(Marker.BACKWARD_IMPUTE.value))
                    )
                    .drop("target")
                    .withColumn(
                        "previous_period", calculate_previous_period(col("period"))
                    )
                    .withColumn("next_period", calculate_next_period(col("period")))
                )
            else:
                # Set the prepared_back_data_df to be empty when back_data not
                # supplied.
                prepared_back_data_df = prepared_df.filter(col(period_col).isNull())

            prepared_back_data_df = prepared_back_data_df.persist()

            # Ratio calculation needs all the responses from the back data
            prepared_df = prepared_df.unionByName(
                prepared_back_data_df.filter(
                    col("marker") == lit(Marker.RESPONSE.value)
                )
            )

            return calculate_ratios(prepared_df)

        return prepare

    # --- Calculate Ratios ---

    def calculate_ratios(df: DataFrame) -> DataFrame:
        if "forward" in df.columns:
            df = df.fillna(1.0, ["forward", "backward", "construction"])
            return df

        # Since we're going to join on to the main df at the end filtering for
        # nulls won't cause us to lose strata as they'll just be filled with
        # default ratios.
        filtered_df = df.filter(~df.output.isNull()).select(
            "ref",
            "period",
            "strata",
            "output",
            "aux",
            "previous_period",
            "next_period",
        )

        # Put the values from the current and previous periods for a
        # contributor on the same row. Then calculate the sum for both
        # for all contributors in a period as the values now line up.
        working_df = filtered_df.alias("current")
        working_df = working_df.join(
            filtered_df.select("ref", "period", "output", "strata").alias("prev"),
            [
                col("current.ref") == col("prev.ref"),
                col("current.previous_period") == col("prev.period"),
                col("current.strata") == col("prev.strata"),
            ],
            "leftouter",
        ).select(
            col("current.strata").alias("strata"),
            col("current.period").alias("period"),
            when(~col("prev.output").isNull(), col("current.output")).alias("output"),
            col("current.aux").alias("aux"),
            col("prev.output").alias("other_output"),
            col("current.output").alias("output_for_construction"),
        )

        working_df = working_df.groupBy("period", "strata").agg(
            {
                "output": "sum",
                "other_output": "sum",
                "aux": "sum",
                "output_for_construction": "sum",
            }
        )
        # Calculate the forward ratio for every period using 1 as the link in
        # the case of a 0 denominator. We also calculate construction
        # links at the same time for efficiency reasons. This shares
        # the same behaviour of defaulting to a 1 in the case of a 0
        # denominator.
        forward_df = (
            working_df.withColumn(
                "forward", col("sum(output)") / col("sum(other_output)")
            )
            .withColumn(
                "construction",
                col("sum(output_for_construction)") / col("sum(aux)"),
            )
            .join(
                filtered_df.select("period", "strata", "next_period").distinct(),
                ["period", "strata"],
            )
        )

        # Calculate backward ratio as 1/forward for the next period for each
        # strata.
        strata_ratio_df = forward_df.join(
            forward_df.select(
                col("period").alias("other_period"),
                col("forward").alias("next_forward"),
                col("strata").alias("other_strata"),
            ),
            [
                col("next_period") == col("other_period"),
                col("strata") == col("other_strata"),
            ],
            "leftouter",
        ).select(
            col("period"),
            col("strata"),
            col("forward"),
            (lit(1.0) / col("next_forward")).alias("backward"),
            col("construction"),
        )

        # Join the strata ratios onto the input such that each contributor has
        # a set of ratios.
        return df.join(strata_ratio_df, ["period", "strata"], "leftouter").fillna(
            1.0, ["forward", "backward", "construction"]
        )

    # Caching for both imputed and unimputed data.
    imputed_df = None
    null_response_df = None

    # --- Impute helper ---
    def impute_helper(
        df: DataFrame, link_col: str, marker: Marker, direction: bool
    ) -> DataFrame:
        nonlocal imputed_df
        nonlocal null_response_df
        if direction:
            # Forward imputation
            other_period_col = "previous_period"
        else:
            # Backward imputation
            other_period_col = "next_period"

        if imputed_df is None:
            working_df = df.select(
                "ref",
                "period",
                "strata",
                "output",
                "marker",
                "previous_period",
                "next_period",
                "forward",
                "backward",
            )

            # Anything which isn't null is already imputed or a response and thus
            # can be imputed from. Note that in the case of backward imputation
            # this still holds since it always happens after forward imputation
            # and thus it can never attempt to backward impute from a forward
            # imputation since there will never be a null value directly prior to
            # one.
            imputed_df = working_df.filter(~col("output").isNull()).localCheckpoint(
                eager=True
            )
            # Any refs which have no values at all can't be imputed from so we
            # don't care about them here.
            ref_df = imputed_df.select("ref").distinct()
            null_response_df = (
                working_df.filter(col("output").isNull())
                .drop("output", "marker")
                .join(ref_df, "ref")
                .localCheckpoint(eager=True)
            )

        while True:
            other_df = imputed_df.selectExpr(
                "ref AS other_ref",
                "period AS other_period",
                "output AS other_output",
                "strata AS other_strata",
            )
            calculation_df = (
                null_response_df.join(
                    other_df,
                    [
                        col(other_period_col) == col("other_period"),
                        col("ref") == col("other_ref"),
                        col("strata") == col("other_strata"),
                    ],
                )
                .select(
                    "ref",
                    "period",
                    "strata",
                    (col(link_col) * col("other_output")).alias("output"),
                    lit(marker.value).alias("marker"),
                    "previous_period",
                    "next_period",
                    "forward",
                    "backward",
                )
                .localCheckpoint(eager=False)
            )
            # If we've imputed nothing then we've got as far as we can get for
            # this phase.
            if calculation_df.count() == 0:
                break

            # Store this set of imputed values in our main set for the next
            # iteration. Use eager checkpoints to help prevent rdd DAG explosion.
            imputed_df = imputed_df.union(calculation_df).localCheckpoint(eager=True)
            # Remove the newly imputed rows from our filtered set.
            null_response_df = null_response_df.join(
                calculation_df.select("ref", "period"), ["ref", "period"], "leftanti"
            ).localCheckpoint(eager=True)

        # We should now have an output column which is as fully populated as
        # this phase of imputation can manage. As such replace the existing
        # output column with our one. Same goes for the marker column.
        return df.drop("output", "marker").join(
            imputed_df.select("ref", "period", "output", "marker"),
            ["ref", "period"],
            "leftouter",
        )

    # --- Imputation functions ---
    def forward_impute_from_response(df: DataFrame) -> DataFrame:
        # Add the forward imputes from responses from the back data
        df = df.unionByName(
            prepared_back_data_df.filter(
                col("marker") == lit(Marker.FORWARD_IMPUTE_FROM_RESPONSE.value)
            ),
            True,
        )
        return impute_helper(df, "forward", Marker.FORWARD_IMPUTE_FROM_RESPONSE, True)

    def backward_impute(df: DataFrame) -> DataFrame:
        return impute_helper(df, "backward", Marker.BACKWARD_IMPUTE, False)

    # --- Construction functions ---
    def construct_values(df: DataFrame) -> DataFrame:
        # Add in the constructions and forward imputes from construction in the back data
        df = df.unionByName(
            prepared_back_data_df.filter(
                (
                    (col("marker") == lit(Marker.CONSTRUCTED.value))
                    | (
                        col("marker")
                        == lit(Marker.FORWARD_IMPUTE_FROM_CONSTRUCTION.value)
                    )
                )
            ),
            True,
        )
        construction_df = df.filter(df.output.isNull()).select(
            "ref", "period", "strata", "aux", "construction", "previous_period"
        )
        other_df = df.select("ref", "period", "strata").alias("other")
        construction_df = construction_df.alias("construction")
        construction_df = construction_df.join(
            other_df,
            [
                col("construction.ref") == col("other.ref"),
                col("construction.previous_period") == col("other.period"),
                col("construction.strata") == col("other.strata"),
            ],
            "leftanti",
        ).select(
            col("construction.ref").alias("ref"),
            col("construction.period").alias("period"),
            (col("aux") * col("construction")).alias("constructed_output"),
            lit(Marker.CONSTRUCTED.value).alias("constructed_marker"),
        )

        return (
            df.withColumnRenamed("output", "existing_output")
            .withColumnRenamed("marker", "existing_marker")
            .join(
                construction_df,
                ["ref", "period"],
                "leftouter",
            )
            .select(
                "*",
                when(col("existing_output").isNull(), col("constructed_output"))
                .otherwise(col("existing_output"))
                .alias("output"),
                when(col("existing_marker").isNull(), col("constructed_marker"))
                .otherwise(col("existing_marker"))
                .alias("marker"),
            )
            .drop("existing_output", "constructed_output", "constructed_marker")
        )

    def forward_impute_from_construction(df: DataFrame) -> DataFrame:
        # We need to recalculate our imputed and null response data frames to
        # account for construction.
        nonlocal imputed_df
        nonlocal null_response_df
        imputed_df = None
        null_response_df = None
        return impute_helper(
            df, "forward", Marker.FORWARD_IMPUTE_FROM_CONSTRUCTION, True
        )

    # --- Utility functions ---
    def create_output(df: DataFrame) -> DataFrame:
        return select_cols(
            df.join(prepared_back_data_df, ["period"], "leftanti"), reversed=False
        ).withColumnRenamed("output", output_col)

    def select_cols(df: DataFrame, reversed: bool = True) -> DataFrame:
        col_mapping = (
            {v: k for k, v in full_col_mapping.items()}
            if reversed
            else full_col_mapping
        )

        return df.select(
            [
                col(k).alias(col_mapping[k])
                for k in set(col_mapping.keys()) & set(df.columns)
            ]
        )

    def calculate_previous_period(period: Column) -> Column:
        return when(
            period.endswith("01"), (period.cast("int") - 89).cast("string")
        ).otherwise((period.cast("int") - 1).cast("string"))

    def calculate_next_period(period: Column) -> Column:
        return when(
            period.endswith("12"), (period.cast("int") + 89).cast("string")
        ).otherwise((period.cast("int") + 1).cast("string"))

    # ----------

    return run()
