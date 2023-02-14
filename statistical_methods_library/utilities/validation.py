"""
For Copyright information, please see LICENCE.
"""

from pyspark.sql import DataFrame
from pyspark.sql.functions import col

from statistical_methods_library.utilities.exceptions import ValidationError


def validate_dataframe(
    input_df, expected_columns, type_mapping, unique_cols, excluded_columns=[]
):
    if not isinstance(input_df, DataFrame):
        raise TypeError("input_df must be an instance of pyspark.sql.DataFrame")
    expected_input_col_names = set(expected_columns.values())
    # Check to see if the column names have been passed in properly.
    for col_name in expected_input_col_names:
        if not isinstance(col_name, str):
            raise TypeError("All column names provided in params must be strings.")

        if not len(col_name):
            raise ValueError(
                "Column name strings provided in params must not be empty."
            )

    # Check to see if any required columns are missing from the dataframe.
    missing_columns = expected_input_col_names - set(input_df.columns)
    if missing_columns:
        raise ValidationError(
            f"Missing columns: {', '.join(c for c in missing_columns)}"
        )

    # Alias columns
    column_list = []
    for alias, name in expected_columns.items():
        column_list.append(
            col(name).alias(alias),
        )
    aliased_df = input_df.select(column_list)
    # Check that the set of aliased columns have the correct data types.
    wrong_types = [
        (field.name, field.dataType)
        for field in aliased_df.schema.fields
        if not isinstance(field.dataType, type_mapping[field.name])
    ]
    if wrong_types:
        raise ValidationError(
            "Wrong data types for columns: "
            + ", ".join(
                f"{expected_columns[c[0]]}: {c[1].typeName}" for c in wrong_types
            )
        )

    # Duplicate check
    if (
        aliased_df.select(*unique_cols).distinct().count()
        != aliased_df.select(*unique_cols).count()
    ):
        raise ValidationError("Duplicate contributors")

    # Check to see if the columns contain null values.
    for col_name in expected_columns:
        if col_name in excluded_columns:
            continue

        if aliased_df.filter(col(col_name).isNull()).count() > 0:
            raise ValidationError(
                f"Input column {col_name} must not contain null values."
            )

    return aliased_df


def validate_one_value_per_group(input_df, group_cols, value_col):
    if (
        input_df.select(*group_cols).distinct().count()
        != input_df.select(*group_cols, value_col).distinct().count()
    ):
        raise ValidationError(
            f"The {value_col} must be the same per " + " ".join(group_cols) + "."
        )


def validate_no_matching_rows(input_df, filter, error_message):
    if input_df.filter(filter).count() > 0:
        raise ValidationError(error_message)
