from chispa.schema_comparer import are_schemas_equal_ignore_nullable
from pyspark.sql import DataFrame
from pyspark.sql.functions import col
from pyspark.sql.types import StructField, StructType

from statistical_methods_library.utilities.exceptions import ValidationError


def validate_dataframe(input_df, expected_columns, type_mapping, excluded_columns=[]):

    if not isinstance(input_df, DataFrame):
        raise TypeError("input_df must be an instance of pyspark.sql.DataFrame")

    # Check to see if the column names have been passed in properly.
    for col_name in expected_columns.values():
        if not isinstance(col_name, str):
            raise TypeError("All column names provided in params must be strings.")

        if col_name == "":
            raise ValueError(
                "Column name strings provided in params must not be empty."
            )

    # Check to see if any required columns are missing from the dataframe.
    missing_columns = set(expected_columns.values()) - set(input_df.columns)
    if missing_columns:
        raise ValidationError(
            f"Missing columns: {', '.join(c for c in missing_columns)}"
        )

    column_list = []
    for alias, name in expected_columns.items():
        column_list.append(
            col(name).alias(alias),
        )
    aliased_df = input_df.select(column_list)

    column_type_list = []
    for alias in expected_columns:
        column_type_list.append(StructField(alias, type_mapping[alias]))

    schema = StructType(column_type_list)
    if not are_schemas_equal_ignore_nullable(aliased_df.schema, schema):
        raise Exception

    # Check to see if the columns contain null values.
    for col_name in expected_columns:
        if col_name in excluded_columns:
            continue

        if aliased_df.filter(col(col_name).isNull()).count() > 0:
            raise ValidationError(
                f"Input column {col_name} must not contain null values."
            )

    return aliased_df
