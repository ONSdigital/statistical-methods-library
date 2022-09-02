from chispa.schema_comparer import are_schemas_equal_ignore_nullable
from pyspark.sql.functions import col
from pyspark.sql.types import StructField, StructType
from util.exceptions import ValidationError


def validate_dataframe(input_df, expected_columns, alias_mapping, type_mapping):

    # Check to see if any required columns are missing from the dataframe.
    missing_columns = set(expected_columns.values()) - set(input_df.columns)
    if missing_columns:
        raise ValidationError(
            f"Missing columns: {', '.join(c for c in missing_columns)}"
        )

    column_list = []
    for name in expected_columns:
        column_list.append(
            col(expected_columns[name]).alias(alias_mapping[name]),
        )
    aliased_df = input_df.select(column_list)

    column_type_list = []
    for name in expected_columns:
        alias = alias_mapping[name]
        column_type_list.append(StructField(alias, type_mapping[alias]))

    schema = StructType(column_type_list)
    if not are_schemas_equal_ignore_nullable(aliased_df.schema, schema):
        raise Exception
