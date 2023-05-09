import pytest
from pyspark.sql.functions import lit
from pyspark.sql.types import DecimalType, LongType, StringType

from statistical_methods_library.imputation import impute, ratio_of_means
from statistical_methods_library.utilities.exceptions import ValidationError

auxiliary_col = "other"
backward_col = "backward"
forward_col = "forward"
marker_col = "marker"
output_col = "output"
period_col = "date"
reference_col = "identifier"
grouping_col = "group"
target_col = "question"
construction_col = "construction"
count_forward_col = "count_forward"
count_backward_col = "count_backward"
count_construction_col = "count_construction"

decimal_type = DecimalType(15, 6)

reference_type = StringType()
period_type = StringType()
grouping_type = StringType()
target_type = decimal_type
auxiliary_type = decimal_type
min_accuracy = decimal_type
marker_type = StringType()
backward_type = decimal_type
forward_type = decimal_type
construction_type = decimal_type
count_forward_type = LongType()
count_backward_type = LongType()
count_construction_type = LongType()

# Columns we expect in either our input or output test dataframes and their
# respective types
dataframe_columns = (
    reference_col,
    period_col,
    grouping_col,
    target_col,
    auxiliary_col,
    output_col,
    marker_col,
    forward_col,
    backward_col,
    construction_col,
    count_forward_col,
    count_backward_col,
    count_construction_col,
)

dataframe_types = {
    reference_col: reference_type,
    period_col: period_type,
    grouping_col: grouping_type,
    target_col: target_type,
    auxiliary_col: auxiliary_type,
    output_col: min_accuracy,
    marker_col: marker_type,
    backward_col: backward_type,
    forward_col: forward_type,
    construction_col: construction_type,
    count_forward_col: count_forward_type,
    count_backward_col: count_backward_type,
    count_construction_col: count_construction_type,
}

bad_dataframe_types = dataframe_types.copy()
bad_dataframe_types[target_col] = reference_type

# Params used when calling impute
params = {
    "reference_col": reference_col,
    "period_col": period_col,
    "grouping_col": grouping_col,
    "target_col": target_col,
    "auxiliary_col": auxiliary_col,
    "output_col": output_col,
    "marker_col": marker_col,
    "ratio_calculator": ratio_of_means,
}

test_scenarios = []

# --- Test type validation on the input dataframe(s) ---


def test_input_not_a_dataframe():
    with pytest.raises(TypeError):
        # noinspection PyTypeChecker
        impute(input_df="not_a_dataframe", **params)


# --- Test type validation on the back_data dataframe(s) ---


def test_back_data_not_a_dataframe(fxt_load_test_csv):
    test_dataframe = fxt_load_test_csv(
        dataframe_columns,
        dataframe_types,
        "imputation",
        "engine",
        "unit",
        "basic_functionality",
    )
    with pytest.raises(TypeError):
        # noinspection PyTypeChecker
        impute(input_df=test_dataframe, **params, back_data_df="not_a_dataframe")


# --- Test if cols missing from input dataframe(s) ---


def test_dataframe_column_missing(fxt_load_test_csv):
    test_dataframe = fxt_load_test_csv(
        dataframe_columns,
        dataframe_types,
        "imputation",
        "engine",
        "unit",
        "basic_functionality",
    )
    bad_dataframe = test_dataframe.drop(grouping_col)
    with pytest.raises(ValidationError):
        impute(input_df=bad_dataframe, **params)


# --- Test if dataframe has duplicate rows ---


def test_dataframe_duplicate_rows(fxt_load_test_csv):
    test_dataframe = fxt_load_test_csv(
        dataframe_columns,
        dataframe_types,
        "imputation",
        "engine",
        "unit",
        "duplicate_rows",
    )
    with pytest.raises(ValidationError):
        impute(input_df=test_dataframe, **params)


# --- Test if target missing from input dataframe(s) ---


def test_dataframe_target_missing(fxt_load_test_csv):
    test_dataframe = fxt_load_test_csv(
        dataframe_columns,
        dataframe_types,
        "imputation",
        "engine",
        "unit",
        "basic_functionality",
    )
    bad_dataframe = test_dataframe.drop(target_col)
    with pytest.raises(ValidationError):
        impute(input_df=bad_dataframe, **params)


# --- Test if params null ---


def test_params_none(fxt_load_test_csv):
    test_dataframe = fxt_load_test_csv(
        dataframe_columns,
        dataframe_types,
        "imputation",
        "engine",
        "unit",
        "basic_functionality",
    )
    bad_params = params.copy()
    bad_params["target_col"] = None
    with pytest.raises(TypeError):
        impute(input_df=test_dataframe, **bad_params)


def test_params_empty_string(fxt_load_test_csv):
    test_dataframe = fxt_load_test_csv(
        dataframe_columns,
        dataframe_types,
        "imputation",
        "engine",
        "unit",
        "basic_functionality",
    )
    bad_params = params.copy()
    bad_params["target_col"] = ""
    with pytest.raises(ValueError):
        impute(input_df=test_dataframe, **bad_params)


def test_params_missing_link_column(fxt_load_test_csv):
    test_dataframe = fxt_load_test_csv(
        dataframe_columns,
        dataframe_types,
        "imputation",
        "engine",
        "unit",
        "basic_functionality_partial_link_cols",
    )
    with pytest.raises(ValidationError):
        impute(input_df=test_dataframe, **params)


def test_params_not_string(fxt_load_test_csv):
    test_dataframe = fxt_load_test_csv(
        dataframe_columns,
        dataframe_types,
        "imputation",
        "engine",
        "unit",
        "basic_functionality",
    )
    bad_params = params.copy()
    bad_params["reference_col"] = 23
    with pytest.raises(TypeError):
        impute(input_df=test_dataframe, **bad_params)


# --- Test if output contents are as expected, both new columns and data ---


def test_dataframe_returned_as_expected(fxt_spark_session, fxt_load_test_csv):
    test_dataframe = fxt_load_test_csv(
        dataframe_columns,
        dataframe_types,
        "imputation",
        "engine",
        "unit",
        "basic_functionality",
    )
    # Make sure that no extra columns pass through.
    test_dataframe = test_dataframe.withColumn("bonus_column", lit(0))
    ret_val = impute(input_df=test_dataframe, **params)
    assert isinstance(ret_val, type(test_dataframe))
    ret_cols = ret_val.columns
    assert "bonus_column" not in ret_cols


# --- Test that when provided back data does not match input schema then fails ---
def test_back_data_missing_column(fxt_load_test_csv, fxt_spark_session):
    test_dataframe = fxt_load_test_csv(
        dataframe_columns,
        dataframe_types,
        "imputation",
        "engine",
        "unit",
        "basic_functionality",
    )
    bad_back_data = fxt_load_test_csv(
        dataframe_columns,
        dataframe_types,
        "imputation",
        "engine",
        "unit",
        "back_data_missing_column",
    )
    with pytest.raises(ValidationError):
        impute(input_df=test_dataframe, **params, back_data_df=bad_back_data)


def test_back_data_contains_nulls(fxt_load_test_csv, fxt_spark_session):
    test_dataframe = fxt_load_test_csv(
        dataframe_columns,
        dataframe_types,
        "imputation",
        "engine",
        "unit",
        "basic_functionality",
    )
    bad_back_data = fxt_load_test_csv(
        dataframe_columns,
        dataframe_types,
        "imputation",
        "engine",
        "unit",
        "back_data_nulls",
    )

    with pytest.raises(ValidationError):
        impute(input_df=test_dataframe, **params, back_data_df=bad_back_data)


def test_back_data_without_output_is_invalid(fxt_load_test_csv, fxt_spark_session):
    test_dataframe = fxt_load_test_csv(
        dataframe_columns,
        dataframe_types,
        "imputation",
        "engine",
        "unit",
        "basic_functionality",
    )
    bad_back_data = fxt_load_test_csv(
        [
            reference_col,
            period_col,
            grouping_col,
            target_col,
            marker_col,
            auxiliary_col,
        ],
        dataframe_types,
        "imputation",
        "engine",
        "unit",
        "back_data_no_output",
    )

    with pytest.raises(ValidationError):
        impute(input_df=test_dataframe, **params, back_data_df=bad_back_data)


# Test if when the back data input has link cols and the main data input does not
# then the columns are ignored.
def test_back_data_drops_link_cols_when_present(fxt_load_test_csv, fxt_spark_session):
    test_dataframe = fxt_load_test_csv(
        dataframe_columns,
        dataframe_types,
        "imputation",
        "engine",
        "unit",
        "basic_functionality",
    )

    back_data = fxt_load_test_csv(
        dataframe_columns,
        dataframe_types,
        "imputation",
        "engine",
        "unit",
        "back_data_with_link_cols",
    )

    ret_val = impute(input_df=test_dataframe, **params, back_data_df=back_data)

    assert ret_val.count() == 1


# Test when main data input has link cols and the back data input does not
# then columns aren't lost.
def test_input_has_link_cols_and_back_data_does_not_have_link_cols(
    fxt_load_test_csv, fxt_spark_session
):
    test_dataframe = fxt_load_test_csv(
        dataframe_columns,
        dataframe_types,
        "imputation",
        "engine",
        "unit",
        "basic_functionality_with_link_cols",
    )

    back_data = fxt_load_test_csv(
        dataframe_columns,
        dataframe_types,
        "imputation",
        "engine",
        "unit",
        "back_data_without_link_cols",
    )

    imputation_kwargs = params.copy()
    imputation_kwargs.update(
        {
            "forward_link_col": forward_col,
            "backward_link_col": backward_col,
            "construction_link_col": construction_col,
            "input_df": test_dataframe,
            "back_data_df": back_data,
        }
    )

    ret_val = impute(**imputation_kwargs)

    assert ret_val.count() == 1


# Test if columns of the incorrect type are caught.
def test_incorrect_column_types(fxt_load_test_csv):
    test_dataframe = fxt_load_test_csv(
        dataframe_columns,
        bad_dataframe_types,
        "imputation",
        "engine",
        "unit",
        "basic_functionality",
    )
    with pytest.raises(ValidationError):
        impute(input_df=test_dataframe, **params)


def test_input_data_contains_nulls(fxt_load_test_csv, fxt_spark_session):
    test_dataframe = fxt_load_test_csv(
        dataframe_columns,
        dataframe_types,
        "imputation",
        "engine",
        "unit",
        "input_data_nulls",
    )

    with pytest.raises(ValidationError):
        impute(input_df=test_dataframe, **params)
