import pytest
import os

from pyspark.sql.functions import col

from statistical_methods_library import imputation

auxiliary_col = "auxiliary"
backward_col = "backward"
forward_col = "forward"
marker_col = "marker"
output_col = "output"
period_col = "period"
reference_col = "reference"
strata_col = "strata"
target_col = "target"
construction_col = "construction"

# Columns we expect in either our input or output test dataframes and their
# respective types
dataframe_columns = (
    reference_col,
    period_col,
    strata_col,
    target_col,
    auxiliary_col,
    output_col,
    marker_col,
    forward_col,
    backward_col,
    construction_col
)

dataframe_types = {
    reference_col: "string",
    period_col: "string",
    strata_col: "string",
    target_col: "double",
    auxiliary_col: "double",
    output_col: "double",
    marker_col: "string",
    backward_col: "double",
    forward_col: "double",
    construction_col: "double"
}

# Params used when calling imputation
params = (
    reference_col,
    period_col,
    strata_col,
    target_col,
    auxiliary_col,
    output_col,
    marker_col
)


def load_test_csv(spark_session, filename):
    path = "tests/imputation/fixture_data/"
    filepath = os.path.join(path, filename)
    test_dataframe = spark_session.read.csv(filepath, header=True)
    select_col_list = []
    for dataframe_col in dataframe_columns:
        if dataframe_col in test_dataframe.columns:
            select_col_list.append(
                col(dataframe_col).cast(dataframe_types[dataframe_col])
            )

    return test_dataframe.select(select_col_list)


# ====================================================================================
# --------------- TESTING TEMPLATE ---------------------------
# ====================================================================================
# --- Test type validation on the input dataframe(s) ---
# --- Test if cols missing from input dataframe(s) ---
# --- Test if any run-time params are null ---
# --- Test if output is a dataframe (or the expected type)---
# --- Test if output contents is as expected, both new columns and data content ---
# --- Test any other error based outputs ---

# IMPORTANT:
# 1) If the test contains any form of condition or loop, you must test the logical
#    branches to ensure that each assert is actually being performed.
# 2) Do not test internal structure of functions, it may be refactored. Stick
#    to the inputs and outputs.
# 3) Avoid referring to specific rows of test data where possible, they may change.
#    Instead, follow the existing templates to add conditional tests.
# 4) If you load the test data in for each test rather than as a module level
#    constant, you can amend data in the tests without needing new test data.
# 5) Don't test for python language errors. :)

# We're using double-quotes for strings since SQL requires single-quotes so
# this helps avoid escaping.

# ====================================================================================


# --- Test type validation on the input dataframe(s) ---

def test_dataframe_not_a_dataframe():
    with pytest.raises(TypeError):
        imputation.imputation('not_a_dataframe', *params)


# --- Test if cols missing from input dataframe(s) ---

def test_dataframe_column_missing(fxt_spark_session):
    test_dataframe = load_test_csv(fxt_spark_session, "test_basic_functionality.csv")
    bad_dataframe = test_dataframe.drop(strata_col)
    with pytest.raises(imputation.ValidationError):
        imputation.imputation(bad_dataframe, *params)


# --- Test if params null ---

def test_params_blank(fxt_spark_session):
    test_dataframe = load_test_csv(fxt_spark_session, "test_basic_functionality.csv")
    bad_params = (
        reference_col,
        period_col,
        strata_col,
        "",
        auxiliary_col,
        output_col,
        marker_col
    )
    with pytest.raises(ValueError):
        imputation.imputation(test_dataframe, *bad_params)


def test_params_not_string(fxt_spark_session):
    test_dataframe = load_test_csv(fxt_spark_session, "test_basic_functionality.csv")
    bad_params = (
        reference_col,
        period_col,
        strata_col,
        ["target_col"],
        auxiliary_col,
        output_col,
        marker_col
    )
    with pytest.raises(TypeError):
        imputation.imputation(test_dataframe, *bad_params)


# --- Test if output is a dataframe (or the expected type)---
# --- Test if output contents is as expected, both new columns and data content ---

def test_dataframe_returned(fxt_spark_session):
    test_dataframe = load_test_csv(fxt_spark_session, "test_basic_functionality.csv")
    ret_val = imputation.imputation(test_dataframe, *params)
    # perform action on the dataframe to trigger lazy evaluation
    _row_count = ret_val.count()
    assert isinstance(ret_val, type(test_dataframe))
    ret_cols = ret_val.columns
    assert "output" in ret_cols
    assert "marker" in ret_cols
