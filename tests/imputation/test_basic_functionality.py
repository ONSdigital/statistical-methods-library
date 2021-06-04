import pytest
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

def test_dataframe_column_missing(fxt_spark_session, fxt_load_test_csv):
    test_dataframe = fxt_load_test_csv("test_basic_functionality.csv")
    bad_dataframe = test_dataframe.drop(strata_col)
    with pytest.raises(imputation.ValidationError):
        imputation.imputation(bad_dataframe, *params)


# --- Test if params null ---

def test_params_blank(fxt_spark_session, fxt_load_test_csv):
    test_dataframe = fxt_load_test_csv("test_basic_functionality.csv")
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


def test_params_not_string(fxt_spark_session, fxt_load_test_csv):
    test_dataframe = fxt_load_test_csv("test_basic_functionality.csv")
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

def test_dataframe_returned(fxt_spark_session, fxt_load_test_csv):
    test_dataframe = fxt_load_test_csv("test_basic_functionality.csv")
    ret_val = imputation.imputation(test_dataframe, *params)
    # perform action on the dataframe to trigger lazy evaluation
    _row_count = ret_val.count()
    assert isinstance(ret_val, type(test_dataframe))
    ret_cols = ret_val.columns
    assert "output" in ret_cols
    assert "marker" in ret_cols
