import pytest

from statistical_methods_library import imputation

auxiliary_col = "auxiliary"
marker_col = "marker"
output_col = "output"
period_col = "period"
reference_col = "reference"
strata_col = "strata"
target_col = "target"
params = (
    reference_col,
    period_col,
    strata_col,
    target_col,
    auxiliary_col,
    output_col,
    marker_col
)

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
# ====================================================================================


# --- Test type validation on the input dataframe(s) ---

def test_dataframe_not_a_dataframe():
    with pytest.raises(imputation.ValidationError):
        imputation.imputation('not_a_dataframe', *params)


# --- Test if cols missing from input dataframe(s) ---

def test_dataframe_column_missing(fxt_spark_session):
    test_dataframe = fxt_spark_session.read.csv(
        'tests/imputation/fixture_data/test_dataframe_returned.csv')
    bad_dataframe = test_dataframe.drop(strata_col)
    with pytest.raises(imputation.ValidationError):
        imputation.imputation(bad_dataframe, *params)


# --- Test if params null ---

def test_params_blank(fxt_spark_session):
    test_dataframe = fxt_spark_session.read.csv(
        'tests/imputation/fixture_data/test_dataframe_returned.csv')
    bad_params = (
        reference_col,
        period_col,
        strata_col,
        "",
        auxiliary_col,
        output_col,
        marker_col
    )
    with pytest.raises(imputation.ValidationError):
        imputation.imputation(test_dataframe, *bad_params)


# --- Test if output is a dataframe (or the expected type)---

def test_dataframe_returned(fxt_spark_session):
    test_dataframe = fxt_spark_session.read.csv(
        'tests/imputation/fixture_data/test_dataframe_returned.csv')
    ret_val = imputation.imputation(test_dataframe, *params)
    assert isinstance(ret_val, type(test_dataframe))
