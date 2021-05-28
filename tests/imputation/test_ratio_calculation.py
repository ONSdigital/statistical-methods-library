import pytest
import os

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


def load_test_csv(spark_session, filename):
    path = "tests/imputation/fixture_data/"
    filepath = os.path.join(path, filename)
    test_dataframe = spark_session.read.csv(filepath, header=True)
    return test_dataframe


# ====================================================================================
# --------------- TESTING TEMPLATE ---------------------------
# ====================================================================================

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


# --- Test if output is a dataframe (or the expected type)---

def test_dataframe_returned(fxt_spark_session):
    test_dataframe = load_test_csv(fxt_spark_session, "test_ratio_calculation.csv")
    ret_val = imputation.imputation(test_dataframe, *params)
    # perform action on the dataframe to trigger lazy evaluation
    _row_count = ret_val.count()
    assert isinstance(ret_val, type(test_dataframe))


# --- Test if output contents is as expected, both new columns and data content ---

def test_new_columns_created(fxt_spark_session):
    test_dataframe = load_test_csv(fxt_spark_session, "test_ratio_calculation.csv")
    ret_val = imputation.imputation(test_dataframe, *params)
    # perform action on the dataframe to trigger lazy evaluation
    _row_count = ret_val.count()
    ret_cols = ret_val.columns
    assert "forward" in ret_cols
    assert "backward" in ret_cols


def test_ratios_as_expected(fxt_spark_session, capsys):
    test_dataframe = load_test_csv(fxt_spark_session, "test_ratio_calculation.csv")
    ret_val = imputation.imputation(test_dataframe, *params)
    # perform action on the dataframe to trigger lazy evaluation
    assert ret_val.count() > 0
    # Disable output capturing so we can see the contents of the returned
    # dataframe
    with capsys.disabled():
        ret_val.show()
