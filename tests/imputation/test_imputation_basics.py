
from statistical_methods_library.imputation import imputation

# ====================================================================================
# --------------- TESTING TEMPLATE ---------------------------
# ====================================================================================
# --- Test fails with type error if no input ---
# --- Test type validation on the input dataframe(s) ---
# --- Test type validation on the target column lists(s) ---
# --- Test if cols missing from input dataframe(s) ---
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
# ====================================================================================


# noinspection PyMethodMayBeStatic
def test_dataframe_returned(fxt_spark_session):
    test_dataframe = fxt_spark_session.read.csv('data/sample_data.csv')
    ret_val = imputation(test_dataframe)
    assert isinstance(ret_val, type(test_dataframe))