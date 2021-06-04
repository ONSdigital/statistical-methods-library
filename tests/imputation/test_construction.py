from chispa.dataframe_comparer import assert_approx_df_equality

from statistical_methods_library import imputation

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

# --- Test if output is a dataframe (or the expected type)---
# --- Test if output contents is as expected, both new columns and data content ---

def test_imputed_values_as_expected(fxt_spark_session, capsys):
    test_dataframe = load_test_csv(fxt_spark_session,
                                   "test_construction_input.csv")
    exp_val = load_test_csv(fxt_spark_session,
                            "test_construction_output.csv")
    with capsys.disabled():
        ret_val = imputation.imputation(test_dataframe, *params)
        assert isinstance(ret_val, type(test_dataframe))
        sort_col_list = ["reference", "period"]
        assert_approx_df_equality(
            ret_val.sort(sort_col_list).select("output", "marker"),
            exp_val.sort(sort_col_list).select("output", "marker"),
            0.0001,
            ignore_nullable=True
        )

# --- Test any other error based outputs ---

# No error based outputs to test at this time
