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
# 3) If you load the test data in for each test rather than as a module level
#    constant, you can amend data in the tests without needing new test data.
# 4) Avoid referring to specific rows of test data where possible, they may change.
# 5) Don't test for python language errors. :)

# We're using double-quotes for strings since SQL requires single-quotes;  this helps
# avoid having to use escape characters.

# ====================================================================================


# --- Test if output is a dataframe (or the expected type)---
# --- Test if output contents is as expected, both new columns and data content ---

def test_dataframe_returned_as_expected(fxt_spark_session, capsys):
    test_dataframe = load_test_csv(fxt_spark_session, "test_ratio_calculation_input.csv")
    exp_val = load_test_csv(fxt_spark_session, "test_ratio_calculation_output.csv")
    with capsys.disabled():
        ret_val = imputation.imputation(test_dataframe, *params)
        # perform action on the dataframe to trigger lazy evaluation
        _row_count = ret_val.count()
        assert isinstance(ret_val, type(test_dataframe))
        ret_cols = ret_val.columns
        assert "forward" in ret_cols
        assert "backward" in ret_cols
        assert "construction" in ret_cols
        sort_col_list = ["reference", "period"]
        assert_approx_df_equality(
            ret_val.sort(sort_col_list).select("forward", "backward", "construction"),
            exp_val.sort(sort_col_list).select("forward", "backward", "construction"),
            0.0001,
            ignore_nullable=True
        )


# --- Test any other error based outputs ---

# No error based outputs to test at this time
