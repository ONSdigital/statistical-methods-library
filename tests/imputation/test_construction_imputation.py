import os

from pyspark.sql.functions import col
from chispa.dataframe_comparer import assert_approx_df_equality

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

# --- Test if output is a dataframe (or the expected type)---
# --- Test if output contents is as expected, both new columns and data content ---

def test_imputed_values_as_expected(fxt_spark_session, capsys):
    test_dataframe = load_test_csv(fxt_spark_session,
                                   "test_construction_imputation_input.csv")
    exp_val = load_test_csv(fxt_spark_session,
                            "test_construction_imputation_output.csv")
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
