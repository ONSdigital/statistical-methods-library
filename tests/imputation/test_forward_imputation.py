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

# We're using double-quotes for strings since SQL requires single-quotes so
# this helps avoid escaping.

# ====================================================================================