import os
from pyspark.sql.functions import col

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
