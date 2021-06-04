import os
import pytest
from chispa.dataframe_comparer import assert_approx_df_equality
from pyspark.sql import SparkSession
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

@pytest.fixture
def fxt_load_test_csv(fxt_spark_session, filename):
    path = "tests/imputation/fixture_data/"
    filepath = os.path.join(path, filename)
    test_dataframe = fxt_spark_session.read.csv(filepath, header=True)
    select_col_list = []
    for dataframe_col in dataframe_columns:
        if dataframe_col in test_dataframe.columns:
            select_col_list.append(
                col(dataframe_col).cast(dataframe_types[dataframe_col])
            )

    yield test_dataframe.select(select_col_list)

@pytest.fixture(scope='session')
def fxt_spark_session():
    """
    Creates a Spark session to be used throughout all tests.
    """
    yield (
        SparkSession.builder
        .appName("tests")
        .master("local")
        .enableHiveSupport()
        .getOrCreate()
    )
