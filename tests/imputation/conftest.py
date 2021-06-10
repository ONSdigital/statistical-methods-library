import os

import pytest
from pyspark.sql import SparkSession
from pyspark.sql.functions import col


@pytest.fixture
def fxt_load_test_csv(fxt_spark_session):
    # We don't want to yield the function because we're explicitly using it to
    # be called as a function in our tests and thus this fixture would have no
    # access to its return value. We do this so that our loader can be
    # passed a file name.
    def load(columns, types, filename):
        path = "tests/imputation/fixture_data/"
        filepath = os.path.join(path, filename)
        test_dataframe = fxt_spark_session.read.csv(filepath, header=True)
        select_col_list = []
        for dataframe_col in columns:
            if dataframe_col in test_dataframe.columns:
                select_col_list.append(col(dataframe_col).cast(types[dataframe_col]))

        return test_dataframe.select(select_col_list)

    return load

# We want to isolate the session per tests to prevent interactions
@pytest.fixture(scope="function")
def fxt_spark_session():
    """
    Creates a Spark session to be used throughout all tests.
    """
    session = (
        SparkSession.builder.appName("tests")
        .master("local[*]")
        .enableHiveSupport()
        .getOrCreate()
    )
    yield session
    session.stop()
