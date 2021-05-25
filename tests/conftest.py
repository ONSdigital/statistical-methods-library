import pytest
from pyspark.sql import SparkSession


@pytest.fixture(scope='session')
def spark_test_session():
    """
    Creates a Spark session to be used throughout all tests.
    """
    return (SparkSession.builder
        .appName("tests")
        .master("local")
        .enableHiveSupport()
        .getOrCreate()
    )
