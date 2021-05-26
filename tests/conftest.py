import pytest
from pyspark.sql import SparkSession


@pytest.fixture(scope='session')
def fxt_spark_session():
    """
    Creates a Spark session to be used throughout all tests.
    """
    yield (SparkSession.builder
        .appName("tests")
        .master("local")
        .enableHiveSupport()
        .getOrCreate())
