import glob
import os
import pathlib

import pytest
from chispa import assert_approx_df_equality
from pyspark.sql.functions import lit

from statistical_methods_library import winsorisation

reference_col = "ref"
period_col = "period"
grouping_col = "grouping"
auxiliary_col = "auxiliary"
design_weight_col = "design_weight"
calibration_weight_col = "calibration_weight"
l_value_col = "l_value"
target_col = "target"
outlier_weight_col = "outlier_weight"


dataframe_columns = (
    reference_col,
    period_col,
    grouping_col,
    auxiliary_col,
    design_weight_col,
    calibration_weight_col,
    l_value_col,
    target_col,
    outlier_weight_col,
)

dataframe_types = {
    reference_col: "string",
    period_col: "string",
    grouping_col: "string",
    auxiliary_col: "double",
    design_weight_col: "double",
    calibration_weight_col: "double",
    l_value_col: "double",
    target_col: "double",
    outlier_weight_col: "double",
}

params = (
    reference_col,
    period_col,
    grouping_col,
    target_col,
    design_weight_col,
    l_value_col,
    outlier_weight_col,
)

test_scenarios = []

for scenario_category in ("dev", "methodology"):
    for file_name in glob.iglob(
        str(
            pathlib.Path(
                "tests",
                "fixture_data",
                "winsorisation",
                f"{scenario_category}_scenarios",
                "*_input.csv",
            )
        )
    ):
        test_scenarios.append(
            (
                f"{scenario_category}_scenarios",
                os.path.basename(file_name).replace("_input.csv", ""),
            )
        )

# --- Test type validation on the input dataframe(s) ---


@pytest.mark.dependency()
def test_dataframe_not_a_dataframe():
    with pytest.raises(TypeError):
        # noinspection PyTypeChecker
        winsorisation.one_sided_winsorise("not_a_dataframe", *params)
