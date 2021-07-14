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


# --- Test if params not strings  ---


@pytest.mark.dependency()
def test_params_not_string(fxt_load_test_csv):
    test_dataframe = fxt_load_test_csv(
        dataframe_columns,
        dataframe_types,
        "winsorisation",
        "unit",
        "basic_functionality",
    )
    bad_params = (
        reference_col,
        period_col,
        grouping_col,
        ["target_col"],
        design_weight_col,
        l_value_col,
        outlier_weight_col,
    )
    with pytest.raises(TypeError):
        winsorisation.one_sided_winsorise(test_dataframe, *bad_params)


# --- Test if params null  ---


@pytest.mark.dependency()
def test_params_null(fxt_load_test_csv):
    test_dataframe = fxt_load_test_csv(
        dataframe_columns,
        dataframe_types,
        "winsorisation",
        "unit",
        "basic_functionality",
    )
    bad_params = (
        reference_col,
        period_col,
        grouping_col,
        "",
        design_weight_col,
        l_value_col,
        outlier_weight_col,
    )
    with pytest.raises(ValueError):
        winsorisation.one_sided_winsorise(test_dataframe, *bad_params)


# --- Test validation fail if nulls in data  ---


@pytest.mark.dependency()
def test_dataframe_nulls_in_data(fxt_load_test_csv):
    test_dataframe = fxt_load_test_csv(
        dataframe_columns,
        dataframe_types,
        "winsorisation",
        "unit",
        "null_value_present",
    )
    with pytest.raises(winsorisation.ValidationError):
        winsorisation.one_sided_winsorise(test_dataframe, *params)


# --- Test if cols missing from input dataframe(s)  ---


@pytest.mark.dependency()
def test_dataframe_column_missing(fxt_load_test_csv):
    test_dataframe = fxt_load_test_csv(
        dataframe_columns,
        dataframe_types,
        "winsorisation",
        "unit",
        "basic_functionality",
    )
    bad_dataframe = test_dataframe.drop(target_col)
    with pytest.raises(winsorisation.ValidationError):
        winsorisation.one_sided_winsorise(bad_dataframe, *params)


# --- Test validation fail if mismatched calibration cols  ---


@pytest.mark.dependency()
def test_params_mismatched_calibration_cols(fxt_load_test_csv):
    test_dataframe = fxt_load_test_csv(
        dataframe_columns,
        dataframe_types,
        "winsorisation",
        "unit",
        "basic_functionality",
    )
    bad_params = (
        reference_col,
        period_col,
        grouping_col,
        target_col,
        design_weight_col,
        l_value_col,
        outlier_weight_col,
        calibration_weight_col,
    )
    with pytest.raises(TypeError):
        winsorisation.one_sided_winsorise(test_dataframe, *bad_params)


# --- Test if output contents are as expected, both new columns and data ---


@pytest.mark.dependency()
def test_return_has_no_unexpected_columns(fxt_spark_session, fxt_load_test_csv):
    test_dataframe = fxt_load_test_csv(
        dataframe_columns,
        dataframe_types,
        "winsorisation",
        "unit",
        "basic_functionality",
    )
    # Make sure that no extra columns pass through.
    test_dataframe = test_dataframe.withColumn("bonus_column", lit(0))
    ret_val = winsorisation.one_sided_winsorise(test_dataframe, *params)
    # perform action on the dataframe to trigger lazy evaluation...
    ret_val.count()
    # ...and then check
    assert isinstance(ret_val, type(test_dataframe))
    ret_cols = ret_val.columns
    assert "bonus_column" not in ret_cols


@pytest.mark.parametrize(
    "scenario_type, scenario",
    sorted(test_scenarios, key=lambda t: pathlib.Path(t[0], t[1])),
)
@pytest.mark.dependency(
    depends=[
        "test_dataframe_not_a_dataframe",
        "test_params_not_string",
        "test_params_null",
        "test_dataframe_nulls_in_data",
        "test_dataframe_column_missing",
        "test_params_mismatched_calibration_cols",
        "test_return_has_no_unexpected_columns",
    ]
)
def test_calculations(fxt_load_test_csv, scenario_type, scenario):
    test_dataframe = fxt_load_test_csv(
        dataframe_columns,
        dataframe_types,
        "winsorisation",
        scenario_type,
        f"{scenario}_input",
    )

    winsorisation_kwargs = {}
    if auxiliary_col in test_dataframe.columns:
        winsorisation_kwargs["auxiliary_col"] = auxiliary_col
        winsorisation_kwargs["calibration_col"] = calibration_weight_col

    exp_val = fxt_load_test_csv(
        dataframe_columns,
        dataframe_types,
        "winsorisation",
        scenario_type,
        f"{scenario}_output",
    )

    ret_val = winsorisation.one_sided_winsorise(
        test_dataframe, *params, **winsorisation_kwargs
    )

    assert isinstance(ret_val, type(test_dataframe))
    sort_col_list = [reference_col, period_col]
    assert_approx_df_equality(
        ret_val.sort(sort_col_list),
        exp_val.sort(sort_col_list),
        0.01,
        ignore_nullable=True,
    )
