import glob
import os
import pathlib

import pytest
from chispa import assert_approx_df_equality

from statistical_methods_library import estimation

period_col = "period"
strata_col = "strata"
sample_col = "sample_mkr"
death_col = "death_mkr"
h_col = "H"
auxiliary_col = "aux"
calibration_group_col = "cal_group"
design_weight_col = "design_weight"
calibration_weight_col = "calibration_weight"

dataframe_columns = (
    period_col,
    strata_col,
    sample_col,
    death_col,
    h_col,
    auxiliary_col,
    calibration_group_col,
    design_weight_col,
    calibration_weight_col,
)

dataframe_types = {
    period_col: "string",
    strata_col: "string",
    sample_col: "int",
    death_col: "int",
    h_col: "double",
    auxiliary_col: "double",
    calibration_group_col: "string",
    design_weight_col: "double",
    calibration_weight_col: "double",
}

params = {
        period_col,
        strata_col,
        sample_col
    }


# --- Test type validation on the input dataframe(s) ---

@pytest.mark.dependency()
def test_dataframe_not_a_dataframe():
    with pytest.raises(TypeError):
        # noinspection PyTypeChecker
        estimation.estimate("not_a_dataframe", *params)


# --- Test if cols missing from input dataframe(s) ---

@pytest.mark.dependency()
def test_dataframe_column_missing(fxt_load_test_csv):
    test_dataframe = fxt_load_test_csv(
        dataframe_columns, dataframe_types, "estimation", "unit", "basic_functionality"
    )
    bad_dataframe = test_dataframe.drop(strata_col)
    with pytest.raises(estimation.ValidationError):
        estimation.estimate(bad_dataframe, *params)


# --- Test if output contents are as expected, both new columns and data ---

test_scenarios = []

for scenario_category in ("dev", "methodology"):
    for file_name in glob.iglob(
        str(
            pathlib.Path(
                "tests",
                "fixture_data",
                "estimation",
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


@pytest.mark.parametrize(
    "scenario_type, scenario",
    sorted(test_scenarios, key=lambda t: pathlib.Path(t[0], t[1])),
)
def test_calculations(fxt_load_test_csv, scenario_type, scenario):
    test_dataframe = fxt_load_test_csv(
        dataframe_columns,
        dataframe_types,
        "estimation",
        scenario_type,
        f"{scenario}_input",
    )

    # We use estimation_kwargs to allow us to pass in the appropriate columns.
    estimation_kwargs = {
        "period_col": period_col,
        "strata_col": strata_col,
        "sample_marker_col": sample_col,
    }
    if death_col in test_dataframe.columns:
        estimation_kwargs["death_marker_col"] = death_col
        estimation_kwargs["h_value_col"] = h_col

    if auxiliary_col in test_dataframe.columns:
        estimation_kwargs["auxiliary_col"] = auxiliary_col

    if calibration_group_col in test_dataframe.columns:
        estimation_kwargs["calibration_group_col"] = calibration_group_col

    exp_val = fxt_load_test_csv(
        dataframe_columns,
        dataframe_types,
        "estimation",
        scenario_type,
        f"{scenario}_output",
    )

    ret_val = estimation.estimate(test_dataframe, **estimation_kwargs)

    assert isinstance(ret_val, type(test_dataframe))
    sort_col_list = ["period", "strata"]
    if calibration_group_col in test_dataframe.columns:
        sort_col_list.append(calibration_group_col)

    assert_approx_df_equality(
        ret_val.sort(sort_col_list),
        exp_val.sort(sort_col_list),
        0.0001,
        ignore_nullable=True,
    )
