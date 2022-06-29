import glob
import os
import pathlib

import pytest
from chispa import assert_approx_df_equality
from pyspark.sql.functions import lit

from statistical_methods_library import estimation

unique_identifier_col = "reference"
period_col = "period"
strata_col = "strata"
sample_col = "sample_inclusion_marker"
death_col = "death_marker"
h_col = "H"
auxiliary_col = "auxiliary"
calibration_group_col = "calibration_group"
unadjusted_design_weight_col = "unadjusted_design_weight"
design_weight_col = "design_weight"
calibration_weight_col = "calibration_weight"

dataframe_columns = (
    unique_identifier_col,
    period_col,
    strata_col,
    sample_col,
    death_col,
    h_col,
    auxiliary_col,
    calibration_group_col,
    design_weight_col,
    unadjusted_design_weight_col,
    calibration_weight_col,
)

dataframe_types = {
    unique_identifier_col: "string",
    period_col: "string",
    strata_col: "string",
    sample_col: "int",
    death_col: "int",
    h_col: "boolean",
    auxiliary_col: "double",
    calibration_group_col: "string",
    design_weight_col: "double",
    unadjusted_design_weight_col: "double",
    calibration_weight_col: "double",
}

params = (unique_identifier_col, period_col, strata_col, sample_col)

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


# --- Test type validation on the input dataframe(s) ---
@pytest.mark.dependency()
def test_input_not_a_dataframe():
    with pytest.raises(TypeError):
        # noinspection PyTypeChecker
        estimation.estimate("not_a_dataframe", *params)


# --- Test validation fail if mismatched death cols  ---
@pytest.mark.dependency()
def test_params_mismatched_death_cols(fxt_load_test_csv):
    test_dataframe = fxt_load_test_csv(
        dataframe_columns, dataframe_types, "estimation", "unit", "basic_functionality"
    )
    bad_params = (unique_identifier_col, period_col, strata_col, sample_col, death_col)
    with pytest.raises(TypeError):
        estimation.estimate(test_dataframe, *bad_params)


# --- Test validation fail if mismatched calibration cols  ---
@pytest.mark.dependency()
def test_params_mismatched_calibration_cols(fxt_load_test_csv):
    test_dataframe = fxt_load_test_csv(
        dataframe_columns, dataframe_types, "estimation", "unit", "basic_functionality"
    )
    bad_params = (
        unique_identifier_col,
        period_col,
        strata_col,
        sample_col,
        calibration_group_col,
    )
    with pytest.raises(TypeError):
        estimation.estimate(test_dataframe, *bad_params)


# --- Test if params not strings  ---
@pytest.mark.dependency()
def test_params_not_string(fxt_load_test_csv):
    test_dataframe = fxt_load_test_csv(
        dataframe_columns, dataframe_types, "estimation", "unit", "basic_functionality"
    )
    bad_params = (unique_identifier_col, period_col, ["strata_col"], sample_col)
    with pytest.raises(TypeError):
        estimation.estimate(test_dataframe, *bad_params)


# --- Test if params null  ---
@pytest.mark.dependency()
def test_params_null(fxt_load_test_csv):
    test_dataframe = fxt_load_test_csv(
        dataframe_columns, dataframe_types, "estimation", "unit", "basic_functionality"
    )
    bad_params = (unique_identifier_col, period_col, "", sample_col)
    with pytest.raises(ValueError):
        estimation.estimate(test_dataframe, *bad_params)


# --- Test validation fail if nulls in data  ---
@pytest.mark.dependency()
def test_dataframe_nulls_in_data(fxt_load_test_csv):
    test_dataframe = fxt_load_test_csv(
        dataframe_columns, dataframe_types, "estimation", "unit", "null_value_present"
    )
    with pytest.raises(estimation.ValidationError):
        estimation.estimate(test_dataframe, *params)


# --- Test if cols missing from input dataframe(s)  ---
@pytest.mark.dependency()
def test_dataframe_column_missing(fxt_load_test_csv):
    test_dataframe = fxt_load_test_csv(
        dataframe_columns, dataframe_types, "estimation", "unit", "basic_functionality"
    )
    bad_dataframe = test_dataframe.drop(strata_col)
    with pytest.raises(estimation.ValidationError):
        estimation.estimate(bad_dataframe, *params)


# --- Test if references are duplicated in the input dataframe  ---
@pytest.mark.dependency()
def test_dataframe_duplicate_reference(fxt_load_test_csv):
    test_dataframe = fxt_load_test_csv(
        dataframe_columns, dataframe_types, "estimation", "unit", "duplicate_references"
    )
    with pytest.raises(estimation.ValidationError):
        estimation.estimate(test_dataframe, *params)


# --- Test validation fail if non-boolean markers in data  ---
@pytest.mark.dependency()
def test_dataframe_non_boolean_markers(fxt_load_test_csv):
    test_dataframe = fxt_load_test_csv(
        dataframe_columns, dataframe_types, "estimation", "unit", "non_boolean_markers"
    )
    with pytest.raises(estimation.ValidationError):
        estimation.estimate(test_dataframe, *params)


@pytest.mark.dependency()
def test_dataframe_large_death_count(fxt_load_test_csv):
    test_dataframe = fxt_load_test_csv(
        dataframe_columns,
        dataframe_types,
        "estimation",
        "unit",
        "large_death_count",
    )
    with pytest.raises(estimation.ValidationError):
        estimation_params = [*params, death_col, h_col]
        estimation.estimate(test_dataframe, *estimation_params)


# --- Test validation fail if mixed h values in a strata  ---
@pytest.mark.dependency()
def test_dataframe_mixed_h_values_in_strata(fxt_load_test_csv):
    test_dataframe = fxt_load_test_csv(
        dataframe_columns,
        dataframe_types,
        "estimation",
        "unit",
        "mixed_h-values_in_strata",
    )
    with pytest.raises(estimation.ValidationError):
        estimation_params = [*params, death_col, h_col]
        estimation.estimate(test_dataframe, *estimation_params)


# --- Test output is correct type ---
@pytest.mark.dependency()
def test_dataframe_correct_type(fxt_spark_session, fxt_load_test_csv):
    test_dataframe = fxt_load_test_csv(
        dataframe_columns, dataframe_types, "estimation", "unit", "basic_functionality"
    )

    test_dataframe = test_dataframe.withColumn("bonus_column", lit(0))
    ret_val = estimation.estimate(test_dataframe, *params)
    assert isinstance(ret_val, type(test_dataframe))


# --- Test no extra columns are copied to the output ---
@pytest.mark.dependency()
def test_dataframe_no_extra_columns(fxt_spark_session, fxt_load_test_csv):
    test_dataframe = fxt_load_test_csv(
        dataframe_columns, dataframe_types, "estimation", "unit", "basic_functionality"
    )
    test_dataframe = test_dataframe.withColumn("bonus_column", lit(0))
    ret_val = estimation.estimate(test_dataframe, *params)
    # perform action on the dataframe to trigger lazy evaluation
    ret_val.count()
    ret_cols = ret_val.columns
    assert "bonus_column" not in ret_cols


# --- Test expected columns are in the output ---
@pytest.mark.dependency()
def test_dataframe_expected_columns(fxt_spark_session, fxt_load_test_csv):
    test_dataframe = fxt_load_test_csv(
        dataframe_columns, dataframe_types, "estimation", "unit", "basic_functionality"
    )
    ret_val = estimation.estimate(
        test_dataframe,
        *params,
        auxiliary_col=auxiliary_col,
        calibration_group_col=calibration_group_col,
    )
    # perform action on the dataframe to trigger lazy evaluation
    ret_val.count()
    ret_cols = set(ret_val.columns)
    expected_cols = {
        period_col,
        strata_col,
        calibration_group_col,
        design_weight_col,
        calibration_weight_col,
    }
    assert expected_cols == ret_cols


# --- Test expected columns are in the output when default names aren't used ---
@pytest.mark.dependency()
def test_dataframe_expected_columns_not_defaults(fxt_spark_session, fxt_load_test_csv):
    test_dataframe = fxt_load_test_csv(
        dataframe_columns, dataframe_types, "estimation", "unit", "basic_functionality"
    )
    ret_val = estimation.estimate(
        test_dataframe,
        *params,
        auxiliary_col=auxiliary_col,
        calibration_group_col=calibration_group_col,
        unadjusted_design_weight_col="u_a",
        design_weight_col="a",
        calibration_weight_col="g",
    )
    # perform action on the dataframe to trigger lazy evaluation
    ret_val.count()
    ret_cols = set(ret_val.columns)
    expected_cols = {period_col, strata_col, calibration_group_col, "u_a", "a", "g"}
    assert expected_cols == ret_cols


# --- Test valid scenarios ---
@pytest.mark.parametrize(
    "scenario_type, scenario",
    sorted(test_scenarios, key=lambda t: pathlib.Path(t[0], t[1])),
)
@pytest.mark.dependency(
    depends=[
        "test_input_not_a_dataframe",
        "test_params_mismatched_death_cols",
        "test_params_mismatched_calibration_cols",
        "test_params_not_string",
        "test_params_null",
        "test_dataframe_duplicate_reference",
        "test_dataframe_nulls_in_data",
        "test_dataframe_column_missing",
        "test_dataframe_non_boolean_markers",
        "test_dataframe_mixed_h_values_in_strata",
        "test_dataframe_correct_type",
        "test_dataframe_no_extra_columns",
        "test_dataframe_expected_columns",
        "test_dataframe_expected_columns_not_defaults",
    ]
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
        "unique_identifier_col": unique_identifier_col,
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

    if "unadjusted" in scenario:
        estimation_kwargs["unadjusted_design_weight_col"] = unadjusted_design_weight_col

    exp_val = fxt_load_test_csv(
        dataframe_columns,
        dataframe_types,
        "estimation",
        scenario_type,
        f"{scenario}_output",
    )

    ret_val = estimation.estimate(test_dataframe, **estimation_kwargs)

    sort_col_list = ["period", "strata"]
    if calibration_group_col in test_dataframe.columns:
        sort_col_list.append(calibration_group_col)

    assert_approx_df_equality(
        ret_val.sort(sort_col_list),
        exp_val.sort(sort_col_list),
        0.01,
        ignore_nullable=True,
    )
