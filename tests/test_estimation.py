import glob
import os
import pathlib

import pytest
from chispa import assert_approx_df_equality
from pyspark.sql.functions import lit

from statistical_methods_library.estimation import ht_ratio
from statistical_methods_library.utilities.exceptions import ValidationError

unique_identifier_col = "reference"
period_col = "period"
strata_col = "strata"
sample_col = "sample_inclusion_marker"
adjustment_col = "adjustment_marker"
h_col = "H"
auxiliary_col = "auxiliary"
calibration_group_col = "calibration_group"
unadjusted_design_weight_col = "unadjusted_design_weight"
design_weight_col = "design_weight"
calibration_factor_col = "calibration_factor"

dataframe_columns = (
    unique_identifier_col,
    period_col,
    strata_col,
    sample_col,
    adjustment_col,
    h_col,
    auxiliary_col,
    calibration_group_col,
    design_weight_col,
    unadjusted_design_weight_col,
    calibration_factor_col,
)

dataframe_types = {
    unique_identifier_col: "string",
    period_col: "string",
    strata_col: "string",
    sample_col: "boolean",
    adjustment_col: "string",
    h_col: "boolean",
    auxiliary_col: "double",
    calibration_group_col: "string",
    design_weight_col: "double",
    unadjusted_design_weight_col: "double",
    calibration_factor_col: "double",
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
                "ht_ratio",
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
        ht_ratio.estimate("not_a_dataframe", *params)


# --- Test validation fail if mismatched death cols  ---
@pytest.mark.dependency()
def test_params_mismatched_death_cols(fxt_load_test_csv):
    test_dataframe = fxt_load_test_csv(
        dataframe_columns,
        dataframe_types,
        "estimation",
        "ht_ratio",
        "unit",
        "basic_functionality",
    )
    bad_params = (
        unique_identifier_col,
        period_col,
        strata_col,
        sample_col,
        adjustment_col,
    )
    with pytest.raises(TypeError):
        ht_ratio.estimate(test_dataframe, *bad_params)


# --- Test validation fail if mismatched calibration cols  ---
@pytest.mark.dependency()
def test_params_mismatched_calibration_cols(fxt_load_test_csv):
    test_dataframe = fxt_load_test_csv(
        dataframe_columns,
        dataframe_types,
        "estimation",
        "ht_ratio",
        "unit",
        "basic_functionality",
    )
    bad_params = (
        unique_identifier_col,
        period_col,
        strata_col,
        sample_col,
        calibration_group_col,
    )
    with pytest.raises(TypeError):
        ht_ratio.estimate(test_dataframe, *bad_params)


# --- Test if params not strings  ---
@pytest.mark.dependency()
def test_params_not_string(fxt_load_test_csv):
    test_dataframe = fxt_load_test_csv(
        dataframe_columns,
        dataframe_types,
        "estimation",
        "ht_ratio",
        "unit",
        "basic_functionality",
    )
    bad_params = (unique_identifier_col, period_col, ["strata_col"], sample_col)
    with pytest.raises(TypeError):
        ht_ratio.estimate(test_dataframe, *bad_params)


# --- Test if params null  ---
@pytest.mark.dependency()
def test_params_null(fxt_load_test_csv):
    test_dataframe = fxt_load_test_csv(
        dataframe_columns,
        dataframe_types,
        "estimation",
        "ht_ratio",
        "unit",
        "basic_functionality",
    )
    bad_params = (unique_identifier_col, period_col, "", sample_col)
    with pytest.raises(ValueError):
        ht_ratio.estimate(test_dataframe, *bad_params)


# --- Test validation fail if nulls in data  ---
@pytest.mark.dependency()
def test_dataframe_nulls_in_data(fxt_load_test_csv):
    test_dataframe = fxt_load_test_csv(
        dataframe_columns,
        dataframe_types,
        "estimation",
        "ht_ratio",
        "unit",
        "null_value_present",
    )
    with pytest.raises(ValidationError):
        ht_ratio.estimate(test_dataframe, *params)


# --- Test if cols missing from input dataframe(s)  ---
@pytest.mark.dependency()
def test_dataframe_column_missing(fxt_load_test_csv):
    test_dataframe = fxt_load_test_csv(
        dataframe_columns,
        dataframe_types,
        "estimation",
        "ht_ratio",
        "unit",
        "basic_functionality",
    )
    bad_dataframe = test_dataframe.drop(strata_col)
    with pytest.raises(ValidationError):
        ht_ratio.estimate(bad_dataframe, *params)


# --- Test if references are duplicated in the input dataframe  ---
@pytest.mark.dependency()
def test_dataframe_duplicate_reference(fxt_load_test_csv):
    test_dataframe = fxt_load_test_csv(
        dataframe_columns,
        dataframe_types,
        "estimation",
        "ht_ratio",
        "unit",
        "duplicate_references",
    )
    with pytest.raises(ValueError):
        ht_ratio.estimate(test_dataframe, *params)


@pytest.mark.dependency()
def test_dataframe_deaths_in_unsampled(fxt_load_test_csv):
    test_dataframe = fxt_load_test_csv(
        dataframe_columns,
        dataframe_types,
        "estimation",
        "ht_ratio",
        "unit",
        "deaths_in_unsampled",
    )
    with pytest.raises(ValueError):
        estimation_params = [*params, adjustment_col, h_col]
        ht_ratio.estimate(test_dataframe, *estimation_params)


# --- Test validation fail if mixed h values in a strata  ---
@pytest.mark.dependency()
def test_dataframe_mixed_h_values_in_strata(fxt_load_test_csv):
    test_dataframe = fxt_load_test_csv(
        dataframe_columns,
        dataframe_types,
        "estimation",
        "ht_ratio",
        "unit",
        "mixed_h-values_in_strata",
    )
    with pytest.raises(ValueError):
        estimation_params = [*params, adjustment_col, h_col]
        ht_ratio.estimate(test_dataframe, *estimation_params)


# --- Test output is correct type ---
@pytest.mark.dependency()
def test_dataframe_correct_type(fxt_spark_session, fxt_load_test_csv):
    test_dataframe = fxt_load_test_csv(
        dataframe_columns,
        dataframe_types,
        "estimation",
        "ht_ratio",
        "unit",
        "basic_functionality",
    )

    test_dataframe = test_dataframe.withColumn("bonus_column", lit(0))
    ret_val = ht_ratio.estimate(test_dataframe, *params)
    assert isinstance(ret_val, type(test_dataframe))


# --- Test no extra columns are copied to the output ---
@pytest.mark.dependency()
def test_dataframe_no_extra_columns(fxt_spark_session, fxt_load_test_csv):
    test_dataframe = fxt_load_test_csv(
        dataframe_columns,
        dataframe_types,
        "estimation",
        "ht_ratio",
        "unit",
        "basic_functionality",
    )
    test_dataframe = test_dataframe.withColumn("bonus_column", lit(0))
    ret_val = ht_ratio.estimate(test_dataframe, *params)
    # perform action on the dataframe to trigger lazy evaluation
    ret_val.count()
    ret_cols = ret_val.columns
    assert "bonus_column" not in ret_cols


# --- Test expected columns are in the output ---
@pytest.mark.dependency()
def test_dataframe_expected_columns(fxt_spark_session, fxt_load_test_csv):
    test_dataframe = fxt_load_test_csv(
        dataframe_columns,
        dataframe_types,
        "estimation",
        "ht_ratio",
        "unit",
        "basic_functionality",
    )
    ret_val = ht_ratio.estimate(
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
        calibration_factor_col,
    }
    assert expected_cols == ret_cols


# --- Test expected columns are in the output when default names aren't used ---
@pytest.mark.dependency()
def test_dataframe_expected_columns_not_defaults(fxt_spark_session, fxt_load_test_csv):
    test_dataframe = fxt_load_test_csv(
        dataframe_columns,
        dataframe_types,
        "estimation",
        "ht_ratio",
        "unit",
        "basic_functionality",
    )
    ret_val = ht_ratio.estimate(
        test_dataframe,
        *params,
        auxiliary_col=auxiliary_col,
        calibration_group_col=calibration_group_col,
        unadjusted_design_weight_col="u_a",
        design_weight_col="a",
        calibration_factor_col="g",
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
        "ht_ratio",
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
    if adjustment_col in test_dataframe.columns:
        estimation_kwargs["adjustment_marker_col"] = adjustment_col
        estimation_kwargs["h_value_col"] = h_col

    if "out_of_scope" in scenario:
        if "full" in scenario:
            estimation_kwargs["out_of_scope_full"] = True
        else:
            estimation_kwargs["out_of_scope_full"] = False

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
        "ht_ratio",
        scenario_type,
        f"{scenario}_output",
    )

    ret_val = ht_ratio.estimate(test_dataframe, **estimation_kwargs)

    assert isinstance(ret_val, type(test_dataframe))
    sort_col_list = ["period", "strata"]
    select_cols = list(set(dataframe_columns) & set(exp_val.columns))
    if calibration_group_col in test_dataframe.columns:
        sort_col_list.append(calibration_group_col)

    assert_approx_df_equality(
        ret_val.sort(sort_col_list).select(select_cols),
        exp_val.sort(sort_col_list).select(select_cols),
        0.01,
        ignore_nullable=True,
    )
