import glob
import os
import pathlib

import pytest
from chispa.dataframe_comparer import assert_approx_df_equality
from pyspark.sql.functions import lit

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

reference_type = "string"
period_type = "string"
strata_type = "string"
target_type = "double"
auxiliary_type = "double"
output_type = "double"
marker_type = "string"
backward_type = "double"
forward_type = "double"
construction_type = "double"

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
    construction_col,
)

dataframe_types = {
    reference_col: reference_type,
    period_col: period_type,
    strata_col: strata_type,
    target_col: target_type,
    auxiliary_col: auxiliary_type,
    output_col: output_type,
    marker_col: marker_type,
    backward_col: backward_type,
    forward_col: forward_type,
    construction_col: construction_type,
}

# Params used when calling impute
params = (
    reference_col,
    period_col,
    strata_col,
    target_col,
    auxiliary_col,
    output_col,
    marker_col,
)

# Mapping for which columns we should select per scenario category
selection_map = {
    "dev": [output_col, marker_col],
    "methodology": [
        output_col,
        marker_col,
        forward_col,
        backward_col,
        construction_col,
    ],
}

test_scenarios = [
    ("unit", "ratio_calculation", ["forward", "backward", "construction"])
]
for scenario_category in ("dev", "methodology"):
    for file_name in glob.iglob(
        str(
            pathlib.Path(
                "tests",
                "fixture_data",
                "imputation",
                f"{scenario_category}_scenarios",
                "*_input.csv",
            )
        )
    ):
        test_scenarios.append(
            (
                f"{scenario_category}_scenarios",
                os.path.basename(file_name).replace("_input.csv", ""),
                selection_map[scenario_category],
            )
        )


# --- Test type validation on the input dataframe(s) ---


@pytest.mark.dependency()
def test_input_not_a_dataframe():
    with pytest.raises(TypeError):
        # noinspection PyTypeChecker
        imputation.impute("not_a_dataframe", *params)


# --- Test if cols missing from input dataframe(s) ---


@pytest.mark.dependency()
def test_dataframe_column_missing(fxt_load_test_csv):
    test_dataframe = fxt_load_test_csv(
        dataframe_columns, dataframe_types, "imputation", "unit", "basic_functionality"
    )
    bad_dataframe = test_dataframe.drop(strata_col)
    with pytest.raises(imputation.ValidationError):
        imputation.impute(bad_dataframe, *params)


# --- Test if params null ---


@pytest.mark.dependency()
def test_params_null(fxt_load_test_csv):
    test_dataframe = fxt_load_test_csv(
        dataframe_columns, dataframe_types, "imputation", "unit", "basic_functionality"
    )
    bad_params = (
        reference_col,
        period_col,
        strata_col,
        "",
        auxiliary_col,
        output_col,
        marker_col,
    )
    with pytest.raises(ValueError):
        imputation.impute(test_dataframe, *bad_params)


@pytest.mark.dependency()
def test_params_missing_link_column(fxt_load_test_csv):
    test_dataframe = fxt_load_test_csv(
        dataframe_columns, dataframe_types, "imputation", "unit", "basic_functionality"
    )
    with pytest.raises(TypeError):
        imputation.impute(
            test_dataframe, *params, construction_link_col=construction_col
        )


@pytest.mark.dependency()
def test_params_not_string(fxt_load_test_csv):
    test_dataframe = fxt_load_test_csv(
        dataframe_columns, dataframe_types, "imputation", "unit", "basic_functionality"
    )
    bad_params = (
        reference_col,
        period_col,
        strata_col,
        ["target_col"],
        auxiliary_col,
        output_col,
        marker_col,
    )
    with pytest.raises(TypeError):
        imputation.impute(test_dataframe, *bad_params)


# --- Test if output contents are as expected, both new columns and data ---


@pytest.mark.dependency()
def test_dataframe_returned_as_expected(fxt_spark_session, fxt_load_test_csv):
    test_dataframe = fxt_load_test_csv(
        dataframe_columns, dataframe_types, "imputation", "unit", "basic_functionality"
    )
    # Make sure that no extra columns pass through.
    test_dataframe = test_dataframe.withColumn("bonus_column", lit(0))
    ret_val = imputation.impute(test_dataframe, *params)
    # perform action on the dataframe to trigger lazy evaluation
    ret_val.count()
    assert isinstance(ret_val, type(test_dataframe))
    ret_cols = ret_val.columns
    assert "bonus_column" not in ret_cols


# --- Test that when provided back data does not match input schema then fails ---
@pytest.mark.dependency()
def test_back_data_type_mismatch(fxt_load_test_csv, fxt_spark_session):
    test_dataframe = fxt_load_test_csv(
        dataframe_columns, dataframe_types, "imputation", "unit", "basic_functionality"
    )
    bad_back_data = fxt_load_test_csv(
        [
            "reference","period","strata"
        ],
        {
            "reference": "int",
            "period": "int",
            "strata": "string",
            "target": "string",
            "aux": "int"
        },
        "imputation",
        "unit", 
        "back_data_bad_schema"
    )
    with pytest.raises(TypeError):
        imputation.impute(test_dataframe, *params, back_data_df=bad_back_data)


@pytest.mark.dependency()
def test_back_data_contains_nulls(fxt_load_test_csv, fxt_spark_session):
    test_dataframe = fxt_load_test_csv(
        dataframe_columns, dataframe_types, "imputation", "unit", "basic_functionality"
    )
    bad_back_data = fxt_load_test_csv(
        dataframe_columns, dataframe_types,
        "imputation",
        "unit", 
        "back_data_nulls"
    )

    with pytest.raises(imputation.ValidationError):
        imputation.impute(test_dataframe, *params, back_data_df=bad_back_data)


@pytest.mark.parametrize(
    "scenario_type, scenario, selection",
    sorted(test_scenarios, key=lambda t: pathlib.Path(t[0], t[1])),
)
@pytest.mark.dependency(
    depends=[
        "test_dataframe_returned_as_expected",
        "test_params_not_string",
        "test_params_null",
        "test_params_missing_link_column",
        "test_dataframe_column_missing",
        "test_input_not_a_dataframe",
    ]
)
def test_calculations(fxt_load_test_csv, scenario_type, scenario, selection):
    test_dataframe = fxt_load_test_csv(
        dataframe_columns,
        dataframe_types,
        "imputation",
        scenario_type,
        f"{scenario}_input",
    )

    # We use imputation_kwargs to allow us to pass in the forward, backward
    # and construction link columns which are usually defaulted to None. This
    # means that we can autodetect when we should pass these.
    if forward_col in test_dataframe.columns:
        imputation_kwargs = {
            "forward_link_col": forward_col,
            "backward_link_col": backward_col,
            "construction_link_col": construction_col,
        }
    else:
        imputation_kwargs = {}

    exp_val = fxt_load_test_csv(
        dataframe_columns,
        dataframe_types,
        "imputation",
        scenario_type,
        f"{scenario}_output",
    )

    ret_val = imputation.impute(test_dataframe, *params, **imputation_kwargs)

    assert isinstance(ret_val, type(test_dataframe))
    sort_col_list = ["reference", "period"]
    assert_approx_df_equality(
        ret_val.sort(sort_col_list).select(selection),
        exp_val.sort(sort_col_list).select(selection),
        0.0001,
        ignore_nullable=True,
    )


@pytest.mark.parametrize(
    "scenario_type, scenario, selection",
    sorted(test_scenarios, key=lambda t: pathlib.Path(t[0], t[1])),
)
def test_back_data_calculations(fxt_load_test_csv, scenario_type, scenario, selection):
    test_dataframe = fxt_load_test_csv(
        dataframe_columns,
        dataframe_types,
        "imputation",
        scenario_type,
        f"{scenario}_input",
    )

    back_data_cols = [
        reference_col,
        period_col,
        strata_col,
        marker_col,
        target_col,
    ]

    back_data_types = {
        reference_col: reference_type,
        period_col: period_type,
        strata_col: strata_type,
        marker_col: marker_type,
        target_col: target_type,
    }

    back_data = fxt_load_test_csv(
        back_data_cols,
        back_data_types,
        "imputation",
        "back_data",
        "201912",
    )

    # We use imputation_kwargs to allow us to pass in the forward, backward
    # and construction link columns which are usually defaulted to None. This
    # means that we can autodetect when we should pass these.
    if forward_col in test_dataframe.columns:
        imputation_kwargs = {
            "forward_link_col": forward_col,
            "backward_link_col": backward_col,
            "construction_link_col": construction_col,
            "back_data": back_data,
        }
    else:
        imputation_kwargs = {
            "back_data": back_data,
        }

    exp_val = fxt_load_test_csv(
        dataframe_columns,
        dataframe_types,
        "imputation",
        scenario_type,
        f"{scenario}_output_back_data",
    )

    ret_val = imputation.impute(test_dataframe, *params, **imputation_kwargs)

    assert isinstance(ret_val, type(test_dataframe))
    sort_col_list = ["reference", "period"]
    assert_approx_df_equality(
        ret_val.sort(sort_col_list).select(selection),
        exp_val.sort(sort_col_list).select(selection),
        0.0001,
        ignore_nullable=True,
    )
