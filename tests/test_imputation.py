import glob
import os
import pathlib

import pytest
from chispa.dataframe_comparer import assert_approx_df_equality
from pyspark.sql.functions import col, lit, when

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

bad_dataframe_types = dataframe_types.copy()
bad_dataframe_types[target_col] = reference_type

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

default_params = (
    reference_col,
    period_col,
    strata_col,
    target_col,
    auxiliary_col,
)

# Mapping for which columns we should select per scenario category
selection_map = {
    "back_data": [reference_col, period_col, output_col, marker_col],
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
for scenario_category in ("dev", "methodology", "back_data"):
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


def scenarios(categories):
    return [scenario for scenario in test_scenarios if scenario[0] in categories]


# --- Test type validation on the input dataframe(s) ---


@pytest.mark.dependency()
def test_input_not_a_dataframe():
    with pytest.raises(TypeError):
        # noinspection PyTypeChecker
        imputation.impute("not_a_dataframe", *params)


# --- Test type validation on the back_data dataframe(s) ---


@pytest.mark.dependency()
def test_back_data_not_a_dataframe(fxt_load_test_csv):
    test_dataframe = fxt_load_test_csv(
        dataframe_columns, dataframe_types, "imputation", "unit", "basic_functionality"
    )
    with pytest.raises(TypeError):
        # noinspection PyTypeChecker
        imputation.impute(test_dataframe, *params, back_data_df="not_a_dataframe")


# --- Test if cols missing from input dataframe(s) ---


@pytest.mark.dependency()
def test_dataframe_column_missing(fxt_load_test_csv):
    test_dataframe = fxt_load_test_csv(
        dataframe_columns, dataframe_types, "imputation", "unit", "basic_functionality"
    )
    bad_dataframe = test_dataframe.drop(strata_col)
    with pytest.raises(imputation.ValidationError):
        imputation.impute(bad_dataframe, *params)


# --- Test if dataframe has duplicate rows ---


@pytest.mark.dependency()
def test_dataframe_duplicate_rows(fxt_load_test_csv):
    test_dataframe = fxt_load_test_csv(
        dataframe_columns, dataframe_types, "imputation", "unit", "duplicate_rows"
    )
    with pytest.raises(imputation.ValidationError):
        imputation.impute(test_dataframe, *params)


# --- Test if target missing from input dataframe(s) ---


@pytest.mark.dependency()
def test_dataframe_target_missing(fxt_load_test_csv):
    test_dataframe = fxt_load_test_csv(
        dataframe_columns, dataframe_types, "imputation", "unit", "basic_functionality"
    )
    bad_dataframe = test_dataframe.drop(target_col)
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
def test_back_data_missing_column(fxt_load_test_csv, fxt_spark_session):
    test_dataframe = fxt_load_test_csv(
        dataframe_columns, dataframe_types, "imputation", "unit", "basic_functionality"
    )
    bad_back_data = fxt_load_test_csv(
        dataframe_columns,
        dataframe_types,
        "imputation",
        "unit",
        "back_data_missing_column",
    )
    with pytest.raises(imputation.ValidationError):
        imputation.impute(test_dataframe, *params, back_data_df=bad_back_data)


@pytest.mark.dependency()
def test_back_data_contains_nulls(fxt_load_test_csv, fxt_spark_session):
    test_dataframe = fxt_load_test_csv(
        dataframe_columns, dataframe_types, "imputation", "unit", "basic_functionality"
    )
    bad_back_data = fxt_load_test_csv(
        dataframe_columns, dataframe_types, "imputation", "unit", "back_data_nulls"
    )

    with pytest.raises(imputation.ValidationError):
        imputation.impute(test_dataframe, *params, back_data_df=bad_back_data)


@pytest.mark.dependency()
def test_back_data_without_output_is_invalid(fxt_load_test_csv, fxt_spark_session):
    test_dataframe = fxt_load_test_csv(
        dataframe_columns, dataframe_types, "imputation", "unit", "basic_functionality"
    )
    bad_back_data = fxt_load_test_csv(
        [reference_col, period_col, strata_col, target_col, marker_col, auxiliary_col],
        dataframe_types,
        "imputation",
        "unit",
        "back_data_no_output",
    )

    with pytest.raises(imputation.ValidationError):
        imputation.impute(test_dataframe, *params, back_data_df=bad_back_data)


# --- Test if when the back data input has link cols and the main data input does not
# --- then the columns are ignored.


@pytest.mark.dependency()
def test_back_data_drops_link_cols_when_present(fxt_load_test_csv, fxt_spark_session):
    test_dataframe = fxt_load_test_csv(
        dataframe_columns, dataframe_types, "imputation", "unit", "basic_functionality"
    )

    back_data = fxt_load_test_csv(
        dataframe_columns,
        dataframe_types,
        "imputation",
        "unit",
        "back_data_with_link_cols",
    )

    ret_val = imputation.impute(test_dataframe, *params, back_data_df=back_data)

    assert ret_val.count() == 1


# --- Test when main data input has link cols and the back data input does not
# --- then columns aren't lost.


@pytest.mark.dependency()
def test_input_has_link_cols_and_back_data_does_not_have_link_cols(
    fxt_load_test_csv, fxt_spark_session
):
    test_dataframe = fxt_load_test_csv(
        dataframe_columns,
        dataframe_types,
        "imputation",
        "unit",
        "basic_functionality_with_link_cols",
    )

    back_data = fxt_load_test_csv(
        dataframe_columns,
        dataframe_types,
        "imputation",
        "unit",
        "back_data_without_link_cols",
    )

    imputation_kwargs = {
        "forward_link_col": forward_col,
        "backward_link_col": backward_col,
        "construction_link_col": construction_col,
    }

    ret_val = imputation.impute(
        test_dataframe, *params, **imputation_kwargs, back_data_df=back_data
    )

    assert ret_val.count() == 1


# --- Test if columns of the incorrect type are caught.


@pytest.mark.dependency()
def test_incorrect_column_types(fxt_load_test_csv):
    test_dataframe = fxt_load_test_csv(
        dataframe_columns,
        bad_dataframe_types,
        "imputation",
        "unit",
        "basic_functionality",
    )
    with pytest.raises(imputation.ValidationError):
        imputation.impute(test_dataframe, *params)


@pytest.mark.dependency()
def test_input_data_contains_nulls(fxt_load_test_csv, fxt_spark_session):
    test_dataframe = fxt_load_test_csv(
        dataframe_columns, dataframe_types, "imputation", "unit", "input_data_nulls"
    )

    with pytest.raises(imputation.ValidationError):
        imputation.impute(test_dataframe, *params)


# --- Test expected columns are in the output ---
@pytest.mark.dependency()
def test_dataframe_expected_columns(fxt_spark_session, fxt_load_test_csv):
    test_dataframe = fxt_load_test_csv(
        dataframe_columns, dataframe_types, "imputation", "unit", "basic_functionality"
    )
    ret_val = imputation.impute(
        test_dataframe,
        *default_params,
    )
    # perform action on the dataframe to trigger lazy evaluation
    ret_val.count()
    ret_cols = set(ret_val.columns)
    expected_cols = {
        reference_col,
        period_col,
        strata_col,
        auxiliary_col,
        forward_col,
        backward_col,
        construction_col,
        "imputed",
        "imputation_marker",
    }
    assert expected_cols == ret_cols


# --- Test Scenarios.


@pytest.mark.parametrize(
    "scenario_type, scenario, selection",
    sorted(
        scenarios(["dev_scenarios", "methodology_scenarios"]),
        key=lambda t: pathlib.Path(t[0], t[1]),
    ),
)
@pytest.mark.dependency(
    depends=[
        "test_dataframe_returned_as_expected",
        "test_params_not_string",
        "test_params_null",
        "test_params_missing_link_column",
        "test_dataframe_column_missing",
        "test_input_not_a_dataframe",
        "test_incorrect_column_types",
        "test_input_data_contains_nulls",
        "test_dataframe_expected_columns",
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

    if scenario.endswith("filtered"):
        imputation_kwargs["link_filter"] = "auxiliary != 3"

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


@pytest.mark.dependency(
    depends=[
        "test_back_data_not_a_dataframe",
        "test_back_data_missing_column",
        "test_back_data_contains_nulls",
        "test_back_data_without_output_is_invalid",
        "test_back_data_drops_link_cols_when_present",
        "test_input_has_link_cols_and_back_data_does_not_have_link_cols",
    ]
)
@pytest.mark.parametrize(
    "scenario_type, scenario, selection",
    sorted(
        scenarios(["dev_scenarios", "back_data_scenarios"]),
        key=lambda t: pathlib.Path(t[0], t[1]),
    ),
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
        target_col,
        auxiliary_col,
        output_col,
        marker_col,
        forward_col,
        backward_col,
    ]

    scenario_output = fxt_load_test_csv(
        back_data_cols,
        dataframe_types,
        "imputation",
        scenario_type,
        f"{scenario}_output",
    )

    select_back_data_cols = [
        col(reference_col),
        col(period_col),
        col(strata_col),
        col(auxiliary_col),
        col(output_col),
        col(output_col).alias(target_col),
        when(col(marker_col).isNotNull(), col(marker_col))
        .otherwise("BI")
        .alias(marker_col),
    ]

    min_period_df = scenario_output.selectExpr("min(period)")

    back_data_df = scenario_output.join(
        min_period_df, [col("period") == col("min(period)")]
    )

    input_df = test_dataframe.join(
        min_period_df, [col("period") == col("min(period)")], "leftanti"
    ).drop("min(period)")

    senario_expected_output = scenario_output.join(
        min_period_df, [col("period") == col("min(period)")], "leftanti"
    ).drop("min(period)")

    # We use imputation_kwargs to allow us to pass in the forward, backward
    # and construction link columns which are usually defaulted to None. This
    # means that we can autodetect when we should pass these.
    if forward_col in test_dataframe.columns:
        select_back_data_cols += [
            when(col(forward_col).isNotNull(), col(forward_col))
            .otherwise(1)
            .alias(forward_col),
            when(col(backward_col).isNotNull(), col(backward_col))
            .otherwise(1)
            .alias(backward_col),
        ]
        imputation_kwargs = {
            "forward_link_col": forward_col,
            "backward_link_col": backward_col,
            "construction_link_col": construction_col,
            "back_data_df": back_data_df.select(select_back_data_cols),
        }
    else:
        imputation_kwargs = {
            "back_data_df": back_data_df.select(select_back_data_cols),
        }

    ret_val = imputation.impute(input_df, *params, **imputation_kwargs)

    assert isinstance(ret_val, type(test_dataframe))
    sort_col_list = ["reference", "period"]
    assert_approx_df_equality(
        ret_val.sort(sort_col_list).select(selection),
        senario_expected_output.sort(sort_col_list).select(selection),
        0.0001,
        ignore_nullable=True,
    )
