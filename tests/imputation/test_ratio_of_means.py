import glob
import os
import pathlib

import pytest
from chispa.dataframe_comparer import assert_df_equality
from pyspark.sql.functions import bround, col, lit
from pyspark.sql.types import DecimalType, LongType, StringType

from statistical_methods_library.imputation import ratio_of_means
from statistical_methods_library.utilities.exceptions import ValidationError

auxiliary_col = "other"
backward_col = "backward"
forward_col = "forward"
marker_col = "marker"
output_col = "output"
period_col = "date"
reference_col = "identifier"
strata_col = "group"
target_col = "question"
construction_col = "construction"
count_forward_col = "count_forward"
count_backward_col = "count_backward"
count_construction_col = "count_construction"
exclude_col = "exclude"

decimal_type = DecimalType(15, 6)

reference_type = StringType()
period_type = StringType()
strata_type = StringType()
target_type = decimal_type
auxiliary_type = decimal_type
min_accuracy = decimal_type
marker_type = StringType()
backward_type = decimal_type
forward_type = decimal_type
construction_type = decimal_type
count_forward_type = LongType()
count_backward_type = LongType()
count_construction_type = LongType()
exclude_type = StringType()

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
    count_forward_col,
    count_backward_col,
    count_construction_col,
    exclude_col,
)

dataframe_types = {
    reference_col: reference_type,
    period_col: period_type,
    strata_col: strata_type,
    target_col: target_type,
    auxiliary_col: auxiliary_type,
    output_col: min_accuracy,
    marker_col: marker_type,
    backward_col: backward_type,
    forward_col: forward_type,
    construction_col: construction_type,
    count_forward_col: count_forward_type,
    count_backward_col: count_backward_type,
    count_construction_col: count_construction_type,
    exclude_col: exclude_type,
}

bad_dataframe_types = dataframe_types.copy()
bad_dataframe_types[target_col] = reference_type

# Params used when calling impute
params = {
    "reference_col": reference_col,
    "period_col": period_col,
    "strata_col": strata_col,
    "target_col": target_col,
    "auxiliary_col": auxiliary_col,
    "output_col": output_col,
    "marker_col": marker_col,
}

test_scenarios = [
    ("unit", "ratio_calculation"),
    ("unit", "imputation_link_counts"),
]

scenario_path_prefix = pathlib.Path(
    "tests", "fixture_data", "imputation", "ratio_of_means"
)
for scenario_category in ("dev", "methodology"):
    scenario_type = f"{scenario_category}_scenarios"
    test_files = glob.glob(str(scenario_path_prefix / scenario_type / "*_input.csv"))
    test_scenarios += sorted(
        (
            (scenario_type, os.path.basename(f).replace("_input.csv", ""))
            for f in test_files
        ),
        key=lambda t: t[1],
    )

test_scenarios += [
    (f"back_data_{scenario_type}", scenario_file)
    for scenario_type, scenario_file in test_scenarios
    if scenario_type.split("_", 1)[0] in ("dev", "methodology")
]


# --- Test type validation on the input dataframe(s) ---


@pytest.mark.dependency()
def test_input_not_a_dataframe():
    with pytest.raises(TypeError):
        # noinspection PyTypeChecker
        ratio_of_means.impute(input_df="not_a_dataframe", **params)


# --- Test type validation on the back_data dataframe(s) ---


@pytest.mark.dependency()
def test_back_data_not_a_dataframe(fxt_load_test_csv):
    test_dataframe = fxt_load_test_csv(
        dataframe_columns,
        dataframe_types,
        "imputation",
        "ratio_of_means",
        "unit",
        "basic_functionality",
    )
    with pytest.raises(TypeError):
        # noinspection PyTypeChecker
        ratio_of_means.impute(
            input_df=test_dataframe, **params, back_data_df="not_a_dataframe"
        )


# --- Test if cols missing from input dataframe(s) ---


@pytest.mark.dependency()
def test_dataframe_column_missing(fxt_load_test_csv):
    test_dataframe = fxt_load_test_csv(
        dataframe_columns,
        dataframe_types,
        "imputation",
        "ratio_of_means",
        "unit",
        "basic_functionality",
    )
    bad_dataframe = test_dataframe.drop(strata_col)
    with pytest.raises(ValidationError):
        ratio_of_means.impute(input_df=bad_dataframe, **params)


# --- Test if dataframe has duplicate rows ---


@pytest.mark.dependency()
def test_dataframe_duplicate_rows(fxt_load_test_csv):
    test_dataframe = fxt_load_test_csv(
        dataframe_columns,
        dataframe_types,
        "imputation",
        "ratio_of_means",
        "unit",
        "duplicate_rows",
    )
    with pytest.raises(ValidationError):
        ratio_of_means.impute(input_df=test_dataframe, **params)


# --- Test if target missing from input dataframe(s) ---


@pytest.mark.dependency()
def test_dataframe_target_missing(fxt_load_test_csv):
    test_dataframe = fxt_load_test_csv(
        dataframe_columns,
        dataframe_types,
        "imputation",
        "ratio_of_means",
        "unit",
        "basic_functionality",
    )
    bad_dataframe = test_dataframe.drop(target_col)
    with pytest.raises(ValidationError):
        ratio_of_means.impute(input_df=bad_dataframe, **params)


# --- Test if params null ---


@pytest.mark.dependency()
def test_params_none(fxt_load_test_csv):
    test_dataframe = fxt_load_test_csv(
        dataframe_columns,
        dataframe_types,
        "imputation",
        "ratio_of_means",
        "unit",
        "basic_functionality",
    )
    bad_params = params.copy()
    bad_params["target_col"] = None
    with pytest.raises(TypeError):
        ratio_of_means.impute(input_df=test_dataframe, **bad_params)


@pytest.mark.dependency()
def test_params_empty_string(fxt_load_test_csv):
    test_dataframe = fxt_load_test_csv(
        dataframe_columns,
        dataframe_types,
        "imputation",
        "ratio_of_means",
        "unit",
        "basic_functionality",
    )
    bad_params = params.copy()
    bad_params["target_col"] = ""
    with pytest.raises(ValueError):
        ratio_of_means.impute(input_df=test_dataframe, **bad_params)


@pytest.mark.dependency()
def test_params_missing_link_column(fxt_load_test_csv):
    test_dataframe = fxt_load_test_csv(
        dataframe_columns,
        dataframe_types,
        "imputation",
        "ratio_of_means",
        "unit",
        "basic_functionality",
    )
    with pytest.raises(TypeError):
        ratio_of_means.impute(
            input_df=test_dataframe, **params, construction_link_col=construction_col
        )


@pytest.mark.dependency()
def test_params_not_string(fxt_load_test_csv):
    test_dataframe = fxt_load_test_csv(
        dataframe_columns,
        dataframe_types,
        "imputation",
        "ratio_of_means",
        "unit",
        "basic_functionality",
    )
    bad_params = params.copy()
    bad_params["reference_col"] = 23
    with pytest.raises(TypeError):
        ratio_of_means.impute(input_df=test_dataframe, **bad_params)


# --- Test if output contents are as expected, both new columns and data ---


@pytest.mark.dependency()
def test_dataframe_returned_as_expected(fxt_spark_session, fxt_load_test_csv):
    test_dataframe = fxt_load_test_csv(
        dataframe_columns,
        dataframe_types,
        "imputation",
        "ratio_of_means",
        "unit",
        "basic_functionality",
    )
    # Make sure that no extra columns pass through.
    test_dataframe = test_dataframe.withColumn("bonus_column", lit(0))
    ret_val = ratio_of_means.impute(input_df=test_dataframe, **params)
    assert isinstance(ret_val, type(test_dataframe))
    ret_cols = ret_val.columns
    assert "bonus_column" not in ret_cols


# --- Test that when provided back data does not match input schema then fails ---
@pytest.mark.dependency()
def test_back_data_missing_column(fxt_load_test_csv, fxt_spark_session):
    test_dataframe = fxt_load_test_csv(
        dataframe_columns,
        dataframe_types,
        "imputation",
        "ratio_of_means",
        "unit",
        "basic_functionality",
    )
    bad_back_data = fxt_load_test_csv(
        dataframe_columns,
        dataframe_types,
        "imputation",
        "ratio_of_means",
        "unit",
        "back_data_missing_column",
    )
    with pytest.raises(ValidationError):
        ratio_of_means.impute(
            input_df=test_dataframe, **params, back_data_df=bad_back_data
        )


@pytest.mark.dependency()
def test_back_data_contains_nulls(fxt_load_test_csv, fxt_spark_session):
    test_dataframe = fxt_load_test_csv(
        dataframe_columns,
        dataframe_types,
        "imputation",
        "ratio_of_means",
        "unit",
        "basic_functionality",
    )
    bad_back_data = fxt_load_test_csv(
        dataframe_columns,
        dataframe_types,
        "imputation",
        "ratio_of_means",
        "unit",
        "back_data_nulls",
    )

    with pytest.raises(ValidationError):
        ratio_of_means.impute(
            input_df=test_dataframe, **params, back_data_df=bad_back_data
        )


@pytest.mark.dependency()
def test_back_data_without_output_is_invalid(fxt_load_test_csv, fxt_spark_session):
    test_dataframe = fxt_load_test_csv(
        dataframe_columns,
        dataframe_types,
        "imputation",
        "ratio_of_means",
        "unit",
        "basic_functionality",
    )
    bad_back_data = fxt_load_test_csv(
        [reference_col, period_col, strata_col, target_col, marker_col, auxiliary_col],
        dataframe_types,
        "imputation",
        "ratio_of_means",
        "unit",
        "back_data_no_output",
    )

    with pytest.raises(ValidationError):
        ratio_of_means.impute(
            input_df=test_dataframe, **params, back_data_df=bad_back_data
        )


# Test if when the back data input has link cols and the main data input does not
# then the columns are ignored.
@pytest.mark.dependency()
def test_back_data_drops_link_cols_when_present(fxt_load_test_csv, fxt_spark_session):
    test_dataframe = fxt_load_test_csv(
        dataframe_columns,
        dataframe_types,
        "imputation",
        "ratio_of_means",
        "unit",
        "basic_functionality",
    )

    back_data = fxt_load_test_csv(
        dataframe_columns,
        dataframe_types,
        "imputation",
        "ratio_of_means",
        "unit",
        "back_data_with_link_cols",
    )

    ret_val = ratio_of_means.impute(
        input_df=test_dataframe, **params, back_data_df=back_data
    )

    assert ret_val.count() == 1


# Test when main data input has link cols and the back data input does not
# then columns aren't lost.
@pytest.mark.dependency()
def test_input_has_link_cols_and_back_data_does_not_have_link_cols(
    fxt_load_test_csv, fxt_spark_session
):
    test_dataframe = fxt_load_test_csv(
        dataframe_columns,
        dataframe_types,
        "imputation",
        "ratio_of_means",
        "unit",
        "basic_functionality_with_link_cols",
    )

    back_data = fxt_load_test_csv(
        dataframe_columns,
        dataframe_types,
        "imputation",
        "ratio_of_means",
        "unit",
        "back_data_without_link_cols",
    )

    imputation_kwargs = params.copy()
    imputation_kwargs.update(
        {
            "forward_link_col": forward_col,
            "backward_link_col": backward_col,
            "construction_link_col": construction_col,
            "input_df": test_dataframe,
            "back_data_df": back_data,
        }
    )

    ret_val = ratio_of_means.impute(**imputation_kwargs)

    assert ret_val.count() == 1


# Test if columns of the incorrect type are caught.
@pytest.mark.dependency()
def test_incorrect_column_types(fxt_load_test_csv):
    test_dataframe = fxt_load_test_csv(
        dataframe_columns,
        bad_dataframe_types,
        "imputation",
        "ratio_of_means",
        "unit",
        "basic_functionality",
    )
    with pytest.raises(ValidationError):
        ratio_of_means.impute(input_df=test_dataframe, **params)


@pytest.mark.dependency()
def test_input_data_contains_nulls(fxt_load_test_csv, fxt_spark_session):
    test_dataframe = fxt_load_test_csv(
        dataframe_columns,
        dataframe_types,
        "imputation",
        "ratio_of_means",
        "unit",
        "input_data_nulls",
    )

    with pytest.raises(ValidationError):
        ratio_of_means.impute(input_df=test_dataframe, **params)


# --- Test Scenarios.


@pytest.mark.parametrize(
    "scenario_type, scenario",
    test_scenarios,
)
@pytest.mark.dependency(
    depends=[
        "test_dataframe_returned_as_expected",
        "test_params_not_string",
        "test_params_none",
        "test_params_empty_string",
        "test_params_missing_link_column",
        "test_dataframe_column_missing",
        "test_input_not_a_dataframe",
        "test_incorrect_column_types",
        "test_input_data_contains_nulls",
        "test_back_data_not_a_dataframe",
        "test_back_data_missing_column",
        "test_back_data_contains_nulls",
        "test_back_data_without_output_is_invalid",
        "test_back_data_drops_link_cols_when_present",
        "test_input_has_link_cols_and_back_data_does_not_have_link_cols",
    ]
)
def test_calculations(fxt_load_test_csv, scenario_type, scenario):
    scenario_file_type = scenario_type.replace("back_data_", "")
    scenario_input = fxt_load_test_csv(
        dataframe_columns,
        dataframe_types,
        "imputation",
        "ratio_of_means",
        scenario_file_type,
        f"{scenario}_input",
    )
    scenario_expected_output = fxt_load_test_csv(
        dataframe_columns,
        dataframe_types,
        "imputation",
        "ratio_of_means",
        scenario_file_type,
        f"{scenario}_output",
    )
    imputation_kwargs = params.copy()
    if forward_col in scenario_input.columns:
        imputation_kwargs.update(
            {
                "forward_link_col": forward_col,
                "backward_link_col": backward_col,
                "construction_link_col": construction_col,
            }
        )

    if scenario.endswith("filtered"):
        if "dev" in scenario_type:
            imputation_kwargs["link_filter"] = "(" + exclude_col + ' == "N")'
        else:
            imputation_kwargs["link_filter"] = (
                "(" + auxiliary_col + " != 71) and (" + target_col + " < 100000)"
            )

    if scenario_type.startswith("back_data"):
        min_period_df = scenario_expected_output.selectExpr("min(" + period_col + ")")

        back_data_df = scenario_expected_output.join(
            min_period_df, [col(period_col) == col("min(" + period_col + ")")]
        )

        if scenario.endswith("filtered"):
            if "dev" in scenario_type:
                back_data_df = back_data_df.join(
                    scenario_input.select(reference_col, period_col, exclude_col),
                    [reference_col, period_col],
                )
            else:
                back_data_df = back_data_df.withColumn(target_col, col(output_col))

        imputation_kwargs["back_data_df"] = back_data_df

        scenario_input = scenario_input.join(
            min_period_df,
            [col(period_col) == col("min(" + period_col + ")")],
            "leftanti",
        ).drop("min(" + period_col + ")")

        scenario_expected_output = scenario_expected_output.join(
            min_period_df,
            [col(period_col) == col("min(" + period_col + ")")],
            "leftanti",
        ).drop("min(" + period_col + ")")

    # We need to drop our strata and auxiliary columns from our output now
    # we've potentially set up our back data as these must not come out of
    # imputation.
    scenario_expected_output = scenario_expected_output.drop(
        strata_col,
        auxiliary_col,
    )
    scenario_actual_output = ratio_of_means.impute(
        input_df=scenario_input, **imputation_kwargs
    )
    scenario_actual_output = scenario_actual_output.withColumn(
        output_col, bround(col(output_col), 6)
    )
    scenario_actual_output = scenario_actual_output.withColumn(
        forward_col, bround(col(forward_col), 6)
    )
    scenario_actual_output = scenario_actual_output.withColumn(
        backward_col, bround(col(backward_col).cast(decimal_type), 6)
    )
    scenario_actual_output = scenario_actual_output.withColumn(
        construction_col, bround(col(construction_col).cast(decimal_type), 6)
    )
    select_cols = list(set(dataframe_columns) & set(scenario_expected_output.columns))
    assert isinstance(scenario_actual_output, type(scenario_input))
    sort_col_list = [reference_col, period_col]
    assert_df_equality(
        scenario_actual_output.sort(sort_col_list).select(select_cols),
        scenario_expected_output.sort(sort_col_list).select(select_cols),
        ignore_nullable=True,
    )
