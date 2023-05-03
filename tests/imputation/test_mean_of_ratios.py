import glob
import os
import pathlib

import pytest
import toml
from chispa.dataframe_comparer import assert_df_equality
from pyspark.sql.functions import bround, col
from pyspark.sql.types import BooleanType, DecimalType, LongType, StringType

from statistical_methods_library.imputation import impute, mean_of_ratios

auxiliary_col = "other"
backward_col = "backward"
forward_col = "forward"
marker_col = "marker"
output_col = "output"
period_col = "date"
reference_col = "identifier"
grouping_col = "group"
target_col = "question"
construction_col = "construction"
count_forward_col = "count_forward"
count_backward_col = "count_backward"
count_construction_col = "count_construction"
forward_growth_col = "growth_forward"
backward_growth_col = "growth_backward"
filtered_marker_col = "filtered_marker"
filtered_forward_col = "filtered_forward"
filtered_backward_col = "filtered_backward"
trimmed_forward_col = "trimmed_forward"
trimmed_backward_col = "trimmed_backward"

decimal_type = DecimalType(15, 6)

reference_type = StringType()
period_type = StringType()
grouping_type = StringType()
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
forward_growth_type = decimal_type
backward_growth_type = decimal_type
filtered_marker_type = BooleanType()
filtered_forward_type = BooleanType()
filtered_backward_type = BooleanType()
trimmed_forward_type = BooleanType()
trimmed_backward_type = BooleanType()

# Columns we expect in either our input or output test dataframes and their
# respective types
dataframe_columns = (
    reference_col,
    period_col,
    grouping_col,
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
    forward_growth_col,
    backward_growth_col,
    filtered_forward_col,
    filtered_backward_col,
    trimmed_forward_col,
    trimmed_backward_col,
)

dataframe_types = {
    reference_col: reference_type,
    period_col: period_type,
    grouping_col: grouping_type,
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
    forward_growth_col: forward_growth_type,
    backward_growth_col: backward_growth_type,
    filtered_marker_col: filtered_marker_type,
    filtered_forward_col: filtered_forward_type,
    filtered_backward_col: filtered_backward_type,
    trimmed_forward_col: trimmed_forward_type,
    trimmed_backward_col: trimmed_backward_type,
}

bad_dataframe_types = dataframe_types.copy()
bad_dataframe_types[target_col] = reference_type

# Params used when calling impute
params = {
    "reference_col": reference_col,
    "period_col": period_col,
    "grouping_col": grouping_col,
    "target_col": target_col,
    "auxiliary_col": auxiliary_col,
    "output_col": output_col,
    "marker_col": marker_col,
    "ratio_calculator": mean_of_ratios,
}

test_scenarios = []

scenario_path_prefix = pathlib.Path(
    "tests", "fixture_data", "imputation", "mean_of_ratios"
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


@pytest.mark.parametrize(
    "scenario_type, scenario",
    test_scenarios,
)
def test_calculations(fxt_load_test_csv, scenario_type, scenario):
    scenario_file_type = scenario_type.replace("back_data_", "")
    scenario_input = fxt_load_test_csv(
        dataframe_columns,
        dataframe_types,
        "imputation",
        "mean_of_ratios",
        scenario_file_type,
        f"{scenario}_input",
    )
    scenario_expected_output = fxt_load_test_csv(
        dataframe_columns,
        dataframe_types,
        "imputation",
        "mean_of_ratios",
        scenario_file_type,
        f"{scenario}_output",
    )
    imputation_kwargs = params.copy()

    with open("tests/imputation/mean_of_ratios.toml", "r") as f:
        new_toml_string = toml.load(f)
    if scenario in new_toml_string.keys():
        imputation_kwargs.update(new_toml_string[scenario])

    if scenario_type.startswith("back_data"):
        min_period_df = scenario_expected_output.selectExpr("min(" + period_col + ")")

        back_data_df = scenario_expected_output.join(
            min_period_df, [col(period_col) == col("min(" + period_col + ")")]
        )

        if "filtered" in scenario:
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

    # We need to drop our grouping and auxiliary columns from our output now
    # we've potentially set up our back data as these must not come out of
    # imputation.
    scenario_expected_output = scenario_expected_output.drop(
        grouping_col,
        auxiliary_col,
    )
    scenario_actual_output = impute(input_df=scenario_input, **imputation_kwargs)

    scenario_actual_output = scenario_actual_output.withColumn(
        output_col, bround(col(output_col), 6)
    ).withColumn(
        forward_col, bround(col(forward_col), 6)
    ).withColumn(
        backward_col, bround(col(backward_col).cast(decimal_type), 6)
    ).withColumn(
        construction_col, bround(col(construction_col).cast(decimal_type), 6)
    )

    if "link_columns" in scenario:
        scenario_expected_output = scenario_expected_output.drop(
            forward_growth_col,
            backward_growth_col,
        )
    else:
        scenario_actual_output = scenario_actual_output.withColumn(
            forward_growth_col, bround(col(forward_growth_col).cast(decimal_type), 6)
        ).withColumn(
            backward_growth_col, bround(col(backward_growth_col).cast(decimal_type), 6)
        )

    select_cols = list(set(dataframe_columns) & set(scenario_expected_output.columns))
    assert isinstance(scenario_actual_output, type(scenario_input))
    sort_col_list = [reference_col, period_col]
    assert_df_equality(
        scenario_actual_output.sort(sort_col_list).select(select_cols),
        scenario_expected_output.sort(sort_col_list).select(select_cols),
        ignore_nullable=True,
    )
