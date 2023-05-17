import glob
import os
import pathlib

import pytest
import toml
from chispa.dataframe_comparer import assert_df_equality
from decimal import Decimal
from pyspark.sql.functions import bround, col
from statistical_methods_library.imputation import impute, mean_of_ratios

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

@pytest.mark.parametrize(
    "scenario_type, scenario",
    test_scenarios,
)
def test_calculations(fxt_load_test_csv, scenario_type, scenario):
    with open("tests/imputation/mean_of_ratios.toml", "r") as f:
        test_params = toml.load(f)

    imputation_kwargs = test_params["field_names"].copy()
    imputation_kwargs["ratio_calculator"] = mean_of_ratios
    if "back_data_" in scenario_type:
        starting_period = test_params["back_data_starting_period"]
    else:
        starting_period = test_params["starting_period"]

    scenario_params = test_params.get(scenario, {}).copy()
    starting_period = scenario_params.pop("starting_period", starting_period)
    imputation_kwargs.update(scenario_params)

    fields = test_params["field_names"]
    types = {
        test_params["field_names"][k]: v
        for k,v in test_params["field_types"].items()
    }
    scenario_file_type = scenario_type.replace("back_data_", "")
    scenario_input = fxt_load_test_csv(
        fields.values(),
        types,
        "imputation",
        "mean_of_ratios",
        scenario_file_type,
        f"{scenario}_input",
    )
    scenario_expected_output = fxt_load_test_csv(
        fields.values(),
        types,
        "imputation",
        "mean_of_ratios",
        scenario_file_type,
        f"{scenario}_output",
    )

    if "weight" in imputation_kwargs:
        imputation_kwargs["weight"] = Decimal(imputation_kwargs["weight"])
    back_data_df = scenario_expected_output.filter(
        col(fields["period_col"]) < starting_period
    )

    imputation_kwargs["back_data_df"] = back_data_df

    scenario_input = scenario_input.filter(
        col(fields["period_col"]) >= starting_period
    )

    scenario_expected_output = scenario_expected_output.filter(
        col(fields["period_col"]) >= starting_period
    )

    # We need to drop our grouping and auxiliary columns from our output now
    # we've potentially set up our back data as these must not come out of
    # imputation.
    scenario_expected_output = scenario_expected_output.drop(
        fields["grouping_col"],
        fields["auxiliary_col"],
    )
    scenario_actual_output = impute(input_df=scenario_input, **imputation_kwargs)

    scenario_actual_output = (
        scenario_actual_output.withColumn(fields["output_col"], bround(col(fields["output_col"]), 6))
        .withColumn(fields["forward_col"], bround(col(fields["forward_col"]), 6))
        .withColumn(fields["backward_col"], bround(col(fields["backward_col"]), 6))
        .withColumn(
            fields["construction_col"], bround(col(fields["construction_col"]), 6)
        )
    )

    if "link_columns" not in scenario:
        scenario_actual_output = scenario_actual_output.withColumn(
            fields["forward_growth_col"], bround(col(fields["forward_growth_col"]), 6)
        ).withColumn(
            fields["backward_growth_col"], bround(col(fields["backward_growth_col"]), 6)
        )

    if "weight" in scenario:
        scenario_actual_output = (
            scenario_actual_output.withColumn(
                fields["unweighted_forward_link_col"], bround(col(fields["unweighted_forward_link_col"]), 6)
            )
            .withColumn(
                fields["unweighted_backward_link_col"], bround(col(fields["unweighted_backward_link_col"]), 6)
            )
            .withColumn(
                fields["unweighted_construction_link_col"],
                bround(col(fields["unweighted_construction_link_col"]), 6),
            )
        )

    select_cols = list(set(fields.values()) & set(scenario_expected_output.columns))
    sort_col_list = [fields["reference_col"], fields["period_col"], fields["grouping_col"]]
    assert_df_equality(
        scenario_actual_output.sort(sort_col_list).select(select_cols),
        scenario_expected_output.sort(sort_col_list).select(select_cols),
        ignore_nullable=True,
    )
