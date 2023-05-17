import glob
import os
import pathlib

import pytest
import toml
from chispa.dataframe_comparer import assert_df_equality
from decimal import Decimal
from pyspark.sql.functions import bround, col
from statistical_methods_library import imputation

scenario_path_prefix = pathlib.Path(
    "tests", "fixture_data", "imputation"
)
test_scenarios = []
for ratio_calculator in ("mean_of_ratios", "ratio_of_means"):
    scenario_type = "methodology_scenarios"
    test_files = glob.glob(str(scenario_path_prefix / ratio_calculator / scenario_type / "*_input.csv"))
    test_scenarios += sorted(
        (
            (ratio_calculator, scenario_type, os.path.basename(f).replace("_input.csv", ""))
            for f in test_files
        ),
        key=lambda t: t[2],
    )

test_scenarios += [
    (ratio_calculator, f"back_data_{scenario_type}", scenario_file)
    for ratio_calculator, scenario_type, scenario_file in test_scenarios
]
toml_path_prefix = pathlib.Path("tests", "imputation")
with open(toml_path_prefix / "default.toml", "r") as f:
    default_params = toml.load(f)

@pytest.mark.parametrize(
    "ratio_calculator, scenario_type, scenario",
    test_scenarios,
)
def test_calculations(fxt_load_test_csv, ratio_calculator, scenario_type, scenario):
    test_params = default_params.copy()
    with open(toml_path_prefix / f"{ratio_calculator}.toml", "r") as f:
        test_params.update(toml.load(f))

    imputation_kwargs = test_params["field_names"].copy()
    imputation_kwargs["ratio_calculator"] = getattr(imputation, ratio_calculator)
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

    # We need to drop our auxiliary column from our output now
    # we've potentially set up our back data as this must not come out of
    # imputation.
    scenario_expected_output = scenario_expected_output.drop(
        fields["auxiliary_col"],
    )
    scenario_actual_output = imputation.impute(input_df=scenario_input, **imputation_kwargs)

    for field_name, field_type in scenario_actual_output.dtypes:
        if field_type.startswith("decimal"):
            scenario_actual_output = scenario_actual_output.withColumn(field_name, bround(field_name, 6))

    select_cols = list(set(fields.values()) & set(scenario_expected_output.columns))
    sort_col_list = [fields["reference_col"], fields["period_col"], fields["grouping_col"]]
    assert_df_equality(
        scenario_actual_output.sort(sort_col_list).select(select_cols),
        scenario_expected_output.sort(sort_col_list).select(select_cols),
        ignore_nullable=True,
    )
