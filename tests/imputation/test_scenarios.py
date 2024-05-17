import glob
import os
import pathlib
from decimal import Decimal

import pytest
import toml
from pyspark.sql.functions import col

from statistical_methods_library import imputation
from tests.helpers import check_df_equality

scenario_path_prefix = pathlib.Path("tests", "fixture_data", "imputation")
ratio_calculators = ("mean_of_ratios", "ratio_of_means")
test_scenarios = []

# Collate list of test scenarios from ratio_calculators folders.
# Also for back data testing, duplicate all tests. Will be using
# first input period as a back data period.
for ratio_calculator in ratio_calculators:
    scenario_type = "methodology_scenarios"
    test_files = glob.glob(
        str(scenario_path_prefix / ratio_calculator / scenario_type / "*_input.csv")
    )
    test_list = sorted(
        (
            (
                ratio_calculator,
                scenario_type,
                os.path.basename(f).replace("_input.csv", ""),
            )
            for f in test_files
        ),
        key=lambda t: t[2],
    )
    test_scenarios += test_list
    test_scenarios += [
        (ratio_calculator, f"back_data_{scenario_type}", scenario_file)
        for ratio_calculator, scenario_type, scenario_file in test_list
    ]

toml_path_prefix = pathlib.Path("tests", "imputation")


@pytest.mark.parametrize(
    "ratio_calculator, scenario_type, scenario",
    test_scenarios,
)
def test_calculations(fxt_load_test_csv, ratio_calculator, scenario_type, scenario):
    # Default parameters and data types stored in default.toml, optional parameters
    # and overwrites stored in matching ratio calculator toml file.
    with open(toml_path_prefix / "default.toml", "r") as f:
        default_config = toml.load(f)

    with open(toml_path_prefix / f"{ratio_calculator}.toml", "r") as f:
        test_config = toml.load(f)

    scenarios = test_config.pop("scenarios", {})
    scenario_config = scenarios.get(scenario, {})
    fields = default_config["field_names"]
    fields.update(test_config.get("field_names", {}))
    fields.update(scenario_config.pop("field_names", {}))
    excluded_fields = default_config.get("excluded_fields", [])
    excluded_fields.extend(test_config.get("excluded_fields", []))
    excluded_fields.extend(scenario_config.pop("excluded_fields", []))
    imputation_kwargs = fields.copy()
    imputation_kwargs["forward_backward_ratio_calculator"] = getattr(
        imputation, ratio_calculator
    )
    if "back_data_" in scenario_type:
        starting_period_key = "back_data_starting_period"
    else:
        starting_period_key = "starting_period"
    starting_period = scenario_config.pop(
        starting_period_key,
        test_config.get(starting_period_key, default_config[starting_period_key]),
    )
    field_types = default_config["field_types"]
    field_types.update(test_config.get("field_types", {}))
    field_types.update(scenario_config.get("field_types", {}))
    imputation_kwargs.update(scenario_config)
    types = {fields[k]: v for k, v in field_types.items()}
    scenario_file_type = scenario_type.replace("back_data_", "")
    scenario_input = fxt_load_test_csv(
        fields.values(),
        types,
        "imputation",
        ratio_calculator,
        scenario_file_type,
        f"{scenario}_input",
    )
    scenario_expected_output = fxt_load_test_csv(
        fields.values(),
        types,
        "imputation",
        ratio_calculator,
        scenario_file_type,
        f"{scenario}_output",
    )

    if "weight" in imputation_kwargs:
        imputation_kwargs["weight"] = Decimal(imputation_kwargs["weight"])

    back_data_df = scenario_expected_output.filter(
        col(fields["period_col"]) < starting_period
    )
    if back_data_df.count() > 0:
        imputation_kwargs["back_data_df"] = back_data_df

    scenario_input = scenario_input.filter(col(fields["period_col"]) >= starting_period)

    scenario_expected_output = scenario_expected_output.filter(
        col(fields["period_col"]) >= starting_period
    )
    scenario_actual_output = imputation.impute(
        input_df=scenario_input, **imputation_kwargs
    )
    for field_name, field_type in scenario_actual_output.dtypes:
        if field_type.startswith("decimal"):
            scenario_actual_output = scenario_actual_output.withColumn(
                field_name, col(field_name).cast("decimal(15, 6)")
            )
    sort_cols = [
        fields["reference_col"],
        fields["period_col"],
        fields["grouping_col"],
    ]
    check_df_equality(
        actual=scenario_actual_output.sort(sort_cols),
        expected=scenario_expected_output.sort(sort_cols),
        keep_cols=sort_cols,
        exclude_cols=excluded_fields,
    )
