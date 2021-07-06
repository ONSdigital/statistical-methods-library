from chispa import assert_approx_df_equality

from statistical_methods_library import estimation

reference_col = "ref"
period_col = "period"
strata_col = "strata"
sample_col = "sample_mkr"
death_col = "death_mkr"
h_col = "H"
auxiliary_col = "aux"
calibration_group_col = "cal_group"
design_weight = "design_weight"
calibration_weight = "calibration_weight"

dataframe_columns = (
    reference_col,
    period_col,
    strata_col,
    sample_col,
    death_col,
    h_col,
    auxiliary_col,
    calibration_group_col,
)

dataframe_types = {
    reference_col: "string",
    period_col: "string",
    strata_col: "string",
    sample_col: "double",
    death_col: "double",
    h_col: "double",
    auxiliary_col: "double",
    calibration_group_col: "string",
}

dataframe_columns_other = (
    period_col,
    strata_col,
    design_weight,
    calibration_weight,
)

dataframe_types_other = {
    period_col: "string",
    strata_col: "string",
    design_weight: "double",
    calibration_weight: "double",
}


def test_calculations(fxt_load_test_csv):
    test_dataframe = fxt_load_test_csv(
        dataframe_columns,
        dataframe_types,
        "estimation",
        "unit",
        "estimation_input",
    )

    exp_val = fxt_load_test_csv(
        dataframe_columns_other,
        dataframe_types_other,
        "estimation",
        "unit",
        "estimation_output",
    )

    ret_val = estimation.estimate(
        test_dataframe, period_col, strata_col, sample_col,
        death_col, h_col, auxiliary_col, calibration_group_col
    )

    assert isinstance(ret_val, type(test_dataframe))
    sort_col_list = ["period", "strata"]
    assert_approx_df_equality(
        ret_val.sort(sort_col_list),
        exp_val.sort(sort_col_list)
    )
